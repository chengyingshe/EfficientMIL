import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score, hamming_loss
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
import logging
import time
from typing import Optional, Dict, Any
import efficientmil_common
import argparse
import os
import copy

# Import common utilities
from common_utils import (
    setup_logging, log_performance, set_random_seed, apply_sparse_init,
    multi_label_roc, optimal_thresh, save_model_checkpoint, load_checkpoint,
    get_current_score, print_epoch_info, print_final_results,
    parse_model_params, log_model_info
)
from data import PatchFeaturesDataset
from torch.utils.tensorboard import SummaryWriter


def compute_l2_loss(model: nn.Module) -> torch.Tensor:
    """Compute L2 regularization loss for model parameters."""
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for param in model.parameters():
        if param.requires_grad:
            l2_loss += torch.norm(param, p=2) ** 2
    return l2_loss


def save_checkpoint(save_path: str,
                   model: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   best_score: float,
                   patience_counter: int,
                   fold: int,
                   thresholds_optimal: Optional[np.ndarray] = None,
                   extra: Optional[Dict[str, Any]] = None) -> None:
    """Save checkpoint including optimizer and scheduler state for resume."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'epoch': epoch,
        'best_score': best_score,
        'patience_counter': patience_counter,
        'fold': fold,
        'thresholds_optimal': thresholds_optimal
    }
    if extra:
        ckpt.update(extra)
    torch.save(ckpt, save_path)


def load_checkpoint_for_resume(args, fold: int, model: nn.Module, 
                              optimizer: Optional[torch.optim.Optimizer] = None,
                              scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
    """
    Load checkpoint for resuming training.
    Returns: dict with keys start_epoch, best_score, patience_counter, thresholds_optimal
    """
    state = {
        'start_epoch': 1,
        'best_score': 0.0,
        'patience_counter': 0,
        'thresholds_optimal': None
    }
    
    if not args.resume:
        return state
    
    # Determine checkpoint path
    if args.resume_path:
        ckpt_path = args.resume_path
    else:
        save_path = os.path.join(args.save_dir, 'weights')
        ckpt_path = os.path.join(save_path, f'fold_{fold}', 'last.pth')
    
    if not os.path.exists(ckpt_path):
        logging.warning(f"Checkpoint not found at {ckpt_path}. Starting from scratch.")
        return state
    
    logging.info(f"Loading checkpoint from {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=torch.device(args.device))
        
        # Load model state
        model.load_state_dict(ckpt['model_state_dict'])
        
        # Load optimizer state
        if optimizer is not None and ckpt.get('optimizer_state_dict') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        
        # Load scheduler state
        if scheduler is not None and ckpt.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        
        # Load training state
        state['start_epoch'] = int(ckpt.get('epoch', 0)) + 1
        state['best_score'] = float(ckpt.get('best_score', 0.0))
        state['patience_counter'] = int(ckpt.get('patience_counter', 0))
        state['thresholds_optimal'] = ckpt.get('thresholds_optimal', None)
        
        logging.info(f"Resumed from epoch {state['start_epoch']}, best_score: {state['best_score']:.4f}")
        
    except Exception as e:
        logging.error(f"Error loading checkpoint: {e}")
        logging.warning("Starting from scratch.")
        
    return state


def train(args, train_loader, milnet, criterion, optimizer, tb_writer=None, epoch=None, fold=None):
    milnet.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    device = torch.device(args.device)
    
    epoch_start_time = time.time()
    
    # For AUC calculation
    train_prob_list = []
    train_label_list = []
    
    train_pbar = tqdm(train_loader, desc='Training')
    
    for batch in train_pbar:
        optimizer.zero_grad()
        # Extract features and labels from batch
        patch_features = batch['patch_features'].to(device)
        labels = batch['label'].to(device)
        
        batch_loss = 0.0
        batch_correct = 0
        
        # Process each bag in the batch (typically batch size is 1 for MIL)
        for j in range(patch_features.size(0)):
            bag_feats = patch_features[j]
            
            # Convert single label to proper format for BCEWithLogitsLoss
            if args.num_classes == 1:
                bag_label = labels[j].float().unsqueeze(0).unsqueeze(0)
                true_label = int(labels[j].item())
            else:
                bag_label = torch.zeros(1, args.num_classes, device=device)
                if labels[j] < args.num_classes:
                    bag_label[0, labels[j]] = 1
                true_label = int(labels[j].item())
            
            bag_feats = bag_feats.view(-1, args.feats_size)
            ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
            max_prediction, _ = torch.max(ins_prediction, 0)        
            
            # Compute main losses
            bag_loss = criterion(bag_prediction.view(1, -1), bag_label)
            max_loss = criterion(max_prediction.view(1, -1), bag_label)
            main_loss = 0.5 * bag_loss + 0.5 * max_loss
            
            # Add L2 regularization
            l2_loss = compute_l2_loss(milnet)
            total_loss = main_loss + args.l2_loss_weight * l2_loss
            
            batch_loss += total_loss.item()
            
            # Calculate accuracy
            if args.num_classes == 1:
                # Binary classification
                pred_prob = torch.sigmoid(bag_prediction).item()
                predicted = 1 if pred_prob >= 0.5 else 0
                train_prob_list.append([1-pred_prob, pred_prob])
            else:
                # Multi-class classification
                probs = torch.softmax(bag_prediction, dim=-1)
                predicted = torch.argmax(probs, dim=-1).item()
                train_prob_list.append(probs.squeeze().detach().cpu().numpy())
            
            batch_correct += (predicted == true_label)
            train_label_list.append(true_label)
            
            total_loss.backward()
        
        train_loss += batch_loss
        train_correct += batch_correct
        train_total += patch_features.size(0)
        optimizer.step()
        
        # Compute running metrics
        running_acc = train_correct / train_total
        running_train_auc = float('nan')
        
        # Compute running AUC
        try:
            if len(train_prob_list) > 0 and len(train_label_list) > 0:
                y_score_tr = np.array(train_prob_list)
                y_true_tr = np.array(train_label_list)
                unique_classes = np.unique(y_true_tr)
                
                if len(unique_classes) >= 2:
                    if args.num_classes <= 2:
                        # Binary classification
                        if y_score_tr.shape[1] >= 2:
                            y_score_binary = y_score_tr[:, -1]  # Use positive class probability
                            running_train_auc = roc_auc_score(y_true_tr, y_score_binary)
                    else:
                        # Multi-class classification
                        try:
                            aucs = []
                            for class_idx in range(args.num_classes):
                                if class_idx in unique_classes:
                                    y_binary = (y_true_tr == class_idx).astype(int)
                                    if len(np.unique(y_binary)) >= 2:
                                        class_auc = roc_auc_score(y_binary, y_score_tr[:, class_idx])
                                        aucs.append(class_auc)
                            if len(aucs) >= 2:
                                running_train_auc = np.mean(aucs)
                        except Exception:
                            running_train_auc = float('nan')
        except Exception:
            running_train_auc = float('nan')
        
        # Update progress bar
        loss_dict = {
            'loss': f"{train_loss/len(train_pbar):.4f}",
            'acc': f"{running_acc:.4f}",
            'auc': f"{running_train_auc:.4f}" if not np.isnan(running_train_auc) else "nan"
        }
        
        train_pbar.set_postfix(loss_dict)
    
    epoch_time = time.time() - epoch_start_time
    avg_loss = train_loss / len(train_loader)
    train_accuracy = train_correct / train_total
    
    # Final AUC calculation
    train_auc = float('nan')
    try:
        if len(train_prob_list) > 0 and len(train_label_list) > 0:
            y_score = np.array(train_prob_list)
            y_true = np.array(train_label_list)
            unique_classes = np.unique(y_true)
            
            if len(unique_classes) >= 2:
                if args.num_classes <= 2:
                    if y_score.shape[1] >= 2:
                        y_score_binary = y_score[:, -1]
                        train_auc = roc_auc_score(y_true, y_score_binary)
                else:
                    try:
                        aucs = []
                        for class_idx in range(args.num_classes):
                            if class_idx in unique_classes:
                                y_binary = (y_true == class_idx).astype(int)
                                if len(np.unique(y_binary)) >= 2:
                                    class_auc = roc_auc_score(y_binary, y_score[:, class_idx])
                                    aucs.append(class_auc)
                        if len(aucs) >= 2:
                            train_auc = np.mean(aucs)
                    except Exception:
                        train_auc = float('nan')
    except Exception:
        train_auc = float('nan')
    
    # Log to TensorBoard if available
    if tb_writer and epoch is not None and fold is not None:
        tag = f"fold_{fold}"
        tb_writer.add_scalar(f"{tag}/train_loss", avg_loss, epoch)
        tb_writer.add_scalar(f"{tag}/train_acc", train_accuracy, epoch)
        if not np.isnan(train_auc):
            tb_writer.add_scalar(f"{tag}/train_auc", train_auc, epoch)
        tb_writer.add_scalar(f"{tag}/train_time", epoch_time, epoch)
    
    return avg_loss, train_accuracy, train_auc

def test(args, test_loader, milnet, criterion, thresholds=None, return_predictions=False, tb_writer=None, epoch=None, fold=None):
    milnet.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    device = torch.device(args.device)
    
    test_start_time = time.time()
    
    # For AUC calculation
    val_prob_list = []
    val_label_list = []
    test_labels = []
    test_predictions = []
    
    with torch.no_grad():
        val_pbar = tqdm(test_loader, desc='Validating')
        
        for batch in val_pbar:
            # Extract features and labels from batch
            patch_features = batch['patch_features'].to(device)
            labels = batch['label'].to(device)
            
            batch_loss = 0.0
            batch_correct = 0
            
            # Process each bag in the batch (typically batch size is 1 for MIL)
            for j in range(patch_features.size(0)):
                bag_feats = patch_features[j]
                
                # Convert single label to proper format for BCEWithLogitsLoss
                if args.num_classes == 1:
                    bag_label = labels[j].float().unsqueeze(0).unsqueeze(0)
                    true_label = int(labels[j].item())
                else:
                    bag_label = torch.zeros(1, args.num_classes, device=device)
                    if labels[j] < args.num_classes:
                        bag_label[0, labels[j]] = 1
                    true_label = int(labels[j].item())
                
                bag_feats = bag_feats.view(-1, args.feats_size)
                
                ins_prediction, bag_prediction, _, _ = milnet(bag_feats)
                max_prediction, _ = torch.max(ins_prediction, 0)  
                
                # Compute main losses (without L2 for validation)
                bag_loss = criterion(bag_prediction.view(1, -1), bag_label)
                max_loss = criterion(max_prediction.view(1, -1), bag_label)
                main_loss = 0.5 * bag_loss + 0.5 * max_loss
                
                batch_loss += main_loss.item()
                
                # Calculate accuracy and collect predictions
                if args.num_classes == 1:
                    # Binary classification
                    pred_prob = torch.sigmoid(bag_prediction).item()
                    predicted = 1 if pred_prob >= 0.5 else 0
                    val_prob_list.append([1-pred_prob, pred_prob])
                    # Store for legacy compatibility
                    test_predictions.append(pred_prob)
                else:
                    # Multi-class classification
                    probs = torch.softmax(bag_prediction, dim=-1)
                    predicted = torch.argmax(probs, dim=-1).item()
                    val_prob_list.append(probs.squeeze().detach().cpu().numpy())
                    # Store for legacy compatibility
                    test_predictions.append(probs.squeeze().detach().cpu().numpy())
                
                batch_correct += (predicted == true_label)
                val_label_list.append(true_label)
                
                # Store for legacy compatibility
                test_labels.append(bag_label.squeeze().cpu().numpy().astype(int))
            
            val_loss += batch_loss
            val_correct += batch_correct
            val_total += patch_features.size(0)
            
            # Compute running metrics
            running_acc = val_correct / val_total
            running_val_auc = float('nan')
            
            # Compute running AUC
            try:
                if len(val_prob_list) > 0 and len(val_label_list) > 0:
                    y_score_val = np.array(val_prob_list)
                    y_true_val = np.array(val_label_list)
                    unique_classes = np.unique(y_true_val)
                    
                    if len(unique_classes) >= 2:
                        if args.num_classes <= 2:
                            # Binary classification
                            if y_score_val.shape[1] >= 2:
                                y_score_binary = y_score_val[:, -1]  # Use positive class probability
                                running_val_auc = roc_auc_score(y_true_val, y_score_binary)
                        else:
                            # Multi-class classification
                            try:
                                aucs = []
                                for class_idx in range(args.num_classes):
                                    if class_idx in unique_classes:
                                        y_binary = (y_true_val == class_idx).astype(int)
                                        if len(np.unique(y_binary)) >= 2:
                                            class_auc = roc_auc_score(y_binary, y_score_val[:, class_idx])
                                            aucs.append(class_auc)
                                if len(aucs) >= 2:
                                    running_val_auc = np.mean(aucs)
                            except Exception:
                                running_val_auc = float('nan')
            except Exception:
                running_val_auc = float('nan')
            
            # Update progress bar
            loss_dict = {
                'loss': f"{val_loss/len(val_pbar):.4f}",
                'acc': f"{running_acc:.4f}",
                'auc': f"{running_val_auc:.4f}" if not np.isnan(running_val_auc) else "nan"
            }
            val_pbar.set_postfix(loss_dict)
    
    test_time = time.time() - test_start_time
    avg_loss = val_loss / len(test_loader)
    val_accuracy = val_correct / val_total
    
    # Final AUC calculation
    val_auc = float('nan')
    try:
        if len(val_prob_list) > 0 and len(val_label_list) > 0:
            y_score = np.array(val_prob_list)
            y_true = np.array(val_label_list)
            unique_classes = np.unique(y_true)
            
            if len(unique_classes) >= 2:
                if args.num_classes <= 2:
                    if y_score.shape[1] >= 2:
                        y_score_binary = y_score[:, -1]
                        val_auc = roc_auc_score(y_true, y_score_binary)
                else:
                    try:
                        aucs = []
                        for class_idx in range(args.num_classes):
                            if class_idx in unique_classes:
                                y_binary = (y_true == class_idx).astype(int)
                                if len(np.unique(y_binary)) >= 2:
                                    class_auc = roc_auc_score(y_binary, y_score[:, class_idx])
                                    aucs.append(class_auc)
                        if len(aucs) >= 2:
                            val_auc = np.mean(aucs)
                    except Exception:
                        val_auc = float('nan')
    except Exception:
        val_auc = float('nan')
    
    # Legacy compatibility: compute thresholds using original method
    test_labels = np.array(test_labels)
    test_predictions = np.array(test_predictions)
    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes, pos_label=1)
    if thresholds: 
        thresholds_optimal = thresholds
    else:
        # Use computed AUC if available
        if not np.isnan(val_auc):
            if isinstance(auc_value, list):
                auc_value = [val_auc] * len(auc_value)
            else:
                auc_value = val_auc
    
    # Compute final accuracy using thresholds
    if args.num_classes == 1:
        class_prediction_bag = copy.deepcopy(test_predictions)
        class_prediction_bag[test_predictions >= thresholds_optimal[0]] = 1
        class_prediction_bag[test_predictions < thresholds_optimal[0]] = 0
        test_predictions = class_prediction_bag
        test_labels = np.squeeze(test_labels)
    else:        
        for i in range(args.num_classes):
            class_prediction_bag = copy.deepcopy(test_predictions[:, i])
            class_prediction_bag[test_predictions[:, i] >= thresholds_optimal[i]] = 1
            class_prediction_bag[test_predictions[:, i] < thresholds_optimal[i]] = 0
            test_predictions[:, i] = class_prediction_bag
    
    bag_score = 0
    for i in range(0, len(test_labels)):
        bag_score = np.array_equal(test_labels[i], test_predictions[i]) + bag_score         
    avg_score = bag_score / len(test_labels)
    
    # Log to TensorBoard if available
    if tb_writer and epoch is not None and fold is not None:
        tag = f"fold_{fold}"
        tb_writer.add_scalar(f"{tag}/val_loss", avg_loss, epoch)
        tb_writer.add_scalar(f"{tag}/val_acc", val_accuracy, epoch)
        
        # Handle AUC logging (can be single value or list)
        if isinstance(auc_value, list):
            for i, auc in enumerate(auc_value):
                tb_writer.add_scalar(f"{tag}/val_auc_class_{i}", auc, epoch)
            # Also log average AUC
            tb_writer.add_scalar(f"{tag}/val_auc_avg", sum(auc_value) / len(auc_value), epoch)
        else:
            tb_writer.add_scalar(f"{tag}/val_auc", auc_value, epoch)
        
        tb_writer.add_scalar(f"{tag}/val_time", test_time, epoch)
    
    if return_predictions:
        return avg_loss, avg_score, auc_value, thresholds_optimal, test_predictions, test_labels
    return avg_loss, avg_score, auc_value, thresholds_optimal, val_accuracy, val_auc

def main():
    parser = argparse.ArgumentParser(description='Train model on dataset')
    parser.add_argument('--num_classes', default=2, type=int, help='Number of output classes')
    parser.add_argument('--feats_size', default=1536, type=int, help='Dimension of the feature size')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of workers')
    parser.add_argument('--lr', default=2e-4, type=float, help='Initial learning rate')
    parser.add_argument('--num_epochs', default=50, type=int, help='Number of total training epochs')
    parser.add_argument('--patience', default=5, type=int, help='Skip remaining epochs if training has not improved after N epochs')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight decay')
    parser.add_argument('--dataset_dir', default='datasets/mydatasets/CAMELYON16-uni2', type=str, help='Dataset folder name')
    parser.add_argument('--label_file', default='Camelyon16.csv', type=str, help='CSV label file name')
    parser.add_argument('--num_folds', default=5, type=int, help='Number of folds for stratified k-fold cross-validation')
    parser.add_argument('--train_folds', type=str, default='all', help='Comma-separated list of fold indices to train (e.g., "1,3,5" or "all"). If "all", train all folds.')
    
    parser.add_argument('--model', default='efficientmil_gru', type=str, help='MIL model', choices=['efficientmil_lstm', 'efficientmil_gru', 'efficientmil_mamba'])
    parser.add_argument('--average', type=bool, default=False, help='Average the score of max-pooling and bag aggregating')
    parser.add_argument('--save_dir', default='outputs/camelyon16/efficientmil_gru', type=str, help='Output save directory')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed for reproducible data splits')
    
    # Model parameters
    parser.add_argument('--gru_hidden_size', type=int, default=768, help='GRU hidden size')
    parser.add_argument('--gru_num_layers', type=int, default=2, help='GRU number of layers')
    parser.add_argument('--gru_bidirectional', type=str, default='True', choices=['True', 'False'], help='GRU bidirectional')
    parser.add_argument('--gru_selection_strategy', type=str, default='aps', choices=['random-k', 'top-k', 'aps'], help='GRU selection strategy')
    
    parser.add_argument('--lstm_hidden_size', default=768, type=int, help='LSTM hidden size')
    parser.add_argument('--lstm_num_layers', default=2, type=int, help='LSTM number of layers')
    parser.add_argument('--lstm_bidirectional', default='True', type=str, choices=['True', 'False'], help='LSTM bidirectional')
    parser.add_argument('--lstm_selection_strategy', default='aps', type=str, choices=['random-k', 'top-k', 'aps'], help='LSTM selection strategy')
    
    parser.add_argument('--mamba_depth', type=int, default=8, help='Mamba depth')
    parser.add_argument('--mamba_d_state', type=int, default=32, help='Mamba d_state')
    parser.add_argument('--mamba_d_conv', type=int, default=4, help='Mamba d_conv')
    parser.add_argument('--mamba_expand', type=int, default=2, help='Mamba expand')
    parser.add_argument('--mamba_selection_strategy', type=str, default='aps', choices=['random-k', 'top-k', 'aps'], help='Mamba selection strategy')
    parser.add_argument('--big_lambda', type=int, default=64, help='Big lambda (patch selection number)')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    # JSON parameter configuration support
    parser.add_argument('--params', type=str, help='JSON format model parameter configuration dictionary')
    
    # TensorBoard parameters
    parser.add_argument('--use_tensorboard', action='store_true', default=True, help='Enable TensorBoard logging')
    parser.add_argument('--tensorboard_dir', default='tensorboard', type=str, help='TensorBoard log directory name')
    
    # Resume training
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint if available')
    parser.add_argument('--resume_path', type=str, default=None, help='Path to a specific checkpoint to resume from (overrides --resume default path)')
    
    # L2 regularization
    parser.add_argument('--l2_loss_weight', type=float, default=1e-4, help='L2 regularization weight')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_file=os.path.join(args.save_dir, 'log.txt'))
    
    # Set random seed
    set_random_seed(args.random_seed)
    
    # Process JSON parameter configuration
    if args.params:
        try:
            import json
            if os.path.isfile(args.params):
                with open(args.params, 'r') as f:
                    params_dict = json.load(f)
                    params_dict['model'] = args.model
            else:
                params_dict = json.loads(args.params)
            logging.info(f"Loading JSON parameter configuration: {params_dict}")
            
            # Override args with JSON parameters (JSON config takes priority)
            for key, value in params_dict.items():
                if hasattr(args, key):
                    setattr(args, key, value)
                    logging.info(f"   Override parameter {key}: {value}")
                else:
                    logging.warning(f"   Unknown parameter {key}: {value}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON parameter parsing failed: {e}")
            sys.exit(1)
    
    # Initialize TensorBoard writer
    tb_writer = None
    if args.use_tensorboard:
        tb_dir = os.path.join(args.save_dir, args.tensorboard_dir)
        os.makedirs(tb_dir, exist_ok=True)
        tb_writer = SummaryWriter(log_dir=tb_dir)
        logging.info(f"TensorBoard logging enabled. Log directory: {tb_dir}")
    
    
    # Parse model parameters
    model_params = parse_model_params(args)
    logging.info(f"Model: {args.model}")
    logging.info(f"Model parameters: {model_params}")
    
    # Log hyperparameters to TensorBoard
    if tb_writer:
        hparams = {
            'model': args.model,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'num_folds': args.num_folds,
            'feats_size': args.feats_size,
            'num_classes': args.num_classes,
            'weight_decay': args.weight_decay,
            'dropout': args.dropout
        }
        # Add model-specific parameters
        hparams.update(model_params)
        tb_writer.add_hparams(hparams, {'hparam/placeholder': 0})
    
    def init_model(args):
        model_start_time = time.time()
        device = torch.device(args.device)  
        
        i_classifier = efficientmil_common.FCLayer(in_size=args.feats_size, out_size=args.num_classes).to(device)
        
        if args.model == 'efficientmil_mamba':
            from efficientmil_mamba import BClassifier
            mamba_params = parse_model_params(args)
            b_classifier = BClassifier(input_size=args.feats_size, 
                                          output_class=args.num_classes, 
                                          **mamba_params).to(device)
        elif args.model == 'efficientmil_gru':
            from efficientmil_gru import BClassifier
            gru_params = parse_model_params(args)
            b_classifier = BClassifier(input_size=args.feats_size, 
                                          output_class=args.num_classes, 
                                          **gru_params).to(device)
        elif args.model == 'efficientmil_lstm':
            from efficientmil_lstm import BClassifier
            lstm_params = parse_model_params(args)
            b_classifier = BClassifier(input_size=args.feats_size, 
                                          output_class=args.num_classes, 
                                          **lstm_params).to(device)
        else:
            raise ValueError(f"Model {args.model} not supported")
        
        milnet = efficientmil_common.MILNet(i_classifier, b_classifier).to(device)
        
        # Apply sparse initialization
        milnet.apply(lambda m: apply_sparse_init(m))
        
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(milnet.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs, 5e-6)
        
        model_time = time.time() - model_start_time
        log_model_info(milnet, model_time)
        
        return milnet, criterion, optimizer, scheduler
    
    # Dataset setup
    patch_features_dir = os.path.join(args.dataset_dir, 'pt_files')
    label_file = os.path.join(args.dataset_dir, args.label_file)
    dataset = PatchFeaturesDataset(patch_features_dir, label_file)
    
    logging.info(f"Dataset loaded with {len(dataset)} samples")

    # Create base save directory structure
    base_save_path = os.path.join(args.save_dir, 'weights')
    os.makedirs(base_save_path, exist_ok=True)
    logging.info(f"Model base save path: {base_save_path}")

    # Get labels for stratified sampling
    all_labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    
    logging.info(f"Stratified {args.num_folds}-fold cross-validation - Total samples: {len(dataset)}")
    
    # Use StratifiedKFold for stratified cross-validation
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.random_seed)
    # Materialize splits so we can select specific folds
    all_splits = list(skf.split(range(len(dataset)), all_labels))

    # Determine which folds to train (1-based indices in CLI)
    if args.train_folds is None or str(args.train_folds).lower() == 'all':
        folds_to_train = list(range(args.num_folds))  # 0-based internal indices
    else:
        try:
            requested = [int(f.strip()) for f in str(args.train_folds).split(',') if f.strip() != '']
            # Convert to 0-based and validate
            folds_to_train = []
            for f in requested:
                if 1 <= f <= args.num_folds:
                    folds_to_train.append(f - 1)
                else:
                    logging.error(f"Invalid fold index: {f}. Valid range is 1..{args.num_folds}")
            if not folds_to_train:
                logging.error("No valid folds selected. Exiting.")
                return
        except ValueError:
            logging.error(f"Invalid train_folds format: {args.train_folds}. Use comma-separated integers (e.g., '1,3,5') or 'all'")
            return

    logging.info(f"Training folds (1-based): {[f+1 for f in folds_to_train]}")

    fold_results = []

    training_start_time = time.time()
    
    for fold in folds_to_train:
        train_index, test_index = all_splits[fold]
        fold_start_time = time.time()
        
        logging.info(f"Starting fold {fold + 1}/{args.num_folds} cross-validation")
        
        milnet, criterion, optimizer, scheduler = init_model(args)
        
        # Create fold-specific save directory
        fold_save_path = os.path.join(base_save_path, f'fold_{fold + 1}')
        os.makedirs(fold_save_path, exist_ok=True)
        
        # Load checkpoint for resume if requested
        resume_state = load_checkpoint_for_resume(args, fold, milnet, optimizer, scheduler)
        start_epoch = resume_state['start_epoch']
        fold_best_score = resume_state['best_score']
        counter = resume_state['patience_counter']
        thresholds_optimal = resume_state['thresholds_optimal'] if resume_state['thresholds_optimal'] is not None else None
        
        # Create subset datasets for this fold
        train_dataset = torch.utils.data.Subset(dataset, train_index)
        test_dataset = torch.utils.data.Subset(dataset, test_index)
        
        # Create data loaders - batch size 1 for MIL (each sample is a bag)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        logging.info(f"Training samples: {len(train_dataset)}, Testing samples: {len(test_dataset)}")
        if args.resume and start_epoch > 1:
            logging.info(f"Resuming from epoch {start_epoch}, previous best score: {fold_best_score:.4f}")
        
        # Initialize training state
        best_ac = 0
        best_auc = 0

        for epoch in range(start_epoch, args.num_epochs+1):
            epoch_start_time = time.time()
            counter += 1
            
            # Enhanced epoch header
            logging.info("-"*30 + f" Fold [{fold + 1}/{args.num_folds}], Epoch [{epoch}/{args.num_epochs}], LR={optimizer.param_groups[0]['lr']:.8f} " + "-"*30)
            
            # Training phase
            train_loss_bag, train_accuracy, train_auc = train(args, train_loader, milnet, criterion, optimizer, tb_writer, epoch, fold)
            
            # Validation phase  
            test_result = test(args, test_loader, milnet, criterion, thresholds=thresholds_optimal, tb_writer=tb_writer, epoch=epoch, fold=fold)
            test_loss_bag, avg_score, aucs, current_thresholds, val_accuracy, val_auc = test_result
            
            # Update thresholds if we got new ones
            if thresholds_optimal is None:
                thresholds_optimal = current_thresholds
            
            # Enhanced logging with both training and validation metrics
            logging.info(f'Train Loss: {train_loss_bag:.4f}, Train Acc: {train_accuracy:.4f}, Train AUC: {train_auc:.4f}')
            logging.info(f'Val Loss: {test_loss_bag:.4f}, Val Acc: {val_accuracy:.4f}, Val AUC: {val_auc:.4f}')
            
            scheduler.step()

            current_score = get_current_score(avg_score, aucs)
            epoch_time = time.time() - epoch_start_time
            
            logging.info(f"Epoch {epoch} total time: {epoch_time:.2f}s, Current score: {current_score:.4f}")
            
            # Save last checkpoint every epoch (for resuming training)
            last_ckpt_path = os.path.join(fold_save_path, 'last.pth')
            save_checkpoint(
                save_path=last_ckpt_path,
                model=milnet,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=fold_best_score,
                patience_counter=counter,
                fold=fold,
                thresholds_optimal=thresholds_optimal,
                extra={
                    'avg_score': avg_score, 
                    'aucs': aucs,
                    'train_acc': train_accuracy,
                    'train_auc': train_auc,
                    'val_acc': val_accuracy,
                    'val_auc': val_auc
                }
            )
            logging.info(f'Last model saved to {last_ckpt_path}')
            
            if current_score > fold_best_score:
                counter = 0
                fold_best_score = current_score
                best_ac = avg_score
                best_auc = aucs
                
                # Save best checkpoint
                best_ckpt_path = os.path.join(fold_save_path, 'best.pth')
                save_checkpoint(
                    save_path=best_ckpt_path,
                    model=milnet,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    best_score=fold_best_score,
                    patience_counter=counter,
                    fold=fold,
                    thresholds_optimal=thresholds_optimal,
                    extra={
                        'avg_score': avg_score, 
                        'aucs': aucs,
                        'train_acc': train_accuracy,
                        'train_auc': train_auc,
                        'val_acc': val_accuracy,
                        'val_auc': val_auc
                    }
                )
                logging.info(f"Best model [acc={val_accuracy:.4f}] saved to {best_ckpt_path}")
            else:
                counter += 1
                
            if counter > args.patience: 
                logging.info(f"Early stopping triggered - No improvement for {args.patience} epochs")
                break
                
        fold_time = time.time() - fold_start_time
        fold_results.append((best_ac, best_auc))
        logging.info(f"Fold {fold + 1} completed - Best accuracy: {best_ac:.4f}, Best AUC: {best_auc}, Time: {fold_time:.2f}s")

    # Final results statistics
    if fold_results:
        print_final_results(fold_results, time.time() - training_start_time)
    else:
        logging.warning("No fold produced results, possibly all folds completed or errors occurred")

    logging.info("=" * 80)
    logging.info("Training completed!")
    logging.info("=" * 80)
    
    # Close TensorBoard writer
    if tb_writer:
        tb_writer.close()
        logging.info("TensorBoard logging completed")

if __name__ == '__main__':
    main()