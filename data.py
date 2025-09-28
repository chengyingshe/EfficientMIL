import torch
from torch.utils.data import Dataset
import glob
import os
import pandas as pd
import logging

class PatchFeaturesDataset(Dataset):
    def __init__(self, patch_features_dir, csv_label_file):
        label_df = pd.read_csv(csv_label_file)
        self.patch_features_files = [os.path.join(patch_features_dir, filename) for filename in label_df.iloc[:, 0].values]
        self.labels = label_df.iloc[:, 1].values
        

    def __len__(self):
        return len(self.patch_features_files)
    
    def __getitem__(self, idx):
        patch_features = torch.load(self.patch_features_files[idx]).to(torch.float32)
        # Handle different data formats
        if patch_features.ndim == 3:
            patch_features = patch_features.squeeze(0)
        
        return {
            'patch_features': patch_features,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def collate_fn(batch):
    """
    Custom collate function to handle variable-sized patch features
    """
    # Separate patch features from other data
    patch_features = [item['patch_features'] for item in batch]
    other_data = {}
    
    # Get keys for other data (time, status, or label)
    for key in batch[0].keys():
        if key != 'patch_features':
            other_data[key] = [item[key] for item in batch]
    
    return {
        'patch_features': patch_features,  # Keep as list since sizes vary
        **other_data
    }


class ClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for classification tasks, adapted from MultiTaskDataset.
    """
    def __init__(self, 
                 dataset_dir: str,
                 patch_features_dir_name: str = 'pt_files',
                 classification_file: str = 'clinical_data.csv',
                 id_column: str = 'ID',
                 classification_label_column: str = 'T',
                 id_mapping_file: str = 'id_mapping.csv',
                 cache_size: int = 100):
        """
        Initialize classification dataset.
        
        Args:
            dataset_dir: Root directory containing data files
            patch_features_dir_name: Directory name for patch features
            classification_file: CSV file with classification labels
            id_column: Column name for patient ID
            classification_label_column: Column name for classification label
            id_mapping_file: CSV file mapping patient IDs to feature files
            cache_size: Size of feature cache
        """
        self.dataset_dir = dataset_dir
        self.patch_features_dir = os.path.join(dataset_dir, patch_features_dir_name)
        self.id_column = id_column
        self.classification_label_column = classification_label_column
        
        # Load classification data
        classification_path = os.path.join(dataset_dir, classification_file)
        if not os.path.exists(classification_path):
            raise FileNotFoundError(f"Classification file not found: {classification_path}")
        
        self.classification_data = pd.read_csv(classification_path)
        
        # Check if label column exists
        if classification_label_column not in self.classification_data.columns:
            raise ValueError(f"Label column '{classification_label_column}' not found in classification file")
        
        # Create label mapping
        raw_values = self.classification_data[classification_label_column].dropna().astype(str)
        unique_vals = sorted(raw_values.unique().tolist())
        logging.info(f"Unique classification labels: {unique_vals}")
        self.class_label_to_index = {v: i for i, v in enumerate(unique_vals)}
        self.index_to_class_label = {i: v for v, i in self.class_label_to_index.items()}
        self.num_classes = len(unique_vals)
        
        # Load ID mapping
        mapping_file = os.path.join(dataset_dir, id_mapping_file)
        if os.path.exists(mapping_file):
            self.id_mapping = pd.read_csv(mapping_file)
            self.id_to_filename = dict(zip(self.id_mapping['csv_id'], self.id_mapping['pt_filename']))
        else:
            logging.warning(f"ID mapping file not found: {mapping_file}")
            self.id_to_filename = {}
        
        # Build sample list
        self.samples = self._build_sample_list()
        
        # Feature cache
        self.cache_size = cache_size
        self._cache = {}
        self._cache_order = []
        
        logging.info(f"ClassificationDataset initialized with {len(self.samples)} samples")
        logging.info(f"Number of classes: {self.num_classes}")
    
    def _build_sample_list(self):
        """Build list of valid samples with classification labels."""
        samples = []
        
        for _, row in self.classification_data.iterrows():
            patient_id = str(row[self.id_column])
            
            # Check if classification label is valid
            if pd.isna(row[self.classification_label_column]):
                continue
            
            raw_label = str(row[self.classification_label_column])
            if raw_label not in self.class_label_to_index:
                continue
            
            # Check if patch features exist
            if patient_id in self.id_to_filename:
                feature_file = self.id_to_filename[patient_id]
            else:
                feature_file = f"{patient_id}.pt"
            
            feature_path = os.path.join(self.patch_features_dir, feature_file)
            if not os.path.exists(feature_path):
                continue
            
            # Create sample record
            sample = {
                'patient_id': patient_id,
                'feature_path': feature_path,
                'label': self.class_label_to_index[raw_label]
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get sample by index."""
        sample = self.samples[idx]
        
        # Load patch features (with caching)
        if idx in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(idx)
            self._cache_order.append(idx)
            patch_features = self._cache[idx]
        else:
            # Load from file
            patch_features = torch.load(sample['feature_path']).to(torch.float32)
            
            if patch_features.dim() > 2:
                patch_features = patch_features.squeeze(0)
            
            # Add to cache
            if len(self._cache) >= self.cache_size:
                # Remove least recently used
                oldest_idx = self._cache_order.pop(0)
                del self._cache[oldest_idx]
            
            self._cache[idx] = patch_features
            self._cache_order.append(idx)
        
        return {
            'patch_features': patch_features,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'patient_id': sample['patient_id']
        }

