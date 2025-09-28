# efficientmil_lstm.py - EfficientMIL with LSTM Encoder
import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from efficientmil_common import (
    FCLayer, IClassifier, IntelligentPatchSelector, BaseBClassifier, 
    MILNet, initialize_weights, clones
)


class LSTMBlock(nn.Module):
    """
    LSTM-based encoding block for replacing Transformer self-attention mechanism
    """
    def __init__(
        self, 
        input_size, 
        hidden_size=None,
        num_layers=2,
        dropout=0.1,
        bidirectional=True,
        batch_first=True,
        **kwargs
    ):
        """
        LSTM encoding block
        
        Args:
            input_size: Input feature dimension
            hidden_size: LSTM hidden layer dimension, default to input_size
            num_layers: Number of LSTM layers
            dropout: Dropout ratio
            bidirectional: Whether to use bidirectional LSTM
            batch_first: Whether batch dimension comes first
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size is not None else input_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.batch_first = batch_first
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=batch_first
        )
        
        # Output dimension calculation
        lstm_output_size = self.hidden_size * (2 if bidirectional else 1)
        
        # Projection layer to map LSTM output back to original dimension
        if lstm_output_size != input_size:
            self.projection = nn.Linear(lstm_output_size, input_size)
        else:
            self.projection = nn.Identity()
        
        # Normalization and Dropout
        # self.layer_norm = nn.LayerNorm(input_size)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden_state=None):
        """
        Forward pass
        
        Args:
            x: Input features [B, N, D]
            hidden_state: Optional initial hidden state
        
        Returns:
            Tuple[Tensor, Tuple]: (output features, (h_n, c_n))
        """
        # Save input for residual connection
        # residual = x
        
        # Through LSTM
        if hidden_state is not None:
            lstm_output, hidden_state = self.lstm(x, hidden_state)
        else:
            lstm_output, hidden_state = self.lstm(x)
        
        # Project to original dimension
        output = self.projection(lstm_output)
        
        # Residual connection and normalization
        # output = self.layer_norm(output + residual)
        # output = self.dropout(output)
        
        return output, hidden_state


class BClassifier(BaseBClassifier):
    """
    LSTM-based bag-level classifier replacing original Transformer
    """
    def __init__(self, input_size, output_class, 
                 # LSTM specific parameters
                 lstm_hidden_size=512, 
                 lstm_num_layers=4, 
                 bidirectional=True,
                 dropout=0.1,
                 big_lambda=64,
                 selection_strategy='hybrid', 
                 use_intelligent_selector=True):
        super().__init__(
            input_size=input_size,
            output_class=output_class,
            big_lambda=big_lambda,
            selection_strategy=selection_strategy,
            use_intelligent_selector=use_intelligent_selector
        )
        
        # LSTM encoder layers
        self.lstm_hidden_size = lstm_hidden_size if lstm_hidden_size is not None else input_size
        self.lstm_num_layers = lstm_num_layers
        
        # Build LSTM layer
        self.lstm_encoder = LSTMBlock(
            input_size=input_size,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
        )
        # Normalization and Dropout
        self.layer_norm = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, feats, c):
        """
        Forward pass maintaining DSMIL-consistent interface
        
        Args:
            feats: Input features [N, D] 
            c: Instance classification scores [N, C]
        
        Returns:
            Tuple: (bag prediction [1, C], attention weights [N, C], bag representation [1, D])
        """
        # Apply patch selection
        m_feats, attention_weights = self._apply_patch_selection(feats, c)
        
        # Through LSTM encoder
        encoded_feats, _ = self.lstm_encoder(m_feats)
        encoded_feats = encoded_feats + m_feats  # add residual connection
        encoded_feats = self.layer_norm(encoded_feats)
        encoded_feats = self.dropout(encoded_feats)
        
        # Global pooling: average pooling
        bag_representation = encoded_feats.mean(dim=1)  # [1, D]
        
        # Final classification
        bag_prediction = self.classifier(bag_representation)  # [1, C]
        
        # Format output to match DSMIL
        return self._format_output(bag_prediction, attention_weights, bag_representation, c)


def create_efficientmil_lstm_model(input_size=1024, output_class=1, 
                            dropout=0.1, lstm_hidden_size=512, 
                            lstm_num_layers=4, selection_strategy='hybrid', 
                            bidirectional=True, use_intelligent_selector=True):
    """
    Factory function: Create EfficientMIL-LSTM model
    
    Args:
        input_size: Input feature dimension
        output_class: Number of output classes
        dropout: Dropout ratio
        lstm_hidden_size: LSTM hidden layer dimension
        lstm_num_layers: Number of LSTM layers
        selection_strategy: Patch selection strategy ('original' or 'hybrid')
        bidirectional: Whether to use bidirectional LSTM
        use_intelligent_selector: Whether to use intelligent selector
    Returns:
        MILNet: Complete model
    """
    # Instance-level classifier
    i_classifier = FCLayer(input_size, output_class)
    
    # Bag-level classifier (LSTM-based)
    b_classifier = BClassifier(
        input_size=input_size,
        output_class=output_class,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        selection_strategy=selection_strategy,
        use_intelligent_selector=use_intelligent_selector,
        bidirectional=bidirectional,
        dropout=dropout,
    )
    
    # Complete model
    model = MILNet(i_classifier, b_classifier)
    
    # Weight initialization
    initialize_weights(model)
    
    return model

