# efficientmil_mamba.py - EfficientMIL with Mamba Encoder
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

# Try to import mamba_ssm, fallback to simple implementation if not available
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
    MAMBA_AVAILABLE = True
except Exception as e:
    print(f"Warning: mamba_ssm not available ({e}). Using fallback implementation.")
    Mamba = None
    RMSNorm = None
    layer_norm_fn = None
    rms_norm_fn = None
    MAMBA_AVAILABLE = False


class MambaBlock(nn.Module):
    """
    Mamba block adapted for MIL tasks, based on Vision Mamba
    """
    def __init__(
        self, 
        dim, 
        d_state=16,
        d_conv=4,
        expand=2,
        norm_cls=nn.LayerNorm, 
        fused_add_norm=False, 
        residual_in_fp32=False,
        dropout=0.,
        layer_idx=None,
        bimamba_type="v2",
        if_divide_out=True,
        **factory_kwargs
    ):
        """
        Mamba block, similar to original Block class but using Mamba instead of Multi-head Attention
        
        Args:
            dim: Feature dimension
            d_state: Mamba state dimension
            d_conv: Convolution kernel size
            expand: Expansion ratio
            norm_cls: Normalization layer type
            fused_add_norm: Whether to use fused add+norm
            residual_in_fp32: Whether residual connection uses fp32
            dropout: DropPath ratio
            layer_idx: Layer index
            bimamba_type: BiMamba type
            if_divide_out: Whether to divide output
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        
        # Mamba mixer
        if MAMBA_AVAILABLE and Mamba is not None:
            self.mixer = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                layer_idx=layer_idx,
                bimamba_type=bimamba_type,
                if_divide_out=if_divide_out,
                **factory_kwargs
            )
        else:
            # Fallback: use simple linear layers
            self.mixer = nn.Sequential(
                nn.Linear(dim, dim * expand),
                nn.GELU(),
                nn.Linear(dim * expand, dim)
            )
        
        self.norm = norm_cls(dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
        
        if self.fused_add_norm:
            if not MAMBA_AVAILABLE or RMSNorm is None:
                raise RuntimeError("RMSNorm not available. Cannot use fused_add_norm without mamba_ssm.")
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, 
        hidden_states: Tensor, 
        residual: Optional[Tensor] = None, 
        inference_params=None,
        **kwargs
    ):
        """
        Forward pass
        
        Args:
            hidden_states: Input features [B, N, D]
            residual: Residual connection
            inference_params: Inference parameters
        
        Returns:
            Tuple[Tensor, Tensor]: (output features, residual)
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.dropout(hidden_states)
            
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            if not MAMBA_AVAILABLE or rms_norm_fn is None or layer_norm_fn is None:
                raise RuntimeError("Fused norm functions not available. Cannot use fused_add_norm without mamba_ssm.")
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.dropout(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        
        # Through Mamba mixer
        if hasattr(self.mixer, 'forward') and 'inference_params' in self.mixer.forward.__code__.co_varnames:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        else:
            hidden_states = self.mixer(hidden_states)
        
        return hidden_states, residual


class BClassifier(BaseBClassifier):
    """
    Mamba-based bag-level classifier replacing original Transformer
    """
    def __init__(self, input_size, output_class, 
                 # Mamba specific parameters
                 mamba_depth=6, d_state=16, 
                 d_conv=4, expand=2, big_lambda=64,
                 selection_strategy='aps', 
                 use_intelligent_selector=True,
                 dropout=0.1):
        super().__init__(
            input_size=input_size,
            output_class=output_class,
            big_lambda=big_lambda,
            selection_strategy=selection_strategy,
            use_intelligent_selector=use_intelligent_selector,
        )
        
        # Mamba encoder layers
        self.mamba_depth = mamba_depth
        self.embed_dim = input_size
        
        # Build Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(
                dim=input_size,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                norm_cls=RMSNorm if MAMBA_AVAILABLE and RMSNorm is not None else nn.LayerNorm,
                fused_add_norm=False,
                residual_in_fp32=False,
                dropout=dropout,
                layer_idx=i,
                bimamba_type="v2",
                if_divide_out=True,
            )
            for i in range(mamba_depth)
        ])
        
        # Final normalization layer
        self.norm_f = (RMSNorm if MAMBA_AVAILABLE and RMSNorm is not None else nn.LayerNorm)(input_size)

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
        
        # Through Mamba encoder
        hidden_states = m_feats
        residual = None
        
        for layer in self.mamba_layers:
            hidden_states, residual = layer(hidden_states, residual)
        
        # Final normalization
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + hidden_states
        
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        
        # Global pooling: average pooling
        bag_representation = hidden_states.mean(dim=1)  # [1, D]
        
        # Final classification
        bag_prediction = self.classifier(bag_representation)  # [1, C]
        
        # Format output to match DSMIL
        return self._format_output(bag_prediction, attention_weights, bag_representation, c)


def create_efficientmil_mamba_model(input_size=1024, output_class=1, 
                            dropout=0.1, mamba_depth=6, 
                            d_state=16, selection_strategy='aps',
                            use_intelligent_selector=True):
    """
    Factory function: Create EfficientMIL-Mamba model
    
    Args:
        input_size: Input feature dimension
        output_class: Number of output classes
        dropout: Dropout ratio
        mamba_depth: Number of Mamba layers
        d_state: Mamba state dimension
        selection_strategy: Patch selection strategy
    
    Returns:
        MILNet: Complete model
    """
    # Instance-level classifier
    i_classifier = FCLayer(input_size, output_class)
    
    # Bag-level classifier (Mamba-based)
    b_classifier = BClassifier(
        input_size=input_size,
        output_class=output_class,
        mamba_depth=mamba_depth,
        d_state=d_state,
        selection_strategy=selection_strategy,
        use_intelligent_selector=use_intelligent_selector,
        dropout=dropout
    )
    
    # Complete model
    model = MILNet(i_classifier, b_classifier)
    
    # Weight initialization
    initialize_weights(model)
    
    return model
