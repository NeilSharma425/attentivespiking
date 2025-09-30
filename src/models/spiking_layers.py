"""
Core spiking transformer layer.
"""

import torch
import torch.nn as nn
from ..encoding.adaptive import AdaptiveEncoder
from .attention import SpikingMultiHeadAttention
from .ffn import SpikingFFN


class SpikingTransformerLayer(nn.Module):
    """
    Single transformer layer with adaptive spike encoding.
    """
    
    def __init__(self, d_model=768, nhead=12, dim_feedforward=3072, 
                 T=4, layer_idx=0, dropout=0.1, fixed_encoding=None):
        super().__init__()
        
        self.layer_idx = layer_idx
        self.T = T
        
        # Adaptive encoders for this layer (now support fixed mode)
        self.encoder_attn = AdaptiveEncoder(d_model, T, fixed_encoding=fixed_encoding)
        self.encoder_ffn = AdaptiveEncoder(d_model, T, fixed_encoding=fixed_encoding)
        
        # Spiking attention
        self.attention = SpikingMultiHeadAttention(d_model, nhead, T)
        
        # Spiking feedforward network
        self.ffn = SpikingFFN(d_model, dim_feedforward, T)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, position_ids, mask=None):
        """Forward pass through spiking transformer layer."""
        # Self-attention block
        residual = x
        x_norm = self.norm1(x)
        
        # Encode to spikes
        spikes_attn = self.encoder_attn(x_norm, position_ids, self.layer_idx)
        
        # Process with spiking attention
        attn_output = self.attention(spikes_attn, mask=mask)
        
        # Decode: average over time dimension
        attn_output = attn_output.mean(dim=-1)
        
        # Residual connection
        x = residual + self.dropout(attn_output)
        
        # Feedforward block
        residual = x
        x_norm = self.norm2(x)
        
        # Encode to spikes
        spikes_ffn = self.encoder_ffn(x_norm, position_ids, self.layer_idx)
        
        # Process with spiking FFN
        ffn_output = self.ffn(spikes_ffn)
        
        # Decode: average over time
        ffn_output = ffn_output.mean(dim=-1)
        
        # Residual connection
        x = residual + self.dropout(ffn_output)
        
        return x