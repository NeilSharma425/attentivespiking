"""
Spiking multi-head attention mechanism.
"""

import torch
import torch.nn as nn
import math


class SpikingMultiHeadAttention(nn.Module):
    """
    Multi-head attention operating on spike trains.
    Processes each timestep independently.
    """
    
    def __init__(self, d_model, nhead, T, dropout=0.1):
        super().__init__()
        
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        self.T = T
        
        # Linear projections (shared across time)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spikes, mask=None):
        """
        Args:
            spikes: [batch, seq_len, d_model, T]
            mask: Optional attention mask [batch, seq_len, seq_len]
            
        Returns:
            output: [batch, seq_len, d_model, T]
        """
        B, S, D, T = spikes.shape
        
        outputs = []
        
        # Process each timestep
        for t in range(T):
            # Extract spikes at time t
            x_t = spikes[..., t]  # [B, S, D]
            
            # Linear projections
            Q = self.W_q(x_t)  # [B, S, D]
            K = self.W_k(x_t)
            V = self.W_v(x_t)
            
            # Reshape for multi-head attention
            # [B, S, D] → [B, S, nhead, d_k] → [B, nhead, S, d_k]
            Q = Q.view(B, S, self.nhead, self.d_k).transpose(1, 2)
            K = K.view(B, S, self.nhead, self.d_k).transpose(1, 2)
            V = V.view(B, S, self.nhead, self.d_k).transpose(1, 2)
            
            # Scaled dot-product attention
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Attention weights
            attn_probs = torch.softmax(scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Apply attention to values
            context = torch.matmul(attn_probs, V)  # [B, nhead, S, d_k]
            
            # Concatenate heads
            context = context.transpose(1, 2).contiguous()  # [B, S, nhead, d_k]
            context = context.view(B, S, D)  # [B, S, D]
            
            # Output projection
            output_t = self.W_o(context)
            outputs.append(output_t)
        
        # Stack outputs across time: [B, S, D, T]
        output = torch.stack(outputs, dim=-1)
        
        return output