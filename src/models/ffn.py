"""
Spiking feedforward network.
"""

import torch
import torch.nn as nn


class SpikingFFN(nn.Module):
    """
    Position-wise feedforward network for spike trains.
    """
    
    def __init__(self, d_model, dim_feedforward, T, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.T = T
        
        # Two-layer MLP
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # Can experiment with GELU
        
    def forward(self, spikes):
        """
        Args:
            spikes: [batch, seq_len, d_model, T]
            
        Returns:
            output: [batch, seq_len, d_model, T]
        """
        B, S, D, T = spikes.shape
        
        outputs = []
        
        # Process each timestep independently
        for t in range(T):
            x_t = spikes[..., t]  # [B, S, D]
            
            # First layer
            hidden = self.fc1(x_t)  # [B, S, dim_feedforward]
            hidden = self.activation(hidden)
            hidden = self.dropout(hidden)
            
            # Second layer
            output_t = self.fc2(hidden)  # [B, S, D]
            output_t = self.dropout(output_t)
            
            outputs.append(output_t)
        
        # Stack across time
        output = torch.stack(outputs, dim=-1)  # [B, S, D, T]
        
        return output