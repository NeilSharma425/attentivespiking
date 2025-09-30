"""
Encoding selector: learns to choose optimal encoding scheme
based on input context.
"""

import torch
import torch.nn as nn


class EncodingSelector(nn.Module):
    """
    Learns to select optimal spike encoding based on:
    - Token semantics (from embeddings)
    - Positional context (where in sequence)
    - Layer characteristics (network depth)
    """
    
    def __init__(self, d_model=768, num_encoding_types=5, 
                 max_positions=512, max_layers=24):
        super().__init__()
        
        self.d_model = d_model
        self.num_encoding_types = num_encoding_types
        
        # Context embedding dimensions
        context_dim = 64
        
        # Learnable embeddings for context
        self.positional_embedding = nn.Embedding(max_positions, context_dim)
        self.layer_embedding = nn.Embedding(max_layers, context_dim)
        
        # Selector network (lightweight MLP)
        self.selector = nn.Sequential(
            nn.Linear(d_model + 2 * context_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_encoding_types)
        )
        
    def forward(self, x, position_ids, layer_idx):
        """
        Select encoding weights for each token.
        
        Args:
            x: Token embeddings [batch, seq_len, d_model]
            position_ids: Position indices [batch, seq_len]
            layer_idx: Current layer index (scalar)
            
        Returns:
            weights: Encoding weights [batch, seq_len, num_encoding_types]
        """
        batch_size, seq_len, _ = x.shape
        
        # Get context embeddings
        pos_embed = self.positional_embedding(position_ids)  # [B, S, context_dim]
        
        # Layer embedding (broadcast to all positions)
        layer_idx_tensor = torch.full(
            (batch_size, seq_len), 
            layer_idx, 
            dtype=torch.long, 
            device=x.device
        )
        layer_embed = self.layer_embedding(layer_idx_tensor)  # [B, S, context_dim]
        
        # Concatenate all features
        features = torch.cat([x, pos_embed, layer_embed], dim=-1)
        
        # Predict encoding weights
        logits = self.selector(features)  # [B, S, num_types]
        
        # Softmax for probabilistic selection
        weights = torch.softmax(logits, dim=-1)
        
        return weights
    
    def get_hard_selection(self, x, position_ids, layer_idx):
        """
        Get discrete encoding selection (for inference).
        
        Returns:
            indices: Selected encoding index [batch, seq_len]
        """
        weights = self.forward(x, position_ids, layer_idx)
        indices = torch.argmax(weights, dim=-1)
        return indices