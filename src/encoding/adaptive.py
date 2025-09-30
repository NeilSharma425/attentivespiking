"""
Adaptive encoder: combines selector with encoding library.
"""

import torch
import torch.nn as nn
from .library import EncodingLibrary
from .selector import EncodingSelector


# Add this to AdaptiveEncoder class

class AdaptiveEncoder(nn.Module):
    """
    Adaptively encodes inputs using learned mixture of encoding schemes.
    """
    
    def __init__(self, d_model=768, T=4, num_encodings=5, fixed_encoding=None):
        super().__init__()
        
        self.d_model = d_model
        self.T = T
        self.num_encodings = num_encodings
        self.fixed_encoding = fixed_encoding  # NEW: 'rate', 'temporal', etc. or None
        
        # Only create selector if not fixed
        if fixed_encoding is None:
            self.selector = EncodingSelector(d_model, num_encodings)
        
        self.library = EncodingLibrary()
        
        # For tracking statistics
        self.register_buffer('encoding_usage', torch.zeros(num_encodings))
        self.register_buffer('call_count', torch.tensor(0))
    
    def forward(self, x, position_ids, layer_idx, return_stats=False):
        """Encode inputs with adaptive or fixed strategy."""
        
        if self.fixed_encoding is not None:
            # FIXED MODE: Use only one encoding
            encoder_map = {
                'rate': self.library.rate_encoding,
                'temporal': self.library.temporal_encoding,
                'population': self.library.population_encoding,
                'burst': self.library.burst_encoding,
                'adaptive': self.library.adaptive_threshold_encoding,
            }
            
            spikes = encoder_map[self.fixed_encoding](x, self.T)
            
            if return_stats:
                # Create dummy weights showing 100% for this encoding
                weights = torch.zeros(x.shape[0], x.shape[1], self.num_encodings)
                idx = list(encoder_map.keys()).index(self.fixed_encoding)
                weights[..., idx] = 1.0
                return spikes, weights
            
            return spikes
        
        else:
            # ADAPTIVE MODE: Original code
            weights = self.selector(x, position_ids, layer_idx)
            
            encodings = []
            encodings.append(self.library.rate_encoding(x, self.T))
            encodings.append(self.library.temporal_encoding(x, self.T))
            encodings.append(self.library.population_encoding(x, self.T))
            encodings.append(self.library.burst_encoding(x, self.T))
            encodings.append(self.library.adaptive_threshold_encoding(x, self.T))
            
            encodings = torch.stack(encodings, dim=0)
            weights_expanded = weights.permute(2, 0, 1).unsqueeze(-1).unsqueeze(-1)
            mixed_spikes = (encodings * weights_expanded).sum(dim=0)
            
            if not self.training:
                mixed_spikes = (mixed_spikes > 0.5).float()
            else:
                mixed_spikes = (mixed_spikes > 0.5).float() - mixed_spikes.detach() + mixed_spikes
            
            if self.training:
                with torch.no_grad():
                    self.encoding_usage += weights.mean(dim=(0, 1))
                    self.call_count += 1
            
            if return_stats:
                return mixed_spikes, weights
            
            return mixed_spikes