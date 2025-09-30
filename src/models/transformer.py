"""
Complete spiking transformer model.
"""

import torch
import torch.nn as nn
from .spiking_layers import SpikingTransformerLayer


class SpikingTransformer(nn.Module):
    """
    Full spiking transformer language model with adaptive encoding.
    """
    
    def __init__(self, vocab_size=50257, d_model=768, nhead=12, 
                 num_layers=12, dim_feedforward=3072, T=4,
                 max_seq_len=512, dropout=0.1, fixed_encoding=None):
        super().__init__()
        
        # Validate dimensions
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead}). "
            f"Try: d_model=768 with nhead=12, or d_model=256 with nhead=8"
        )
        
        # Auto-adjust dim_feedforward if not specified
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model  # Standard transformer ratio
        
        self.d_model = d_model
        self.num_layers = num_layers
        self.T = T
        
        # Store fixed encoding mode
        self.fixed_encoding = fixed_encoding
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers (now with fixed_encoding support)
        self.layers = nn.ModuleList([
            SpikingTransformerLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                T=T,
                layer_idx=i,
                dropout=dropout,
                fixed_encoding=fixed_encoding  # NEW
            )
            for i in range(num_layers)
        ])
        
        # Output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        self.apply(self._init_weights)
        
        # Print model info
        self._print_model_info()
    
    def _print_model_info(self):
        """Print model architecture info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        encoding_mode = self.fixed_encoding if self.fixed_encoding else "Adaptive"
        
        print(f"\n{'='*60}")
        print(f"SpikingTransformer Model")
        print(f"{'='*60}")
        print(f"Architecture:")
        print(f"  d_model: {self.d_model}")
        print(f"  num_layers: {self.num_layers}")
        print(f"  Time steps (T): {self.T}")
        print(f"  Encoding: {encoding_mode}")  # NEW
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"{'='*60}\n")
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, attention_mask=None, 
                return_encoding_stats=False):
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len] (optional)
            return_encoding_stats: If True, return encoding statistics
            
        Returns:
            logits: [batch, seq_len, vocab_size]
            encoding_stats: (optional) encoding usage statistics
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create position IDs
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_ids = position_ids.expand(batch_size, -1)
        
        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(position_ids)
        x = self.dropout(token_embeds + pos_embeds)
        
        # Process attention mask
        if attention_mask is not None:
            # Convert to attention mask format [batch, seq_len, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(batch_size, 1, seq_len, seq_len)
        
        # Through transformer layers
        encoding_stats = []
        for layer in self.layers:
            x = layer(x, position_ids, mask=attention_mask)
            
            if return_encoding_stats:
                # Collect encoding statistics from each layer
                stats = {
                    'layer': layer.layer_idx,
                    'usage_attn': layer.encoder_attn.get_encoding_statistics(),
                    'usage_ffn': layer.encoder_ffn.get_encoding_statistics()
                }
                encoding_stats.append(stats)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        if return_encoding_stats:
            return logits, encoding_stats
        
        return logits
    
    def generate(self, input_ids, max_length=50, temperature=1.0, 
                 top_k=None, top_p=None):
        """
        Autoregressive generation.
        
        Args:
            input_ids: [batch, seq_len] starting tokens
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            top_p: Nucleus sampling (optional)
            
        Returns:
            generated: [batch, max_length] generated tokens
        """
        self.eval()
        generated = input_ids
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Get logits for next token
                logits = self.forward(generated)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    indices_to_remove = next_token_logits < torch.topk(
                        next_token_logits, top_k
                    )[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(
                        next_token_logits, descending=True
                    )
                    cumulative_probs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    
                    # Remove tokens with cumulative prob > top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated


# Convenience function for common configurations
def create_small_model():
    """Create a small model for testing (13M params)."""
    return SpikingTransformer(
        vocab_size=50257,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        T=4,
        max_seq_len=512
    )


# Add convenience functions
def create_fixed_rate_model():
    """Create model with fixed rate encoding."""
    return SpikingTransformer(
        vocab_size=50257,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        T=4,
        max_seq_len=512,
        fixed_encoding='rate'
    )


def create_fixed_temporal_model():
    """Create model with fixed temporal encoding."""
    return SpikingTransformer(
        vocab_size=50257,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        T=4,
        max_seq_len=512,
        fixed_encoding='temporal'
    )


def create_medium_model():
    """Create a medium model (125M params - GPT-2 small scale)."""
    return SpikingTransformer(
        vocab_size=50257,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        T=4,
        max_seq_len=1024,
        fixed_encoding=None  # Adaptive
    )


def create_large_model():
    """Create a large model (1B+ params)."""
    return SpikingTransformer(
        vocab_size=50257,
        d_model=1536,
        nhead=16,
        num_layers=24,
        dim_feedforward=6144,
        T=4,
        max_seq_len=2048
    )
def create_large_model():
    """Create 350M parameter model (GPT-2 Medium scale)."""
    return SpikingTransformer(
        vocab_size=50257,
        d_model=1024,      # GPT-2 Medium
        nhead=16,
        num_layers=24,
        dim_feedforward=4096,
        T=4,
        max_seq_len=1024,
        dropout=0.1,
        fixed_encoding=None  # Adaptive
    )


def create_xlarge_model():
    """Create 1.5B parameter model (GPT-2 Large scale)."""
    return SpikingTransformer(
        vocab_size=50257,
        d_model=1536,      # GPT-2 Large
        nhead=25,
        num_layers=48,
        dim_feedforward=6144,
        T=4,
        max_seq_len=1024,
        dropout=0.1,
        fixed_encoding=None
    )