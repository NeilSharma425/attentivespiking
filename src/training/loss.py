"""
Multi-objective loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_adaptive_loss(logits, targets, encoding_weights, spikes,
                         alpha=1.0, beta=0.1, gamma=0.05):
    """
    Multi-objective loss for adaptive spike encoding.
    
    Components:
    1. Task loss: Standard cross-entropy for language modeling
    2. Sparsity loss: Encourage efficient spike usage
    3. Diversity loss: Encourage using multiple encoding types
    
    Args:
        logits: Model predictions [batch, seq_len, vocab_size]
        targets: Ground truth tokens [batch, seq_len]
        encoding_weights: Encoding selection weights [batch, seq_len, num_encodings]
        spikes: Generated spike trains [batch, seq_len, d_model, T]
        alpha: Weight for task loss
        beta: Weight for sparsity loss
        gamma: Weight for diversity loss
        
    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary with individual loss components
    """
    # 1. Task Loss (cross-entropy)
    logits_flat = logits.reshape(-1, logits.size(-1))
    targets_flat = targets.reshape(-1)
    
    task_loss = F.cross_entropy(logits_flat, targets_flat, ignore_index=-100)
    
    # 2. Sparsity Loss (minimize average firing rate)
    spike_rate = spikes.mean()  # Average spikes per neuron per timestep
    sparsity_loss = spike_rate
    
    # 3. Diversity Loss (encourage balanced encoding usage)
    # Compute entropy of encoding selection
    # Higher entropy = more diverse usage
    encoding_probs = encoding_weights.mean(dim=(0, 1))  # [num_encodings]
    encoding_entropy = -(encoding_probs * torch.log(encoding_probs + 1e-8)).sum()
    
    # We want to maximize entropy, so minimize negative entropy
    diversity_loss = -encoding_entropy
    
    # Combined loss
    total_loss = (alpha * task_loss + 
                  beta * sparsity_loss +
                  gamma * diversity_loss)
    
    # Return detailed breakdown
    loss_dict = {
        'total': total_loss.item(),
        'task': task_loss.item(),
        'sparsity': sparsity_loss.item(),
        'diversity': diversity_loss.item(),
        'spike_rate': spike_rate.item(),
        'encoding_entropy': encoding_entropy.item()
    }
    
    return total_loss, loss_dict


class AdaptiveLoss(nn.Module):
    """
    Wrapper class for adaptive loss.
    """
    
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def forward(self, logits, targets, encoding_weights, spikes):
        return compute_adaptive_loss(
            logits, targets, encoding_weights, spikes,
            self.alpha, self.beta, self.gamma
        )
    
    def update_weights(self, alpha=None, beta=None, gamma=None):
        """Update loss weights (for curriculum learning)."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma