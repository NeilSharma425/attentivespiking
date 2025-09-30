"""
Checkpoint save/load utilities.
"""

import torch
from pathlib import Path


def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                   best_val_loss, path):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        global_step: Global training step
        best_val_loss: Best validation loss so far
        path: Save path
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        'best_val_loss': best_val_loss
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, device='cuda'):
    """
    Load checkpoint.
    
    Args:
        path: Checkpoint path
        device: Device to load to
        
    Returns:
        checkpoint dict
    """
    checkpoint = torch.load(path, map_location=device)
    return checkpoint