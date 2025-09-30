"""
Quick training script for tiny model.
Perfect for testing and iteration.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import create_small_model
from src.training.loss import compute_adaptive_loss
from tqdm import tqdm
import pickle


class SimpleDataset(Dataset):
    """Simple dataset wrapper."""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data['input_ids'])
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data['input_ids'][idx]),
            'labels': torch.tensor(self.data['input_ids'][idx])
        }


def train_quick(num_epochs=3, batch_size=4, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Quick training loop for experimentation.
    """
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    with open('data/train_tiny.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/val_tiny.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    train_loader = DataLoader(
        SimpleDataset(train_data),
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        SimpleDataset(val_data),
        batch_size=batch_size,
        shuffle=False
    )
    
    # Create model
    print("Creating model...")
    model = create_small_model()
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'spike_rate': [],
        'encoding_usage': []
    }
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...\n")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_spike_rates = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward
            logits = model(input_ids)
            
            # Simple loss (just cross-entropy for now)
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Track
            train_loss += loss.item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids)
                
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100
                )
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}\n")
    
    # Save model
    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / 'tiny_model.pt')
    print(f"✓ Model saved to {save_dir / 'tiny_model.pt'}")
    
    # Plot results
    plot_training_curves(history)
    
    return model, history


def plot_training_curves(history):
    """Plot training curves."""
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    save_path = Path('results/plots')
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path / 'training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {save_path / 'training_curves.png'}")
    
    plt.show()


if __name__ == '__main__':
    print("="*60)
    print("Quick Training - Spiking Adaptive LLM")
    print("="*60 + "\n")
    
    model, history = train_quick(
        num_epochs=3,
        batch_size=4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)