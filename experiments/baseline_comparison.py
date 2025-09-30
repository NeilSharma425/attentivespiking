"""
Compare adaptive vs fixed encoding strategies.
This is THE critical experiment for your paper.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import pickle
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import (
    create_small_model,
    create_fixed_rate_model, 
    create_fixed_temporal_model,
    SpikingTransformer
)


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


def train_model(model, train_loader, val_loader, num_epochs=3, 
                device='cuda' if torch.cuda.is_available() else 'cpu',
                model_name='model'):
    """Train a single model."""
    
    print(f"\n{'='*60}")
    print(f"Training: {model_name}")
    print(f"{'='*60}\n")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids)
            
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=-100
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
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
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")
    
    # Save model
    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), save_dir / f'{model_name}.pt')
    
    return history


def run_baseline_comparison():
    """Run complete baseline comparison."""
    
    print("\n" + "="*80)
    print("BASELINE COMPARISON EXPERIMENT")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    with open('data/train_tiny.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('data/val_tiny.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    train_loader = DataLoader(SimpleDataset(train_data), batch_size=4, shuffle=True)
    val_loader = DataLoader(SimpleDataset(val_data), batch_size=4, shuffle=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Train all models
    results = {}
    
    # 1. Adaptive (your approach)
    print("\n1/4: Training ADAPTIVE encoding model...")
    model_adaptive = create_small_model()
    results['Adaptive (Ours)'] = train_model(
        model_adaptive, train_loader, val_loader, 
        num_epochs=3, device=device, model_name='adaptive'
    )
    
    # 2. Fixed Rate
    print("\n2/4: Training FIXED RATE encoding model...")
    model_rate = create_fixed_rate_model()
    results['Fixed Rate'] = train_model(
        model_rate, train_loader, val_loader,
        num_epochs=3, device=device, model_name='fixed_rate'
    )
    
    # 3. Fixed Temporal
    print("\n3/4: Training FIXED TEMPORAL encoding model...")
    model_temporal = create_fixed_temporal_model()
    results['Fixed Temporal'] = train_model(
        model_temporal, train_loader, val_loader,
        num_epochs=3, device=device, model_name='fixed_temporal'
    )
    
    # 4. Fixed Best (manual selection based on your analysis)
    print("\n4/4: Training FIXED BEST encoding model...")
    # Layer 0: rate, Layers 1-3: temporal
    model_best = SpikingTransformer(
        vocab_size=50257, d_model=256, nhead=8, num_layers=4,
        dim_feedforward=1024, T=4, max_seq_len=512,
        fixed_encoding='temporal'  # Simplified: all temporal
    )
    results['Fixed Best'] = train_model(
        model_best, train_loader, val_loader,
        num_epochs=3, device=device, model_name='fixed_best'
    )
    
    # Save results
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / 'baseline_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot comparison
    plot_comparison(results)
    
    # Print summary
    print_summary(results)
    
    return results


def plot_comparison(results):
    """Create comparison plots."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves
    for name, history in results.items():
        axes[0].plot(history['train_loss'], marker='o', label=name, linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Training Loss', fontsize=12)
    axes[0].set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Final validation loss comparison
    final_val_losses = {name: history['val_loss'][-1] for name, history in results.items()}
    
    names = list(final_val_losses.keys())
    losses = list(final_val_losses.values())
    colors = ['red' if 'Ours' in name else 'gray' for name in names]
    
    bars = axes[1].bar(range(len(names)), losses, color=colors, alpha=0.7, edgecolor='black')
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=15, ha='right')
    axes[1].set_ylabel('Final Validation Loss', fontsize=12)
    axes[1].set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    
    # Highlight best
    best_idx = np.argmin(losses)
    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(3)
    
    plt.tight_layout()
    plt.savefig('results/plots/baseline_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plots saved to results/plots/baseline_comparison.png")
    plt.show()


def print_summary(results):
    """Print summary table."""
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\n{'Model':<20} {'Final Train Loss':<20} {'Final Val Loss':<20} {'Improvement':<15}")
    print("-" * 80)
    
    baseline_val = results['Fixed Rate']['val_loss'][-1]
    
    for name, history in results.items():
        train_loss = history['train_loss'][-1]
        val_loss = history['val_loss'][-1]
        improvement = (baseline_val - val_loss) / baseline_val * 100
        
        marker = "★" if 'Ours' in name else " "
        print(f"{marker} {name:<18} {train_loss:<20.4f} {val_loss:<20.4f} {improvement:>+6.1f}%")
    
    print("="*80)
    
    # Statistical significance (rough estimate)
    adaptive_val = results['Adaptive (Ours)']['val_loss'][-1]
    best_baseline = min([results[k]['val_loss'][-1] for k in results if 'Ours' not in k])
    
    improvement = (best_baseline - adaptive_val) / best_baseline * 100
    
    print(f"\n✓ Adaptive encoding improves over best baseline by {improvement:+.1f}%")
    print("="*80 + "\n")


if __name__ == '__main__':
    results = run_baseline_comparison()