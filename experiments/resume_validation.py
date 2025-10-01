"""Resume validation from existing checkpoints"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
import numpy as np
from scipy import stats
from pathlib import Path
import json
import sys
from tqdm import tqdm

sys.path.append('/content/attentivespiking')
from src.models.transformer import SpikingTransformer
from experiments.train_125m_colab import SimpleDataset

# Import the validator but we'll modify the run
from experiments.focused_validation import FocusedValidator

def check_completed_runs(scale='125M'):
    """Check which runs are already complete."""
    checkpoint_dir = Path(f'/content/drive/MyDrive/validation_checkpoints_{scale}')
    training_dir = Path('/content/drive/MyDrive/training_checkpoints')
    
    completed = {
        'adaptive': [],
        'fixed': []
    }
    
    for seed in [42, 123, 456]:
        # Check final checkpoints
        if (checkpoint_dir / f'adaptive_seed{seed}.pt').exists():
            completed['adaptive'].append(seed)
            print(f"✓ {scale} adaptive seed {seed} - COMPLETE")
        else:
            print(f"✗ {scale} adaptive seed {seed} - MISSING")
        
        if (checkpoint_dir / f'fixed_seed{seed}.pt').exists():
            completed['fixed'].append(seed)
            print(f"✓ {scale} fixed seed {seed} - COMPLETE")
        else:
            print(f"✗ {scale} fixed seed {seed} - MISSING")
    
    return completed

def resume_validation():
    """Resume validation, skipping completed runs."""
    
    print("="*80)
    print("RESUMING VALIDATION")
    print("="*80)
    
    # Check what's done
    print("\nChecking 125M runs...")
    completed_125 = check_completed_runs('125M')
    
    print("\nChecking 350M runs...")
    completed_350 = check_completed_runs('350M')
    
    # Determine what needs to run
    all_seeds = [42, 123, 456]
    
    needs_125_adaptive = [s for s in all_seeds if s not in completed_125['adaptive']]
    needs_125_fixed = [s for s in all_seeds if s not in completed_125['fixed']]
    needs_350_adaptive = [s for s in all_seeds if s not in completed_350['adaptive']]
    needs_350_fixed = [s for s in all_seeds if s not in completed_350['fixed']]
    
    print("\n" + "="*80)
    print("REMAINING WORK:")
    print("="*80)
    print(f"125M adaptive: seeds {needs_125_adaptive if needs_125_adaptive else 'NONE - all done!'}")
    print(f"125M fixed:    seeds {needs_125_fixed if needs_125_fixed else 'NONE - all done!'}")
    print(f"350M adaptive: seeds {needs_350_adaptive if needs_350_adaptive else 'NONE - all done!'}")
    print(f"350M fixed:    seeds {needs_350_fixed if needs_350_fixed else 'NONE - all done!'}")
    
    # Run only what's needed
    validator = FocusedValidator()
    
    # Load dataset once
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    validator.tokenizer = tokenizer
    
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    def tokenize(examples):
        texts = [t for t in examples['text'] if len(t.strip()) > 100]
        if not texts:
            return {'input_ids': []}
        return tokenizer(texts, truncation=True, max_length=256, padding='max_length')
    
    train_data = dataset['train'].select(range(20000)).map(tokenize, batched=True, remove_columns=['text'])
    val_data = dataset['validation'].select(range(2000)).map(tokenize, batched=True, remove_columns=['text'])
    
    train_loader = DataLoader(SimpleDataset(train_data), batch_size=4, shuffle=True)
    val_loader = DataLoader(SimpleDataset(val_data), batch_size=4, shuffle=False)
    
    # Process 125M
    if needs_125_adaptive or needs_125_fixed:
        print("\n" + "="*80)
        print("TRAINING 125M REMAINING RUNS")
        print("="*80)
        
        for seed in all_seeds:
            print(f"\n{'='*40}")
            print(f"Seed {seed}")
            print(f"{'='*40}")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Adaptive
            if seed in needs_125_adaptive:
                print(f"\nTraining 125M adaptive seed {seed}...")
                model = SpikingTransformer(
                    vocab_size=50257, d_model=768, nhead=12,
                    num_layers=12, dim_feedforward=3072, T=4
                ).to(validator.device)
                
                best_val, _ = validator._train_to_convergence(
                    model, train_loader, val_loader,
                    patience=3, max_epochs=20,
                    model_name=f"125M_adaptive_seed{seed}"
                )
                
                # Save final checkpoint
                save_dir = Path('/content/drive/MyDrive/validation_checkpoints_125M')
                save_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), save_dir / f'adaptive_seed{seed}.pt')
                
                print(f"✓ Completed: {best_val:.4f}")
                del model
                torch.cuda.empty_cache()
            else:
                print(f"✓ 125M adaptive seed {seed} already complete")
            
            # Fixed
            if seed in needs_125_fixed:
                print(f"\nTraining 125M fixed seed {seed}...")
                model = SpikingTransformer(
                    vocab_size=50257, d_model=768, nhead=12,
                    num_layers=12, dim_feedforward=3072, T=4,
                    fixed_encoding='rate'
                ).to(validator.device)
                
                best_val, _ = validator._train_to_convergence(
                    model, train_loader, val_loader,
                    patience=3, max_epochs=20,
                    model_name=f"125M_fixed_seed{seed}"
                )
                
                # Save final checkpoint
                save_dir = Path('/content/drive/MyDrive/validation_checkpoints_125M')
                save_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), save_dir / f'fixed_seed{seed}.pt')
                
                print(f"✓ Completed: {best_val:.4f}")
                del model
                torch.cuda.empty_cache()
            else:
                print(f"✓ 125M fixed seed {seed} already complete")
    
    # Process 350M
    if needs_350_adaptive or needs_350_fixed:
        print("\n" + "="*80)
        print("TRAINING 350M REMAINING RUNS")
        print("="*80)
        
        for seed in all_seeds:
            print(f"\n{'='*40}")
            print(f"Seed {seed}")
            print(f"{'='*40}")
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Adaptive
            if seed in needs_350_adaptive:
                print(f"\nTraining 350M adaptive seed {seed}...")
                model = SpikingTransformer(
                    vocab_size=50257, d_model=1024, nhead=16,
                    num_layers=16, dim_feedforward=4096, T=4
                ).to(validator.device)
                
                best_val, _ = validator._train_to_convergence(
                    model, train_loader, val_loader,
                    patience=3, max_epochs=20,
                    model_name=f"350M_adaptive_seed{seed}"
                )
                
                # Save final checkpoint
                save_dir = Path('/content/drive/MyDrive/validation_checkpoints_350M')
                save_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), save_dir / f'adaptive_seed{seed}.pt')
                
                print(f"✓ Completed: {best_val:.4f}")
                del model
                torch.cuda.empty_cache()
            else:
                print(f"✓ 350M adaptive seed {seed} already complete")
            
            # Fixed
            if seed in needs_350_fixed:
                print(f"\nTraining 350M fixed seed {seed}...")
                model = SpikingTransformer(
                    vocab_size=50257, d_model=1024, nhead=16,
                    num_layers=16, dim_feedforward=4096, T=4,
                    fixed_encoding='rate'
                ).to(validator.device)
                
                best_val, _ = validator._train_to_convergence(
                    model, train_loader, val_loader,
                    patience=3, max_epochs=20,
                    model_name=f"350M_fixed_seed{seed}"
                )
                
                # Save final checkpoint
                save_dir = Path('/content/drive/MyDrive/validation_checkpoints_350M')
                save_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), save_dir / f'fixed_seed{seed}.pt')
                
                print(f"✓ Completed: {best_val:.4f}")
                del model
                torch.cuda.empty_cache()
            else:
                print(f"✓ 350M fixed seed {seed} already complete")
    
    print("\n" + "="*80)
    print("ALL TRAINING COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    resume_validation()
