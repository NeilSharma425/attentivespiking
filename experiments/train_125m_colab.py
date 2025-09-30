"""
Train 125M parameter model optimized for Google Colab.
Estimated time: 8-10 hours on T4/V100.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from pathlib import Path
import sys
from tqdm import tqdm
import json
import time

# Add parent directory to path
sys.path.append('/content/attentivespiking')

from src.models.transformer import SpikingTransformer
torch.manual_seed(42)

class SimpleDataset:
    """Wrapper for tokenized dataset."""
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_ids'])
        }


def prepare_dataset_fast(num_samples=10000, max_length=256):
    """Prepare dataset quickly."""
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading WikiText-103...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    def tokenize(examples):
        texts = [t for t in examples['text'] if len(t.strip()) > 100]
        if not texts:
            return {'input_ids': []}
        
        result = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
        return result
    
    print(f"Tokenizing {num_samples} training samples...")
    train = dataset['train'].select(range(num_samples))
    train = train.map(tokenize, batched=True, remove_columns=['text'])
    
    print(f"Tokenizing {num_samples//10} validation samples...")
    val = dataset['validation'].select(range(num_samples//10))
    val = val.map(tokenize, batched=True, remove_columns=['text'])
    
    return train, val


def train_125m_colab(num_epochs=5, checkpoint_every=500):
    """
    Train 125M parameter model on Colab.
    Saves checkpoints to Google Drive.
    """
    
    print("="*80)
    print("TRAINING 125M PARAMETER MODEL ON COLAB")
    print("="*80 + "\n")
    
    # Mount Google Drive
    try:
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
        print("Google Drive mounted\n")
    except:
        print("Warning: Not running on Colab or Drive already mounted\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\n")
    
    # Checkpoint directory
    checkpoint_dir = Path('/content/drive/MyDrive/spike_checkpoints_125m')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}\n")
    
    # Prepare data
    train_data, val_data = prepare_dataset_fast(
        num_samples=10000,
        max_length=256
    )
    
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        return {'input_ids': input_ids}
    
    train_loader = DataLoader(
        SimpleDataset(train_data),
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        SimpleDataset(val_data),
        batch_size=4,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"Dataset loaded: {len(train_data)} train, {len(val_data)} val")
    print(f"Batches per epoch: {len(train_loader)}\n")
    
    # Create model (125M parameters)
    print("Creating 125M parameter model...")
    model = SpikingTransformer(
        vocab_size=50257,
        d_model=768,
        nhead=12,
        num_layers=12,
        dim_feedforward=3072,
        T=4,
        max_seq_len=512,
        dropout=0.1
    )
    model = model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1
    )
    
    scaler = torch.amp.GradScaler('cuda')
    
    # Resume from checkpoint if exists
    resume_path = checkpoint_dir / 'latest.pt'
    start_epoch = 0
    global_step = 0
    history = {'train_loss': [], 'val_loss': []}
    
    if resume_path.exists():
        print("Found checkpoint, resuming...")
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint.get('global_step', 0)
        history = checkpoint['history']
        print(f"Resumed from epoch {start_epoch}, step {global_step}\n")
    
    # Training configuration
    print("Training configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: 4")
    print(f"  Checkpoint every: {checkpoint_every} steps")
    print(f"  Learning rate: 3e-4\n")
    
    # Training loop
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(start_epoch, num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print(f"{'='*80}")
        
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast():
                logits = model(input_ids)
                
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                    ignore_index=-100
                )
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            train_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{train_loss/(batch_idx+1):.4f}"
            })
            
            # Save checkpoint
            if (batch_idx + 1) % checkpoint_every == 0:
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'batch': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'train_loss': train_loss / (batch_idx + 1)
                }, checkpoint_dir / 'latest.pt')
                
                elapsed = time.time() - start_time
                print(f"\n  Checkpoint saved (step {global_step}, {elapsed/3600:.1f}h elapsed)")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        print("\nValidating...")
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                
                with torch.cuda.amp.autocast():
                    logits = model(input_ids)
                    
                    loss = nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        input_ids[:, 1:].reshape(-1),
                        ignore_index=-100
                    )
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        print(f"\nEpoch {epoch+1} Results:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), checkpoint_dir / 'best_model.pt')
            print(f"  Best model saved!")
        
        # Save epoch checkpoint
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'history': history,
            'best_val_loss': best_val_loss
        }, checkpoint_dir / f'epoch_{epoch+1}.pt')
        
        # Save history JSON
        with open(checkpoint_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("="*80)
    
    return model, history


if __name__ == '__main__':
    model, history = train_125m_colab(
        num_epochs=5,
        checkpoint_every=500
    )
