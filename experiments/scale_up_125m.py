"""
Scale up to GPT-2 small size (125M parameters).
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import create_medium_model


def prepare_larger_dataset(num_samples=5000, max_length=512):
    """Prepare larger dataset for 125M model."""
    
    print("Preparing larger dataset...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    def tokenize_function(examples):
        texts = [text for text in examples['text'] if len(text.strip()) > 50]
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    # More data
    train_dataset = dataset['train'].select(range(min(num_samples, len(dataset['train']))))
    train_tokenized = train_dataset.map(tokenize_function, batched=True, 
                                       remove_columns=train_dataset.column_names)
    
    val_dataset = dataset['validation'].select(range(min(num_samples//10, len(dataset['validation']))))
    val_tokenized = val_dataset.map(tokenize_function, batched=True,
                                    remove_columns=val_dataset.column_names)
    
    print(f"✓ Prepared {len(train_tokenized)} train, {len(val_tokenized)} val samples")
    
    return train_tokenized, val_tokenized


def train_125m_model(num_epochs=5):
    """Train 125M parameter model."""
    
    print("\n" + "="*80)
    print("SCALING TO 125M PARAMETERS")
    print("="*80 + "\n")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    if device == 'cpu':
        print("⚠️  WARNING: Training 125M model on CPU will be VERY slow!")
        print("    Recommended: Use GPU or reduce model size")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Prepare data
    train_data, val_data = prepare_larger_dataset(num_samples=5000, max_length=512)
    
    train_loader = DataLoader(
        train_data,
        batch_size=2,  # Small batch for memory
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=2,
        shuffle=False,
        num_workers=2
    )
    
    # Create model
    print("\nCreating 125M parameter model...")
    model = create_medium_model()
    model = model.to(device)
    
    # Optimizer with gradient accumulation
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    accumulation_steps = 4  # Effective batch size = 2 * 4 = 8
    
    history = {'train_loss': [], 'val_loss': []}
    
    print(f"\nTraining for {num_epochs} epochs...")
    print(f"Effective batch size: {2 * accumulation_steps}\n")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(device)
            
            logits = model(input_ids)
            
            loss = nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
                ignore_index=-100
            )
            
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': f"{loss.item() * accumulation_steps:.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(device)
                
                logits = model(input_ids)
                
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                    ignore_index=-100
                )
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}\n")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            checkpoint_dir = Path('checkpoints')
            checkpoint_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, checkpoint_dir / f'medium_model_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), 'checkpoints/medium_model_final.pt')
    print("\n✓ Training complete!")
    print("✓ Model saved to checkpoints/medium_model_final.pt")
    
    return model, history


if __name__ == '__main__':
    model, history = train_125m_model(num_epochs=5)