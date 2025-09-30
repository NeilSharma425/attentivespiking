"""
Prepare a tiny dataset for quick experimentation.
We'll use a small subset of WikiText for fast iteration.
"""

import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer
from pathlib import Path
import pickle

def prepare_tiny_dataset(num_samples=1000, max_length=128, save_dir='data'):
    """
    Prepare a tiny dataset for quick experiments.
    
    Args:
        num_samples: Number of training samples
        max_length: Maximum sequence length
        save_dir: Where to save processed data
    """
    print("Loading tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    print(f"Processing {num_samples} samples...")
    
    def tokenize_function(examples):
        # Filter out empty texts
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
    
    # Process train set
    train_dataset = dataset['train'].select(range(min(num_samples, len(dataset['train']))))
    train_tokenized = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    
    # Process validation set (smaller)
    val_samples = num_samples // 10
    val_dataset = dataset['validation'].select(range(min(val_samples, len(dataset['validation']))))
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    
    # Save
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    print(f"Saving to {save_path}...")
    
    # Save as pickle for quick loading
    with open(save_path / 'train_tiny.pkl', 'wb') as f:
        pickle.dump(train_tokenized, f)
    
    with open(save_path / 'val_tiny.pkl', 'wb') as f:
        pickle.dump(val_tokenized, f)
    
    print(f"✓ Dataset prepared!")
    print(f"  Training samples: {len(train_tokenized)}")
    print(f"  Validation samples: {len(val_tokenized)}")
    print(f"  Max length: {max_length}")
    
    return train_tokenized, val_tokenized


if __name__ == '__main__':
    train_data, val_data = prepare_tiny_dataset(
        num_samples=5000,  # Very small for quick testing
        max_length=256,
        save_dir='data'
    )