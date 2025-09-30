"""
Scale to 350M parameters (GPT-2 Medium).
Requires: 1 GPU with 24GB+ memory (RTX 3090, A5000, A6000)
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

sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import create_large_model


def prepare_large_dataset(num_samples=50000, max_length=1024):
    """Prepare dataset for large-scale training."""
    
    print("Preparing large dataset...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Use full WikiText-103
    dataset = load_dataset('wikitext', 'wikitext-103-v1')
    
    def tokenize_function(examples):
        texts = [text for text in examples['text'] if len(text.strip()) > 100]
        if not texts:
            return {'input_ids': [], 'attention_mask': []}
        
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_tensors='pt'
        )
    
    # Take subset
    train_size = min(num_samples, len(dataset['train']))
    val_size = min(num_samples // 10, len(dataset['validation']))
    
    train_dataset = dataset['train'].select(range(train_size))
    val_dataset = dataset['validation'].select(range(val_size))
    
    print(f"Tokenizing {train_size} training samples...")
    train_tokenized = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=4
    )
    
    print(f"Tokenizing {val_size} validation samples...")
    val_tokenized = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names,
        num_proc=4
    )
    
    return train_tokenized, val_tokenized


def train_350m(num_epochs=10, batch_size=1, accumulation_steps=16):
    """
    Train 350M parameter model.
    
    Effective batch size = batch_size * accumulation_steps * num_gpus
    With batch_size=1, accumulation=16: effective_batch=16 (on 1 GPU)
    """
    
    print("\n" + "="*80)
    print("SCALING TO 350M PARAMETERS")
    print("="*80 + "\n")
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: GPU required for 350M model!")
        print("Estimated memory: 6-8GB")
        return
    
    device = torch.device('cuda')
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {gpu_memory:.1f} GB")
    
    if gpu_memory < 16:
        print("\nWARNING: Less than 16GB GPU memory detected.")
        print("Training may be slow or fail. Consider:")
        print("  1. Reducing batch_size to 1")
        print("  2. Using gradient checkpointing")
        print("  3. Using mixed precision (FP16)")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Prepare data
    train_data, val_data = prepare_large_dataset(
        num_samples=50000,
        max_length=512  # Shorter for memory
    )
    
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("\nCreating 350M parameter model...")
    model = create_large_model()
    model = model.to(device)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-5,  # Lower LR for large models
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    # Scheduler
    total_steps = len(train_loader) * num_epochs // accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': []
    }
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {accumulation_steps}")
    print(f"  Effective batch size: {batch_size * accumulation_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Learning rate: 6e-5")
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*80}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*80}")
        
        # Train