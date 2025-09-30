"""
Data loading and preprocessing.
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer


def get_dataloaders(config):
    """
    Create train and validation dataloaders.
    
    Args:
        config: Data configuration dict
        
    Returns:
        train_loader, val_loader
    """
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset_name = config.get('dataset', 'wikitext')
    
    if dataset_name == 'wikitext':
        dataset = load_dataset('wikitext', 'wikitext-103-v1')
    elif dataset_name == 'openwebtext':
        dataset = load_dataset('openwebtext')
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.get('max_length', 512),
            padding='max_length',
            return_tensors='pt'
        )
    
    # Tokenize
    tokenized_train = dataset['train'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )
    
    tokenized_val = dataset['validation'].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset['validation'].column_names
    )
    
    # Format for PyTorch
    tokenized_train.set_format('torch')
    tokenized_val.set_format('torch')
    
    # Create dataloaders
    train_loader = DataLoader(
        tokenized_train,
        batch_size=config.get('batch_size', 8),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        tokenized_val,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader