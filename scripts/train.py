"""
Main training script.
Run with: python scripts/train.py --config config/training_config.yaml
"""

import argparse
import torch
import yaml
from pathlib import Path

# Add src to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.models.transformer import SpikingTransformer
from src.training.trainer import Trainer
from src.utils.data import get_dataloaders
from src.utils.logging_utils import setup_logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger = setup_logging(config['logging'])
    
    # Data
    train_loader, val_loader = get_dataloaders(config['data'])
    
    # Model
    model = SpikingTransformer(**config['model'])
    model = model.to(device)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config['training'],
        device=device,
        logger=logger
    )
    
    # Train
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    trainer.train()


if __name__ == '__main__':
    main()