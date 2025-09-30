"""
Training loop and utilities.
"""

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import wandb
from pathlib import Path

from .loss import AdaptiveLoss
from .curriculum import TrainingCurriculum
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """
    Handles training loop for spiking transformer.
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 config, device, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.logger = logger
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config.get('weight_decay', 0.1)
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs']
        )
        
        # Loss
        self.loss_fn = AdaptiveLoss(
            alpha=config.get('alpha', 1.0),
            beta=config.get('beta', 0.1),
            gamma=config.get('gamma', 0.05)
        )
        
        # Curriculum
        self.curriculum = TrainingCurriculum()
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Logging
        if config.get('use_wandb', False):
            wandb.init(
                project=config['project_name'],
                config=config,
                name=config.get('run_name', None)
            )
        
        self.writer = SummaryWriter(config.get('log_dir', 'runs'))
        
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        loss_components = {'task': 0, 'sparsity': 0, 'diversity': 0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids)
            
            # Collect encoding weights and spikes from last forward
            # (This requires modifying model to store these - simplified here)
            encoding_weights = torch.zeros(1)  # Placeholder
            spikes = torch.zeros(1)  # Placeholder
            
            # Compute loss
            loss, loss_dict = self.loss_fn(
                logits[:, :-1].contiguous(),
                labels[:, 1:].contiguous(),
                encoding_weights,
                spikes
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', None):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['grad_clip']
                )
            
            self.optimizer.step()
            
            # Track
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Log to tensorboard/wandb
            if self.global_step % self.config.get('log_interval', 10) == 0:
                self.log_metrics({
                    'train/loss': loss.item(),
                    'train/task_loss': loss_dict['task'],
                    'train/sparsity_loss': loss_dict['sparsity'],
                    'train/diversity_loss': loss_dict['diversity'],
                    'train/spike_rate': loss_dict['spike_rate'],
                    'train/lr': self.optimizer.param_groups[0]['lr']
                })
        
        # Epoch averages
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, loss_components
    
    def validate(self):
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                logits = self.model(input_ids)
                
                # Simplified loss (just task loss for validation)
                loss = nn.functional.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100
                )
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.config['num_epochs']} epochs")
        
        for epoch in range(self.current_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Update curriculum
            loss_weights = self.curriculum.get_loss_weights(epoch)
            self.loss_fn.update_weights(**loss_weights)
            
            # Train
            train_loss, loss_components = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Step scheduler
            self.scheduler.step()
            
            # Log epoch results
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            
            self.log_metrics({
                'epoch': epoch,
                'train/epoch_loss': train_loss,
                'val/loss': val_loss
            })
            
            # Save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint('best_model.pt')
                print(f"  Saved best model (val_loss: {val_loss:.4f})")
            
            if (epoch + 1) % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        print("\nTraining completed!")
        self.writer.close()
        if self.config.get('use_wandb', False):
            wandb.finish()
    
    def log_metrics(self, metrics):
        """Log metrics to tensorboard and wandb."""
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, self.global_step)
        
        if self.config.get('use_wandb', False):
            wandb.log(metrics, step=self.global_step)
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=self.global_step,
            best_val_loss=self.best_val_loss,
            path=checkpoint_dir / filename
        )
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = load_checkpoint(path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")