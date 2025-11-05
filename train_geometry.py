"""
IMPROVED Training Script for Stage 1: Geometry Model
Key improvements:
- Better learning rate schedule with warmup
- Detailed loss component monitoring
- Better checkpointing
- Visualization of training progress
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime
import json
import random
import matplotlib.pyplot as plt

# Import project modules
from config import config
from models.geometry_model import GeometryModel, improved_geometry_loss
# You'll still need your data loader
# from load_data import ShoeDataset, custom_collate_fn


def get_scheduler(optimizer, config):
    """
    Create improved learning rate scheduler
    """
    if config.lr_schedule == 'cosine':
        # Standard cosine annealing
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs_stage1,
            eta_min=config.lr_min
        )
    
    elif config.lr_schedule == 'cosine_restart':
        # Cosine with warm restarts (helps escape local minima)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=20,  # Restart every 20 epochs
            T_mult=2,  # Double the restart interval each time
            eta_min=config.lr_min
        )
    
    elif config.lr_schedule == 'plateau':
        # Reduce on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=config.lr_min
        )
    
    else:
        # Default: cosine
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs_stage1,
            eta_min=config.lr_min
        )
    
    return scheduler


def get_warmup_scheduler(optimizer, warmup_epochs, base_lr):
    """
    Create warmup scheduler that gradually increases LR
    """
    def warmup_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)


class ImprovedGeometryTrainer:
    """IMPROVED Trainer for Stage 1: Geometry prediction"""
    
    def __init__(self, config):
        self.config = config
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.log_dir).mkdir(exist_ok=True)
        
        print("="*70)
        print("STAGE 1: IMPROVED GEOMETRY MODEL TRAINING")
        print("="*70)
        config.print_config()
        
        # Load dataset
        print("\nLoading dataset...")
        # UNCOMMENT WHEN YOU HAVE YOUR DATA LOADER
        # self.dataset = ShoeDataset(
        #     obj_dir=config.obj_dir,
        #     images_dir=config.images_dir,
        #     verify_mappings=True,
        #     image_size=config.image_size
        # )
        # 
        # self.dataloader = DataLoader(
        #     self.dataset,
        #     batch_size=config.batch_size_stage1,
        #     shuffle=True,
        #     num_workers=config.num_workers,
        #     collate_fn=custom_collate_fn
        # )
        # 
        # print(f"✓ Dataset loaded: {len(self.dataset)} shoes")
        
        # Initialize model
        print("\nInitializing model...")
        self.model = GeometryModel(
            num_points=config.num_points,
            freeze_encoder=config.freeze_image_encoder,
            hidden_dim=config.hidden_dim
        ).to(config.device)
        
        total_params, trainable_params = self.model.count_parameters()
        print(f"✓ Model initialized")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=config.learning_rate_stage1,
            weight_decay=config.weight_decay_stage1
        )
        
        # Learning rate schedulers
        self.warmup_scheduler = get_warmup_scheduler(
            self.optimizer, 
            config.warmup_epochs, 
            config.learning_rate_stage1
        )
        self.main_scheduler = get_scheduler(self.optimizer, config)
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        # Enhanced training history
        self.train_history = {
            'epoch': [],
            'loss_total': [],
            'loss_chamfer': [],
            'loss_coverage': [],
            'loss_edge': [],
            'loss_normal': [],
            'loss_smooth': [],
            'lr': []
        }
    
    def _sample_points(self, vertices, num_samples):
        """Randomly sample points from vertices"""
        num_verts = vertices.shape[0]
        if num_verts >= num_samples:
            indices = torch.randperm(num_verts)[:num_samples]
            return vertices[indices]
        else:
            indices = torch.randint(0, num_verts, (num_samples,))
            return vertices[indices]

    def _shuffle_views(self, images_dict):
        """Randomly shuffle the order of views to enforce permutation invariance"""
        view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        perm = list(range(6))
        random.shuffle(perm)
        
        shuffled_images = {}
        stacked_images = torch.stack([images_dict[name] for name in view_names], dim=1)
        shuffled_stacked = stacked_images[:, perm, :, :, :]
        
        for i, name in enumerate(view_names):
            shuffled_images[name] = shuffled_stacked[:, i, :, :, :]
        
        return shuffled_images
    
    def train_epoch(self, epoch):
        """Train for one epoch with detailed loss tracking"""
        self.model.train()
        epoch_losses = {
            'total': 0,
            'chamfer': 0,
            'coverage': 0,
            'edge': 0,
            'normal': 0,
            'smooth': 0
        }
        num_batches = len(self.dataloader)
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move to device
            images = {k: v.to(self.config.device) for k, v in batch['images'].items()}
            
            # Randomly shuffle views
            images = self._shuffle_views(images)
            
            # Process each shoe in batch
            batch_loss = 0
            batch_loss_dict = {k: 0 for k in epoch_losses.keys()}
            
            for i in range(len(batch['vertices'])):
                gt_vertices = batch['vertices'][i].to(self.config.device)
                
                # Forward pass
                images_single = {k: v[i:i+1] for k, v in images.items()}
                pred_points = self.model(images_single)
                
                # Sample GT vertices
                gt_vertices_sample = self._sample_points(gt_vertices, self.config.num_points)
                
                # Compute loss with all components
                loss, loss_dict = improved_geometry_loss(
                    pred_points[0], 
                    gt_vertices_sample,
                    lambda_chamfer=self.config.lambda_chamfer,
                    lambda_coverage=self.config.lambda_coverage,
                    lambda_edge=self.config.lambda_edge,
                    lambda_normal=self.config.lambda_normal,
                    lambda_smooth=self.config.lambda_smooth
                )
                
                batch_loss += loss
                
                # Accumulate loss components
                for key in batch_loss_dict.keys():
                    if key in loss_dict:
                        batch_loss_dict[key] += loss_dict[key]
            
            # Average loss over batch
            loss = batch_loss / len(batch['vertices'])
            for key in batch_loss_dict.keys():
                batch_loss_dict[key] /= len(batch['vertices'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(), 
                self.config.grad_clip_norm
            )
            self.optimizer.step()
            
            # Accumulate epoch losses
            for key in epoch_losses.keys():
                epoch_losses[key] += batch_loss_dict[key]
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Epoch [{epoch+1}/{self.config.num_epochs_stage1}] "
                      f"Batch [{batch_idx+1}/{num_batches}] ({progress:.1f}%)")
                print(f"    Total: {batch_loss_dict['total']:.6f} | "
                      f"Chamfer: {batch_loss_dict['chamfer']:.6f} | "
                      f"Coverage: {batch_loss_dict['coverage']:.6f}")
                print(f"    Edge: {batch_loss_dict['edge']:.6f} | "
                      f"Normal: {batch_loss_dict['normal']:.6f} | "
                      f"Smooth: {batch_loss_dict['smooth']:.6f}")
        
        # Average epoch losses
        for key in epoch_losses.keys():
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def plot_training_progress(self):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_history['epoch'], self.train_history['loss_total'])
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.train_history['epoch'], self.train_history['lr'])
        axes[0, 1].set_title('Learning Rate')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('LR')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Loss components
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['loss_chamfer'], label='Chamfer')
        axes[1, 0].plot(self.train_history['epoch'], self.train_history['loss_coverage'], label='Coverage')
        axes[1, 0].set_title('Main Loss Components')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Regularization terms
        axes[1, 1].plot(self.train_history['epoch'], self.train_history['loss_edge'], label='Edge')
        axes[1, 1].plot(self.train_history['epoch'], self.train_history['loss_normal'], label='Normal')
        axes[1, 1].plot(self.train_history['epoch'], self.train_history['loss_smooth'], label='Smooth')
        axes[1, 1].set_title('Regularization Terms')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = Path(self.config.log_dir) / 'training_progress.png'
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        print(f"✓ Training plot saved: {plot_path}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs_stage1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train one epoch
            epoch_losses = self.train_epoch(epoch)
            
            # Learning rate scheduling
            if epoch < self.config.warmup_epochs:
                self.warmup_scheduler.step()
            else:
                if self.config.lr_schedule == 'plateau':
                    self.main_scheduler.step(epoch_losses['total'])
                else:
                    self.main_scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs_stage1} Summary:")
            print(f"  Total Loss:    {epoch_losses['total']:.6f}")
            print(f"  Chamfer Loss:  {epoch_losses['chamfer']:.6f}")
            print(f"  Coverage Loss: {epoch_losses['coverage']:.6f}")
            print(f"  Edge Loss:     {epoch_losses['edge']:.6f}")
            print(f"  Normal Loss:   {epoch_losses['normal']:.6f}")
            print(f"  Smooth Loss:   {epoch_losses['smooth']:.6f}")
            print(f"  Time:          {epoch_time:.2f}s")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"{'─'*70}\n")
            
            # Save history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['loss_total'].append(epoch_losses['total'])
            self.train_history['loss_chamfer'].append(epoch_losses['chamfer'])
            self.train_history['loss_coverage'].append(epoch_losses['coverage'])
            self.train_history['loss_edge'].append(epoch_losses['edge'])
            self.train_history['loss_normal'].append(epoch_losses['normal'])
            self.train_history['loss_smooth'].append(epoch_losses['smooth'])
            self.train_history['lr'].append(current_lr)
            
            # Plot progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.plot_training_progress()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"geometry_epoch{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.main_scheduler.state_dict(),
                    'loss': epoch_losses['total'],
                    'history': self.train_history
                }, checkpoint_path)
                print(f"✓ Checkpoint saved: {checkpoint_path}\n")
            
            # Early stopping
            if epoch_losses['total'] < self.best_loss - self.config.min_delta:
                self.best_loss = epoch_losses['total']
                self.patience_counter = 0
                
                # Save best model
                best_path = Path(self.config.checkpoint_dir) / "geometry_best.pth"
                torch.save(self.model.state_dict(), best_path)
                print(f"✓ Best model saved: {best_path} (Loss: {epoch_losses['total']:.6f})\n")
            else:
                self.patience_counter += 1
                print(f"⚠️  No improvement for {self.patience_counter} epochs\n")
                
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Training complete
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        
        print("\n" + "="*70)
        print("STAGE 1 TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {hours}h {minutes}m")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Final model: {self.config.checkpoint_dir}/geometry_best.pth")
        print("="*70)
        
        # Save training history
        history_path = Path(self.config.log_dir) / "geometry_training_history_improved.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"\n✓ Training history saved: {history_path}")
        
        # Final plot
        self.plot_training_progress()
        
        return self.model


def main():
    """Main training script"""
    print("\n" + "="*70)
    print("IMPROVED MULTI-VIEW TO 3D MESH - STAGE 1: GEOMETRY")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize trainer
    trainer = ImprovedGeometryTrainer(config)
    
    # Train
    model = trainer.train()
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Stage 1 Complete! Ready for Stage 2 (Texture Training)")


if __name__ == "__main__":
    main()