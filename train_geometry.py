"""
IMPROVED Training Script for Stage 1: Geometry Model
Fixed to work with config_improved.py

Key improvements:
- Warmup learning rate schedule
- Enhanced loss with 5 components
- Normal prediction
- Better logging
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime
import json
import random

# Import project modules
from config import config
from models.geometry_model import GeometryModel, geometry_loss

# You'll need to import your data loader
# from load_data import ShoeDataset, custom_collate_fn


class ImprovedGeometryTrainer:
    """Improved Trainer for Stage 1: Geometry prediction"""
    
    def __init__(self, config):
        self.config = config
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(config.log_dir).mkdir(exist_ok=True, parents=True)
        
        print("="*70)
        print("STAGE 1: IMPROVED GEOMETRY MODEL TRAINING")
        print("="*70)
        print(f"Device: {config.device}")
        print(f"Batch Size: {config.batch_size_stage1}")
        print(f"Learning Rate: {config.learning_rate_stage1}")
        print(f"Epochs: {config.num_epochs_stage1}")
        print(f"Point Cloud Size: {config.num_points}")
        print(f"Freeze Encoder: {config.freeze_image_encoder}")
        print("="*70)
        
        # Load dataset
        print("\nLoading dataset...")
        # IMPORTANT: Uncomment and use your actual dataset
        # from load_data import ShoeDataset, custom_collate_fn
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
        # print(f"✓ Dataset loaded: {len(self.dataset)} shoes")
        
        # FOR TESTING - Remove this when you have real data
        print("⚠️  WARNING: Using dummy dataloader for testing")
        print("   Uncomment the dataset code above to use real data")
        self.dataloader = []  # Placeholder
        
        # Initialize model
        print("\nInitializing improved model...")
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
        
        # Learning rate scheduler with warmup
        self.warmup_epochs = 5
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs_stage1 - self.warmup_epochs,
            eta_min=1e-6  # Don't go below 1e-6
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_history = {
            'epoch': [], 
            'loss': [], 
            'chamfer': [],
            'normal': [],
            'edge': [],
            'smooth': [],
            'coverage': [],
            'lr': []
        }
        
        print(f"\n✓ Trainer initialized")
        print(f"  Warmup epochs: {self.warmup_epochs}")
        print(f"  Scheduler: CosineAnnealingLR (min LR: 1e-6)")
        print(f"  Early stopping patience: {config.patience}")
    
    def _sample_points(self, vertices, num_samples):
        """Randomly sample points from vertices"""
        num_verts = vertices.shape[0]
        if num_verts >= num_samples:
            indices = torch.randperm(num_verts)[:num_samples]
            return vertices[indices]
        else:
            # If not enough vertices, repeat some
            indices = torch.randint(0, num_verts, (num_samples,))
            return vertices[indices]

    def _shuffle_views(self, images_dict):
        """
        Randomly shuffle the order of views to enforce permutation invariance.
        This ensures the model learns view-content patterns, not position patterns.
        """
        view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']

        # Create random permutation
        perm = list(range(6))
        random.shuffle(perm)

        # Shuffle images
        shuffled_images = {}
        stacked_images = torch.stack([images_dict[name] for name in view_names], dim=1)  # (B, 6, C, H, W)
        shuffled_stacked = stacked_images[:, perm, :, :, :]

        for i, name in enumerate(view_names):
            shuffled_images[name] = shuffled_stacked[:, i, :, :, :]

        return shuffled_images
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_losses = {'chamfer': 0, 'normal': 0, 'edge': 0, 'smooth': 0, 'coverage': 0}
        num_batches = len(self.dataloader)
        
        if num_batches == 0:
            print("⚠️  No data to train on! Please uncomment dataset loading code.")
            return 0.0, epoch_losses
        
        # Warmup learning rate
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate_stage1 * lr_scale
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move to device
            images = {k: v.to(self.config.device) for k, v in batch['images'].items()}

            # Randomly shuffle views to enforce permutation invariance
            images = self._shuffle_views(images)

            # Process each shoe in batch (Y values are lists)
            batch_loss = 0
            batch_loss_dict = {'chamfer': 0, 'normal': 0, 'edge': 0, 'smooth': 0, 'coverage': 0}
            
            for i in range(len(batch['vertices'])):
                gt_vertices = batch['vertices'][i].to(self.config.device)
                
                # Forward pass (single shoe)
                images_single = {k: v[i:i+1] for k, v in images.items()}

                # Forward pass - now returns points AND normals
                pred_points, pred_normals = self.model(images_single)
                
                # Sample GT vertices to match prediction size
                gt_vertices_sample = self._sample_points(gt_vertices, self.config.num_points)
                
                # Compute enhanced loss with all 5 components
                loss, loss_dict = geometry_loss(
                    pred_points[0], 
                    pred_normals[0],
                    gt_vertices_sample,
                    lambda_chamfer=self.config.lambda_chamfer,
                    lambda_normal=self.config.lambda_normal,
                    lambda_edge=self.config.lambda_edge,
                    lambda_smooth=self.config.lambda_smooth,
                    lambda_coverage=self.config.lambda_coverage
                )
                
                batch_loss += loss
                for key in batch_loss_dict.keys():
                    batch_loss_dict[key] += loss_dict[key]
            
            # Average loss over batch
            loss = batch_loss / len(batch['vertices'])
            for key in batch_loss_dict.keys():
                batch_loss_dict[key] /= len(batch['vertices'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            for key in epoch_losses.keys():
                epoch_losses[key] += batch_loss_dict[key]
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Epoch [{epoch+1}/{self.config.num_epochs_stage1}] "
                      f"Batch [{batch_idx+1}/{num_batches}] ({progress:.1f}%)")
                print(f"    Total: {loss.item():.6f} | "
                      f"Chamfer: {batch_loss_dict['chamfer']:.6f} | "
                      f"Normal: {batch_loss_dict['normal']:.6f} | "
                      f"Edge: {batch_loss_dict['edge']:.6f}")
        
        avg_loss = epoch_loss / num_batches
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        
        return avg_loss, avg_losses
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING IMPROVED TRAINING")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs_stage1):
            epoch_start = time.time()
            
            # Train one epoch
            avg_loss, avg_losses = self.train_epoch(epoch)
            
            # Step scheduler after warmup
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs_stage1} Summary:")
            print(f"  Total Loss:    {avg_loss:.6f}")
            print(f"  Chamfer:       {avg_losses['chamfer']:.6f}")
            print(f"  Normal:        {avg_losses['normal']:.6f}")
            print(f"  Edge:          {avg_losses['edge']:.6f}")
            print(f"  Smooth:        {avg_losses['smooth']:.6f}")
            print(f"  Coverage:      {avg_losses['coverage']:.6f}")
            print(f"  Time:          {epoch_time:.2f}s")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"{'─'*70}\n")
            
            # Save history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['loss'].append(avg_loss)
            self.train_history['lr'].append(current_lr)
            for key in avg_losses.keys():
                self.train_history[key].append(avg_losses[key])
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"geometry_improved_epoch{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'history': self.train_history
                }, checkpoint_path)
                print(f"✓ Checkpoint saved: {checkpoint_path}\n")
            
            # Early stopping
            if avg_loss < self.best_loss - self.config.min_delta:
                self.best_loss = avg_loss
                self.patience_counter = 0
                
                # Save best model
                best_path = Path(self.config.checkpoint_dir) / "geometry_improved_best.pth"
                torch.save(self.model.state_dict(), best_path)
                print(f"✓ Best model saved: {best_path} (Loss: {avg_loss:.6f})\n")
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
        print("IMPROVED STAGE 1 TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {hours}h {minutes}m")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Final model: {self.config.checkpoint_dir}/geometry_improved_best.pth")
        print("="*70)
        
        # Save training history
        history_path = Path(self.config.log_dir) / "geometry_improved_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"\n✓ Training history saved: {history_path}")
        
        return self.model


def main():
    """Main training script"""
    print("\n" + "="*70)
    print("IMPROVED MULTI-VIEW TO 3D MESH - STAGE 1: GEOMETRY")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Print configuration
    print("\nConfiguration:")
    print(f"  Learning Rate: {config.learning_rate_stage1}")
    print(f"  Epochs: {config.num_epochs_stage1}")
    print(f"  Points: {config.num_points}")
    print(f"  Loss weights:")
    print(f"    - Chamfer:  {config.lambda_chamfer}")
    print(f"    - Normal:   {config.lambda_normal}")
    print(f"    - Edge:     {config.lambda_edge}")
    print(f"    - Smooth:   {config.lambda_smooth}")
    print(f"    - Coverage: {config.lambda_coverage}")
    
    # Initialize trainer
    trainer = ImprovedGeometryTrainer(config)
    
    # Train
    model = trainer.train()
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Improved Stage 1 Complete!")
    print("\nKey improvements applied:")
    print("  • Poisson reconstruction (not convex hull)")
    print("  • Normal prediction for surface orientation")
    print("  • 5-component enhanced loss function")
    print("  • Higher learning rate with warmup (2e-4 → 1e-6)")
    print("  • 8192 points for more detail")
    print("\nReady for Stage 2 (Texture Training)")


if __name__ == "__main__":
    main()