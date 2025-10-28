"""
Train Stage 1: Geometry Model
Fine-tune Multi-View TripoSR for 3D Shape Prediction
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime
import json

# Import project modules
from config import config
from models.geometry_model import MultiViewTripoSR, geometry_loss
from load_data import ShoeDataset, custom_collate_fn


class GeometryTrainer:
    """Trainer for Stage 1: Geometry prediction"""
    
    def __init__(self, config):
        self.config = config
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.log_dir).mkdir(exist_ok=True)
        
        print("="*70)
        print("STAGE 1: GEOMETRY MODEL TRAINING")
        print("="*70)
        print(f"Device: {config.device}")
        print(f"Batch Size: {config.batch_size_stage1}")
        print(f"Learning Rate: {config.learning_rate_stage1}")
        print(f"Freeze Encoder: {config.freeze_image_encoder}")
        print("="*70)
        
        # Load dataset
        print("\nLoading dataset...")
        self.dataset = ShoeDataset(
            obj_dir=config.obj_dir,
            images_dir=config.images_dir,
            verify_mappings=True
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size_stage1,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )
        
        print(f"✓ Dataset loaded: {len(self.dataset)} shoes")
        
        # Initialize model
        print("\nInitializing model...")
        self.model = MultiViewTripoSR(
            pretrained_model_path=config.triposr_model,
            freeze_encoder=config.freeze_image_encoder
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
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs_stage1
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_history = {'epoch': [], 'loss': [], 'lr': []}
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = len(self.dataloader)
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move to device
            images = {k: v.to(self.config.device) for k, v in batch['images'].items()}
            angles = batch['angles'].to(self.config.device)
            
            # Process each shoe in batch (Y values are lists)
            batch_loss = 0
            for i in range(len(batch['vertices'])):
                gt_vertices = batch['vertices'][i].to(self.config.device)
                gt_faces = batch['faces'][i].to(self.config.device)
                
                # Forward pass (single shoe)
                images_single = {k: v[i:i+1] for k, v in images.items()}
                angles_single = angles[i:i+1]
                
                triplane = self.model(images_single, angles_single)
                
                # Extract mesh
                pred_vertices, pred_faces = self.model.extract_mesh(
                    triplane,
                    resolution=self.config.mesh_resolution
                )
                
                # Compute loss
                loss, loss_dict = geometry_loss(pred_vertices, pred_faces, gt_vertices)
                batch_loss += loss
            
            # Average loss over batch
            loss = batch_loss / len(batch['vertices'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Epoch [{epoch+1}/{self.config.num_epochs_stage1}] "
                      f"Batch [{batch_idx+1}/{num_batches}] ({progress:.1f}%) "
                      f"Loss: {loss.item():.6f}")
        
        return epoch_loss / num_batches
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs_stage1):
            epoch_start = time.time()
            
            # Train one epoch
            avg_loss = self.train_epoch(epoch)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs_stage1} Summary:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"{'─'*70}\n")
            
            # Save history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['loss'].append(avg_loss)
            self.train_history['lr'].append(current_lr)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"geometry_epoch{epoch+1}.pth"
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
                best_path = Path(self.config.checkpoint_dir) / "triposr_mv_best.pth"
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
        print("STAGE 1 TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {hours}h {minutes}m")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Final model: {self.config.checkpoint_dir}/triposr_mv_best.pth")
        print("="*70)
        
        # Save training history
        history_path = Path(self.config.log_dir) / "geometry_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"\n✓ Training history saved: {history_path}")
        
        return self.model


def main():
    """Main training script"""
    print("\n" + "="*70)
    print("MULTI-VIEW TO 3D MESH - STAGE 1: GEOMETRY")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Initialize trainer
    trainer = GeometryTrainer(config)
    
    # Train
    model = trainer.train()
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Stage 1 Complete! Ready for Stage 2 (Texture Training)")


if __name__ == "__main__":
    main()