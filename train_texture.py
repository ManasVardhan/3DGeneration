"""
Train Stage 2: Texture Model
Train Vertex Color Prediction (Geometry frozen)

Usage:
    python train_texture.py
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
from models.geometry_model import MultiViewTripoSR
from models.texture_model import VertexColorPredictor, texture_loss, match_vertex_counts
from load_data import ShoeDataset, custom_collate_fn


class TextureTrainer:
    """Trainer for Stage 2: Texture prediction"""
    
    def __init__(self, config, geometry_checkpoint):
        self.config = config
        self.geometry_checkpoint = geometry_checkpoint
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.log_dir).mkdir(exist_ok=True)
        
        print("="*70)
        print("STAGE 2: TEXTURE MODEL TRAINING")
        print("="*70)
        print(f"Device: {config.device}")
        print(f"Batch Size: {config.batch_size_stage2}")
        print(f"Learning Rate: {config.learning_rate_stage2}")
        print(f"Geometry Model: {geometry_checkpoint}")
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
            batch_size=config.batch_size_stage2,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )
        
        print(f"✓ Dataset loaded: {len(self.dataset)} shoes")
        
        # Load frozen geometry model
        print("\nLoading geometry model...")
        self.geometry_model = MultiViewTripoSR(
            pretrained_model_path=config.triposr_model,
            freeze_encoder=True
        ).to(config.device)
        
        self.geometry_model.load_state_dict(
            torch.load(geometry_checkpoint, map_location=config.device)
        )
        self.geometry_model.eval()
        
        # Freeze all parameters
        for param in self.geometry_model.parameters():
            param.requires_grad = False
        
        print(f"✓ Geometry model loaded and frozen")
        
        # Initialize texture model
        print("\nInitializing texture model...")
        self.texture_model = VertexColorPredictor(
            feature_dim=config.feature_dim,
            use_all_views=True
        ).to(config.device)
        
        trainable_params = sum(p.numel() for p in self.texture_model.parameters() if p.requires_grad)
        print(f"✓ Texture model initialized")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.texture_model.parameters(),
            lr=config.learning_rate_stage2,
            weight_decay=config.weight_decay_stage2
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs_stage2
        )
        
        # Training state
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_history = {'epoch': [], 'loss': [], 'lr': []}
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.texture_model.train()
        epoch_loss = 0
        num_batches = len(self.dataloader)
        
        for batch_idx, batch in enumerate(self.dataloader):
            # Move to device
            images = {k: v.to(self.config.device) for k, v in batch['images'].items()}
            angles = batch['angles'].to(self.config.device)
            
            # Process each shoe in batch
            batch_loss = 0
            for i in range(len(batch['vertices'])):
                gt_colors = batch['vertex_colors'][i].to(self.config.device)
                gt_faces = batch['faces'][i].to(self.config.device)
                
                # Get geometry from frozen model
                images_single = {k: v[i:i+1] for k, v in images.items()}
                angles_single = angles[i:i+1]
                
                with torch.no_grad():
                    triplane = self.geometry_model(images_single, angles_single)
                    pred_vertices, pred_faces = self.geometry_model.extract_mesh(
                        triplane,
                        resolution=self.config.mesh_resolution
                    )
                
                # Predict colors
                pred_colors = self.texture_model(pred_vertices, images_single)
                
                # Match vertex counts
                pred_colors_matched, gt_colors_matched = match_vertex_counts(
                    pred_colors, gt_colors, pred_vertices, batch['vertices'][i]
                )
                
                # Compute loss
                loss, loss_dict = texture_loss(
                    pred_colors_matched,
                    gt_colors_matched,
                    faces=pred_faces
                )
                batch_loss += loss
            
            # Average loss over batch
            loss = batch_loss / len(batch['vertices'])
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.texture_model.parameters(), 1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Epoch [{epoch+1}/{self.config.num_epochs_stage2}] "
                      f"Batch [{batch_idx+1}/{num_batches}] ({progress:.1f}%) "
                      f"Loss: {loss.item():.6f}")
        
        return epoch_loss / num_batches
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs_stage2):
            epoch_start = time.time()
            
            # Train one epoch
            avg_loss = self.train_epoch(epoch)
            
            # Epoch summary
            epoch_time = time.time() - epoch_start
            current_lr = self.scheduler.get_last_lr()[0]
            
            print(f"\n{'─'*70}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs_stage2} Summary:")
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
                checkpoint_path = Path(self.config.checkpoint_dir) / f"texture_epoch{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.texture_model.state_dict(),
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
                best_path = Path(self.config.checkpoint_dir) / "texture_best.pth"
                torch.save(self.texture_model.state_dict(), best_path)
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
        print("STAGE 2 TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {hours}h {minutes}m")
        print(f"Best loss: {self.best_loss:.6f}")
        print(f"Final model: {self.config.checkpoint_dir}/texture_best.pth")
        print("="*70)
        
        # Save training history
        history_path = Path(self.config.log_dir) / "texture_training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"\n✓ Training history saved: {history_path}")
        
        return self.texture_model


def main():
    """Main training script"""
    print("\n" + "="*70)
    print("MULTI-VIEW TO 3D MESH - STAGE 2: TEXTURE")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if geometry model exists
    geometry_checkpoint = Path(config.checkpoint_dir) / "triposr_mv_best.pth"
    if not geometry_checkpoint.exists():
        print("\n❌ ERROR: Geometry model not found!")
        print(f"   Expected: {geometry_checkpoint}")
        print("\n   Please train Stage 1 first:")
        print("   python train_geometry.py")
        return
    
    # Initialize trainer
    trainer = TextureTrainer(config, str(geometry_checkpoint))
    
    # Train
    model = trainer.train()
    
    print(f"\nEnd Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✅ Stage 2 Complete! Models ready for inference")
    print("\nNext steps:")
    print("  python inference.py --shoe_id 17")


if __name__ == "__main__":
    main()