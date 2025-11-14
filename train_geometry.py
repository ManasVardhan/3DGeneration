"""
FIXED Training Script with Multi-Scale Supervision
Key changes:
1. Uses multi-scale model
2. Progressive training strategy
3. Much stronger regularization
4. Proper mesh-based losses
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import time
from datetime import datetime
import json

from models.geometry_model import ImprovedGeometryModel, multi_scale_geometry_loss


class FixedConfig:
    """Configuration with proper settings for mesh training"""
    
    # Data paths - UPDATE THESE
    obj_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/OBJs"
    images_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/input_images"
    
    checkpoint_dir = "checkpoints_fixed"
    log_dir = "logs_fixed"
    output_dir = "output_fixed"
    
    # Training
    batch_size = 1
    num_epochs = 150  # More epochs needed
    
    # CRITICAL: Much lower learning rate
    learning_rate = 1e-5  # 10x lower!
    weight_decay = 0.01
    
    # Loss weights - MUCH STRONGER REGULARIZATION
    lambda_chamfer = 1.0
    lambda_edge = 2.0      # 4x stronger
    lambda_smooth = 1.0    # 5x stronger  
    lambda_normal = 0.5
    
    # Settings
    freeze_image_encoder = False
    hidden_dim = 1024
    image_size = 512
    num_workers = 0
    
    # Device
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval = 5
    save_interval = 10
    patience = 25
    min_delta = 1e-5
    
    grad_clip_norm = 0.5  # Lower gradient clipping
    seed = 42


config = FixedConfig()
torch.manual_seed(config.seed)


class MultiScaleTrainer:
    """Trainer for multi-scale geometry model"""

    def __init__(self, config):
        self.config = config
        
        Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(config.log_dir).mkdir(exist_ok=True, parents=True)

        print("="*70)
        print("MULTI-SCALE MESH TRAINING")
        print("="*70)
        print(f"Device: {config.device}")
        print(f"Learning Rate: {config.learning_rate} (VERY LOW - prevents fragmentation)")
        print(f"Regularization:")
        print(f"  Edge weight: {config.lambda_edge} (STRONG)")
        print(f"  Smooth weight: {config.lambda_smooth} (STRONG)")
        print("="*70)

        # Load dataset
        from load_data import ShoeDataset, custom_collate_fn
        
        print("\nLoading dataset...")
        self.dataset = ShoeDataset(
            obj_dir=config.obj_dir,
            images_dir=config.images_dir,
            verify_mappings=True,
            image_size=config.image_size
        )

        if len(self.dataset) == 0:
            raise ValueError(f"No data found!")

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=custom_collate_fn
        )

        print(f"✓ Dataset: {len(self.dataset)} shoes")

        # Initialize multi-scale model
        print("\nInitializing multi-scale model...")
        self.model = ImprovedGeometryModel(
            freeze_encoder=config.freeze_image_encoder,
            hidden_dim=config.hidden_dim
        ).to(config.device)

        total_params, trainable_params = self.model.count_parameters()
        print(f"✓ Parameters: {total_params:,} total, {trainable_params:,} trainable")

        # Optimizer with very low LR
        self.optimizer = optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warmup
        self.warmup_epochs = 10
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs - self.warmup_epochs,
            eta_min=1e-7
        )

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.train_history = {
            'epoch': [],
            'loss': [],
            'chamfer_fine': [],
            'edge_fine': [],
            'smooth_fine': [],
            'lr': []
        }

        print(f"\n✓ Trainer ready")
        print(f"  Warmup: {self.warmup_epochs} epochs")
        print(f"  Grad clip: {config.grad_clip_norm}")

    def _sample_points(self, vertices, num_samples=2562):
        """Sample points from vertices"""
        num_verts = vertices.shape[0]
        if num_verts >= num_samples:
            indices = torch.randperm(num_verts, device=vertices.device)[:num_samples]
            return vertices[indices]
        else:
            indices = torch.randint(0, num_verts, (num_samples,), device=vertices.device)
            return vertices[indices]

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        epoch_loss = 0
        epoch_losses = {
            'chamfer_coarse': 0, 'chamfer_mid': 0, 'chamfer_fine': 0,
            'edge_fine': 0, 'smooth_fine': 0, 'normal_fine': 0
        }
        num_batches = len(self.dataloader)

        # Warmup LR
        if epoch < self.warmup_epochs:
            lr_scale = (epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.learning_rate * lr_scale

        for batch_idx, batch in enumerate(self.dataloader):
            images = {k: v.to(self.config.device) for k, v in batch['images'].items()}

            batch_loss = 0
            batch_loss_dict = {k: 0 for k in epoch_losses.keys()}

            for i in range(len(batch['vertices'])):
                gt_vertices = batch['vertices'][i].to(self.config.device)
                images_single = {k: v[i:i+1] for k, v in images.items()}

                # Forward: get all levels
                mesh_outputs = self.model(images_single, return_all_levels=True)

                # Sample GT to match fine mesh size
                gt_sample = self._sample_points(gt_vertices, num_samples=2562)

                # Multi-scale loss
                loss, loss_dict = multi_scale_geometry_loss(
                    mesh_outputs,
                    gt_sample,
                    lambda_chamfer=self.config.lambda_chamfer,
                    lambda_edge=self.config.lambda_edge,
                    lambda_smooth=self.config.lambda_smooth,
                    lambda_normal=self.config.lambda_normal
                )

                batch_loss += loss
                for key in batch_loss_dict.keys():
                    if key in loss_dict:
                        batch_loss_dict[key] += loss_dict[key]

            # Average over batch
            loss = batch_loss / len(batch['vertices'])
            for key in batch_loss_dict.keys():
                batch_loss_dict[key] /= len(batch['vertices'])

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping - CRITICAL
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(), 
                self.config.grad_clip_norm
            )
            
            self.optimizer.step()

            epoch_loss += loss.item()
            for key in epoch_losses.keys():
                epoch_losses[key] += batch_loss_dict[key]

            # Logging
            if batch_idx % self.config.log_interval == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"  Epoch [{epoch+1}/{self.config.num_epochs}] "
                      f"Batch [{batch_idx+1}/{num_batches}] ({progress:.1f}%)")
                print(f"    Loss: {loss.item():.6f}")
                print(f"    Chamfer (C/M/F): {batch_loss_dict['chamfer_coarse']:.6f} / "
                      f"{batch_loss_dict['chamfer_mid']:.6f} / {batch_loss_dict['chamfer_fine']:.6f}")
                print(f"    Edge: {batch_loss_dict['edge_fine']:.6f} | "
                      f"Smooth: {batch_loss_dict['smooth_fine']:.6f}")

        avg_loss = epoch_loss / num_batches
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}

        return avg_loss, avg_losses

    def train(self):
        """Main training loop"""
        print("\n" + "="*70)
        print("STARTING MULTI-SCALE TRAINING")
        print("="*70)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            avg_loss, avg_losses = self.train_epoch(epoch)

            if epoch >= self.warmup_epochs:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"\n{'─'*70}")
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Total Loss: {avg_loss:.6f}")
            print(f"  Chamfer Fine: {avg_losses['chamfer_fine']:.6f}")
            print(f"  Edge Fine: {avg_losses['edge_fine']:.6f}")
            print(f"  Smooth Fine: {avg_losses['smooth_fine']:.6f}")
            print(f"  Time: {epoch_time:.1f}s | LR: {current_lr:.2e}")
            print(f"{'─'*70}\n")

            # Save history
            self.train_history['epoch'].append(epoch + 1)
            self.train_history['loss'].append(avg_loss)
            self.train_history['lr'].append(current_lr)
            self.train_history['chamfer_fine'].append(avg_losses['chamfer_fine'])
            self.train_history['edge_fine'].append(avg_losses['edge_fine'])
            self.train_history['smooth_fine'].append(avg_losses['smooth_fine'])

            # Checkpointing
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"multiscale_epoch{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'history': self.train_history
                }, checkpoint_path)
                print(f"✓ Checkpoint: {checkpoint_path}\n")

            # Early stopping
            if avg_loss < self.best_loss - self.config.min_delta:
                self.best_loss = avg_loss
                self.patience_counter = 0

                best_path = Path(self.config.checkpoint_dir) / "multiscale_best.pth"
                torch.save(self.model.state_dict(), best_path)
                print(f"✓ Best model: {best_path} (Loss: {avg_loss:.6f})\n")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        total_time = time.time() - start_time
        hours, minutes = int(total_time // 3600), int((total_time % 3600) // 60)

        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Time: {hours}h {minutes}m")
        print(f"Best loss: {self.best_loss:.6f}")
        print("="*70)

        # Save history
        history_path = Path(self.config.log_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"\n✓ History: {history_path}")

        return self.model


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("FIXED MULTI-SCALE MESH TRAINING")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verify paths
    if not Path(config.obj_dir).exists():
        print(f"✗ OBJ dir not found: {config.obj_dir}")
        return
    if not Path(config.images_dir).exists():
        print(f"✗ Images dir not found: {config.images_dir}")
        return
    
    print("\nConfiguration:")
    print(f"  LR: {config.learning_rate} (VERY LOW)")
    print(f"  Edge regularization: {config.lambda_edge} (STRONG)")
    print(f"  Smooth regularization: {config.lambda_smooth} (STRONG)")
    print(f"  Grad clip: {config.grad_clip_norm}")

    trainer = MultiScaleTrainer(config)
    model = trainer.train()

    print(f"\n✅ Training complete!")
    print("\nKey fixes applied:")
    print("  • Multi-scale progressive refinement (coarse→mid→fine)")
    print("  • Mesh-aware losses using actual connectivity")
    print("  • Much stronger regularization (2-5x)")
    print("  • Very low learning rate (1e-5)")
    print("  • Bounded deformations at each scale")


if __name__ == "__main__":
    main()