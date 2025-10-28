"""
Complete Training Script for Multi-View to 3D Mesh Generation
Two-Stage Training: Geometry → Texture
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import json

# Import your dataset
from load_data import ShoeDataset, custom_collate_fn

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
    
    # Paths
    obj_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/Completed"
    images_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/input_images"
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    
    # Training parameters
    batch_size = 2  # Start small for local training
    num_epochs_stage1 = 50  # Geometry training
    num_epochs_stage2 = 50  # Texture training
    learning_rate = 1e-4
    weight_decay = 0.01
    
    # Model parameters
    image_size = 256  # Reduce from 512 for faster training
    feature_dim = 512
    num_views = 6
    
    # Device
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval = 10  # Log every N batches
    save_interval = 5   # Save checkpoint every N epochs
    
    # Early stopping
    patience = 10
    min_delta = 1e-4

config = Config()


# ============================================================================
# Stage 1: Geometry Model (Simplified)
# ============================================================================

class MultiViewEncoder(nn.Module):
    """Encode 6 views into feature representation"""
    
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # Use pretrained ResNet as backbone
        import torchvision.models as models
        resnet = models.resnet34(pretrained=True)
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Project ResNet features
        self.proj = nn.Linear(512, feature_dim)
        
    def forward(self, image):
        """
        Args:
            image: (B, 3, H, W)
        Returns:
            features: (B, feature_dim)
        """
        feat = self.backbone(image)  # (B, 512, 1, 1)
        feat = feat.squeeze(-1).squeeze(-1)  # (B, 512)
        feat = self.proj(feat)  # (B, feature_dim)
        return feat


class MultiViewAggregator(nn.Module):
    """Aggregate features from 6 views"""
    
    def __init__(self, feature_dim=512, num_views=6):
        super().__init__()
        
        # Attention aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Learnable query
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
        
        # View angle encoding
        self.angle_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim)
        )
        
    def forward(self, view_features, view_angles):
        """
        Args:
            view_features: (B, num_views, feature_dim)
            view_angles: (B, num_views, 2)
        Returns:
            aggregated: (B, feature_dim)
        """
        B = view_features.shape[0]
        
        # Add positional encoding from view angles
        angle_emb = self.angle_encoder(view_angles)
        view_features = view_features + angle_emb
        
        # Attention aggregation
        query = self.query.expand(B, -1, -1)
        aggregated, _ = self.attention(query, view_features, view_features)
        
        return aggregated.squeeze(1)


class GeometryPredictor(nn.Module):
    """Predict point cloud from aggregated features"""
    
    def __init__(self, feature_dim=512, num_points=2048):
        super().__init__()
        self.num_points = num_points
        
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, num_points * 3)
        )
        
    def forward(self, features):
        """
        Args:
            features: (B, feature_dim)
        Returns:
            points: (B, num_points, 3)
        """
        B = features.shape[0]
        points = self.decoder(features)
        points = points.reshape(B, self.num_points, 3)
        return torch.tanh(points)  # Normalize to [-1, 1]


class GeometryModel(nn.Module):
    """Complete Stage 1 model: Images → Point Cloud"""
    
    def __init__(self, feature_dim=512, num_points=2048):
        super().__init__()
        
        self.encoder = MultiViewEncoder(feature_dim)
        self.aggregator = MultiViewAggregator(feature_dim)
        self.predictor = GeometryPredictor(feature_dim, num_points)
        
    def forward(self, images_dict, view_angles):
        """
        Args:
            images_dict: Dict with 6 views, each (B, 3, H, W)
            view_angles: (B, 6, 2)
        Returns:
            points: (B, num_points, 3)
        """
        B = images_dict['front'].shape[0]
        
        # Encode each view
        view_features = []
        for view in ['front', 'back', 'left', 'right', 'top', 'bottom']:
            feat = self.encoder(images_dict[view])
            view_features.append(feat)
        
        view_features = torch.stack(view_features, dim=1)  # (B, 6, feature_dim)
        
        # Aggregate views
        aggregated = self.aggregator(view_features, view_angles)
        
        # Predict geometry
        points = self.predictor(aggregated)
        
        return points


# ============================================================================
# Stage 2: Texture Model (Simplified)
# ============================================================================

class TextureModel(nn.Module):
    """Predict colors for each point"""
    
    def __init__(self, feature_dim=512, num_points=2048):
        super().__init__()
        
        self.encoder = MultiViewEncoder(feature_dim)
        
        # Point feature encoder
        self.point_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Color decoder
        self.color_decoder = nn.Sequential(
            nn.Linear(feature_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()  # RGB in [0, 1]
        )
        
    def forward(self, points, images_dict):
        """
        Args:
            points: (B, N, 3) - predicted geometry
            images_dict: Dict with 6 views
        Returns:
            colors: (B, N, 3)
        """
        B, N, _ = points.shape
        
        # Encode images (use front view for simplicity)
        img_feat = self.encoder(images_dict['front'])  # (B, feature_dim)
        
        # Encode point positions
        point_feat = self.point_encoder(points)  # (B, N, 128)
        
        # Broadcast image features to each point
        img_feat_exp = img_feat.unsqueeze(1).expand(-1, N, -1)  # (B, N, feature_dim)
        
        # Combine and predict colors
        combined = torch.cat([img_feat_exp, point_feat], dim=-1)
        colors = self.color_decoder(combined)
        
        return colors


# ============================================================================
# Loss Functions
# ============================================================================

def chamfer_distance(pred_points, gt_points):
    """
    Compute Chamfer Distance between two point clouds
    Simplified version - you may want to use pytorch3d for better performance
    """
    # pred_points: (B, N, 3)
    # gt_points: (M, 3) - single ground truth
    
    # Expand for broadcasting
    pred_exp = pred_points.unsqueeze(2)  # (B, N, 1, 3)
    gt_exp = gt_points.unsqueeze(0).unsqueeze(0)  # (1, 1, M, 3)
    
    # Compute distances
    dist = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)  # (B, N, M)
    
    # Forward: nearest neighbor from pred to gt
    dist_pred_to_gt = torch.min(dist, dim=2)[0]  # (B, N)
    forward_loss = torch.mean(dist_pred_to_gt)
    
    # Backward: nearest neighbor from gt to pred
    dist_gt_to_pred = torch.min(dist, dim=1)[0]  # (B, M)
    backward_loss = torch.mean(dist_gt_to_pred)
    
    return forward_loss + backward_loss


def color_loss(pred_colors, gt_colors):
    """L2 loss on colors"""
    return torch.mean((pred_colors - gt_colors) ** 2)


# ============================================================================
# Training Functions
# ============================================================================

class Trainer:
    """Training orchestrator"""
    
    def __init__(self, config):
        self.config = config
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(exist_ok=True)
        Path(config.log_dir).mkdir(exist_ok=True)
        
        # Initialize dataset
        print("Loading dataset...")
        self.dataset = ShoeDataset(
            obj_dir=config.obj_dir,
            images_dir=config.images_dir,
            verify_mappings=True
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        print(f"Dataset loaded: {len(self.dataset)} shoes")
        print(f"Device: {config.device}")
        
        # Training logs
        self.train_history = {
            'stage1': {'epoch': [], 'loss': []},
            'stage2': {'epoch': [], 'loss': []}
        }
        
    def train_stage1_geometry(self):
        """Train Stage 1: Geometry prediction"""
        print("\n" + "="*70)
        print("STAGE 1: TRAINING GEOMETRY MODEL")
        print("="*70)
        
        # Initialize model
        model = GeometryModel(
            feature_dim=self.config.feature_dim,
            num_points=2048
        ).to(self.config.device)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs_stage1
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs_stage1):
            model.train()
            epoch_loss = 0
            start_time = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                # Move to device
                images = {k: v.to(self.config.device) for k, v in batch['images'].items()}
                angles = batch['angles'].to(self.config.device)
                
                # Process each shoe in batch (since Y values are lists)
                batch_loss = 0
                for i in range(len(batch['vertices'])):
                    gt_vertices = batch['vertices'][i].to(self.config.device)
                    
                    # Sample points from GT mesh
                    num_gt_points = gt_vertices.shape[0]
                    if num_gt_points > 2048:
                        indices = torch.randperm(num_gt_points)[:2048]
                        gt_points = gt_vertices[indices]
                    else:
                        gt_points = gt_vertices
                    
                    # Forward pass (single shoe)
                    images_single = {k: v[i:i+1] for k, v in images.items()}
                    angles_single = angles[i:i+1]
                    
                    pred_points = model(images_single, angles_single)
                    
                    # Compute loss
                    loss = chamfer_distance(pred_points, gt_points)
                    batch_loss += loss
                
                # Average loss over batch
                loss = batch_loss / len(batch['vertices'])
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    print(f"Epoch [{epoch+1}/{self.config.num_epochs_stage1}] "
                          f"Batch [{batch_idx}/{len(self.dataloader)}] "
                          f"Loss: {loss.item():.6f}")
            
            # Epoch summary
            avg_loss = epoch_loss / len(self.dataloader)
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Time: {epoch_time:.2f}s")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save history
            self.train_history['stage1']['epoch'].append(epoch + 1)
            self.train_history['stage1']['loss'].append(avg_loss)
            
            # Learning rate scheduling
            scheduler.step()
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"stage1_epoch{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
            
            # Early stopping
            if avg_loss < best_loss - self.config.min_delta:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                best_path = Path(self.config.checkpoint_dir) / "stage1_best.pth"
                torch.save(model.state_dict(), best_path)
                print(f"  ✓ Best model saved: {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        print("\n" + "="*70)
        print("STAGE 1 COMPLETE")
        print("="*70)
        
        return model
    
    def train_stage2_texture(self, geometry_model):
        """Train Stage 2: Texture prediction"""
        print("\n" + "="*70)
        print("STAGE 2: TRAINING TEXTURE MODEL")
        print("="*70)
        
        # Freeze geometry model
        geometry_model.eval()
        for param in geometry_model.parameters():
            param.requires_grad = False
        
        # Initialize texture model
        texture_model = TextureModel(
            feature_dim=self.config.feature_dim,
            num_points=2048
        ).to(self.config.device)
        
        optimizer = optim.AdamW(
            texture_model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs_stage2
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs_stage2):
            texture_model.train()
            epoch_loss = 0
            start_time = time.time()
            
            for batch_idx, batch in enumerate(self.dataloader):
                images = {k: v.to(self.config.device) for k, v in batch['images'].items()}
                angles = batch['angles'].to(self.config.device)
                
                batch_loss = 0
                for i in range(len(batch['vertices'])):
                    gt_colors = batch['vertex_colors'][i].to(self.config.device)
                    
                    # Sample points
                    num_points = gt_colors.shape[0]
                    if num_points > 2048:
                        indices = torch.randperm(num_points)[:2048]
                        gt_colors_sampled = gt_colors[indices]
                    else:
                        gt_colors_sampled = gt_colors
                    
                    # Get geometry from frozen model
                    images_single = {k: v[i:i+1] for k, v in images.items()}
                    angles_single = angles[i:i+1]
                    
                    with torch.no_grad():
                        pred_points = geometry_model(images_single, angles_single)
                    
                    # Predict colors
                    pred_colors = texture_model(pred_points, images_single)
                    
                    # Compute loss
                    loss = color_loss(pred_colors, gt_colors_sampled.unsqueeze(0))
                    batch_loss += loss
                
                loss = batch_loss / len(batch['vertices'])
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(texture_model.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                
                if batch_idx % self.config.log_interval == 0:
                    print(f"Epoch [{epoch+1}/{self.config.num_epochs_stage2}] "
                          f"Batch [{batch_idx}/{len(self.dataloader)}] "
                          f"Loss: {loss.item():.6f}")
            
            avg_loss = epoch_loss / len(self.dataloader)
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_loss:.6f}")
            print(f"  Time: {epoch_time:.2f}s")
            
            self.train_history['stage2']['epoch'].append(epoch + 1)
            self.train_history['stage2']['loss'].append(avg_loss)
            
            scheduler.step()
            
            if (epoch + 1) % self.config.save_interval == 0:
                checkpoint_path = Path(self.config.checkpoint_dir) / f"stage2_epoch{epoch+1}.pth"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': texture_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")
            
            if avg_loss < best_loss - self.config.min_delta:
                best_loss = avg_loss
                patience_counter = 0
                best_path = Path(self.config.checkpoint_dir) / "stage2_best.pth"
                torch.save(texture_model.state_dict(), best_path)
                print(f"  ✓ Best model saved: {best_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        print("\n" + "="*70)
        print("STAGE 2 COMPLETE")
        print("="*70)
        
        return texture_model
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = Path(self.config.log_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"\n✓ Training history saved: {history_path}")


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    print("="*70)
    print("MULTI-VIEW TO 3D MESH TRAINING")
    print("="*70)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Image Size: {config.image_size}")
    print("="*70)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Stage 1: Train geometry model
    geometry_model = trainer.train_stage1_geometry()
    
    # Stage 2: Train texture model
    texture_model = trainer.train_stage2_texture(geometry_model)
    
    # Save training history
    trainer.save_training_history()
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels saved in: {config.checkpoint_dir}")
    print(f"Logs saved in: {config.log_dir}")
    print("="*70)


if __name__ == "__main__":
    main()