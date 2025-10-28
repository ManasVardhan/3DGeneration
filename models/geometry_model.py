"""
Stage 1: Geometry Model (TripoSR-Free Version)
Multi-View to 3D Point Cloud/Mesh Prediction

Uses only standard packages:
- PyTorch
- Transformers (DINOv2)
- Trimesh
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoImageProcessor


class MultiViewImageEncoder(nn.Module):
    """
    Encode images using DINOv2 (pretrained vision transformer)
    """
    
    def __init__(self, model_name='facebook/dinov2-base', freeze=False):
        super().__init__()
        
        print(f"Loading {model_name}...")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.encoder.config.hidden_size  # 768 for base
        print(f"✓ DINOv2 loaded (feature dim: {self.feature_dim})")
        
        if freeze:
            print("  Freezing encoder weights...")
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        """
        Args:
            images: (B, 3, H, W) RGB images
        Returns:
            features: (B, feature_dim) image features
        """
        # DINOv2 forward pass
        outputs = self.encoder(pixel_values=images)
        
        # Get CLS token (global image representation)
        features = outputs.last_hidden_state[:, 0]  # (B, 768)
        
        return features


class ViewAggregator(nn.Module):
    """
    Aggregate features from 6 views using attention
    """
    
    def __init__(self, feature_dim=768):
        super().__init__()
        
        # View positional encoding
        self.view_pos_encoder = nn.Sequential(
            nn.Linear(2, 128),  # (azimuth, elevation)
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )
        
        # Multi-head attention for aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=8,
            batch_first=True
        )
        
        # Learnable query for global features
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim))
    
    def forward(self, view_features, view_angles):
        """
        Args:
            view_features: (B, num_views, feature_dim)
            view_angles: (B, num_views, 2) - (azimuth, elevation) in degrees
        Returns:
            aggregated: (B, feature_dim)
        """
        B, num_views, _ = view_features.shape
        
        # Add positional encoding based on view angles
        pos_encoding = self.view_pos_encoder(view_angles)  # (B, num_views, feature_dim)
        view_features = view_features + pos_encoding
        
        # Aggregate using attention
        query = self.query.expand(B, -1, -1)  # (B, 1, feature_dim)
        aggregated, _ = self.attention(query, view_features, view_features)
        
        return aggregated.squeeze(1)  # (B, feature_dim)


class PointCloudDecoder(nn.Module):
    """
    Decode aggregated features into 3D point cloud
    """
    
    def __init__(self, feature_dim=768, num_points=4096, hidden_dim=1024):
        super().__init__()
        
        self.num_points = num_points
        
        # MLP decoder
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            
            nn.Linear(hidden_dim * 2, num_points * 3)
        )
    
    def forward(self, features):
        """
        Args:
            features: (B, feature_dim)
        Returns:
            points: (B, num_points, 3) - 3D coordinates
        """
        B = features.shape[0]
        
        # Decode to flat point coordinates
        points_flat = self.decoder(features)  # (B, num_points * 3)
        
        # Reshape to point cloud
        points = points_flat.view(B, self.num_points, 3)
        
        # Normalize to [-1, 1] range
        points = torch.tanh(points)
        
        return points


class GeometryModel(nn.Module):
    """
    Complete Geometry Model: 6 Images → 3D Point Cloud
    
    Architecture:
        6 Images → DINOv2 Encoder → View Aggregation → Point Cloud Decoder
    """
    
    def __init__(self, 
                 num_points=4096,
                 freeze_encoder=False,
                 hidden_dim=1024):
        super().__init__()
        
        print("="*70)
        print("INITIALIZING GEOMETRY MODEL")
        print("="*70)
        
        # Image encoder (DINOv2)
        self.image_encoder = MultiViewImageEncoder(
            model_name='facebook/dinov2-base',
            freeze=freeze_encoder
        )
        feature_dim = self.image_encoder.feature_dim
        
        # View aggregator
        self.view_aggregator = ViewAggregator(feature_dim=feature_dim)
        
        # Point cloud decoder
        self.point_decoder = PointCloudDecoder(
            feature_dim=feature_dim,
            num_points=num_points,
            hidden_dim=hidden_dim
        )
        
        self.num_points = num_points
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"✓ Model initialized")
        print(f"  Output: {num_points} 3D points")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print("="*70)
    
    def forward(self, images_dict, view_angles):
        """
        Forward pass: 6 images → 3D point cloud
        
        Args:
            images_dict: Dict with keys ['front', 'back', 'left', 'right', 'top', 'bottom']
                         Each value is (B, 3, H, W)
            view_angles: (B, 6, 2) - (azimuth, elevation) for each view
        
        Returns:
            points: (B, num_points, 3) - 3D point cloud
        """
        B = images_dict['front'].shape[0]
        
        # Encode each view
        view_features = []
        for view_name in ['front', 'back', 'left', 'right', 'top', 'bottom']:
            img = images_dict[view_name]  # (B, 3, H, W)
            feat = self.image_encoder(img)  # (B, feature_dim)
            view_features.append(feat)
        
        # Stack: (B, 6, feature_dim)
        view_features = torch.stack(view_features, dim=1)
        
        # Aggregate views
        aggregated = self.view_aggregator(view_features, view_angles)  # (B, feature_dim)
        
        # Decode to point cloud
        points = self.point_decoder(aggregated)  # (B, num_points, 3)
        
        return points
    
    def extract_mesh(self, points, method='alpha_shape'):
        """
        Convert point cloud to mesh using surface reconstruction
        
        Args:
            points: (N, 3) numpy array or (B, N, 3) tensor
            method: 'alpha_shape', 'poisson', or 'ball_pivot'
        
        Returns:
            vertices: (M, 3) numpy array
            faces: (F, 3) numpy array
        """
        import trimesh
        
        # Convert to numpy if tensor
        if torch.is_tensor(points):
            if points.dim() == 3:
                points = points[0]  # Take first batch
            points = points.detach().cpu().numpy()
        
        # Create point cloud
        cloud = trimesh.PointCloud(points)
        
        try:
            if method == 'alpha_shape':
                # Alpha shape (fast, works well for dense clouds)
                mesh = cloud.convex_hull
                
            elif method == 'poisson':
                # Poisson reconstruction (requires Open3D)
                try:
                    import open3d as o3d
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.estimate_normals()
                    mesh_o3d, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
                    
                    vertices = np.asarray(mesh_o3d.vertices)
                    faces = np.asarray(mesh_o3d.triangles)
                    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                except ImportError:
                    print("  Warning: Open3D not available, using convex hull")
                    mesh = cloud.convex_hull
            
            else:
                # Default to convex hull
                mesh = cloud.convex_hull
            
            vertices = mesh.vertices
            faces = mesh.faces
            
            return vertices, faces
            
        except Exception as e:
            print(f"  Warning: Mesh extraction failed ({e}), returning point cloud")
            # Return points as "mesh" with no faces
            return points, np.array([])
    
    def get_trainable_parameters(self):
        """Get trainable parameters for optimizer"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        """Count parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================================
# Loss Functions
# ============================================================================

def chamfer_distance_simple(pred_points, gt_points):
    """
    Simplified Chamfer Distance (no external dependencies)
    
    Args:
        pred_points: (N, 3) predicted points
        gt_points: (M, 3) ground truth points
    
    Returns:
        loss: scalar
    """
    # Expand for broadcasting
    pred_exp = pred_points.unsqueeze(1)  # (N, 1, 3)
    gt_exp = gt_points.unsqueeze(0)  # (1, M, 3)
    
    # Compute all pairwise distances
    dist = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)  # (N, M)
    
    # Forward: nearest neighbor from pred to gt
    min_dist_pred = torch.min(dist, dim=1)[0]  # (N,)
    forward_loss = torch.mean(min_dist_pred)
    
    # Backward: nearest neighbor from gt to pred
    min_dist_gt = torch.min(dist, dim=0)[0]  # (M,)
    backward_loss = torch.mean(min_dist_gt)
    
    return forward_loss + backward_loss


def chamfer_distance_pytorch3d(pred_points, gt_points):
    """
    Chamfer distance using PyTorch3D (if available)
    Falls back to simple version if not installed
    """
    try:
        from pytorch3d.loss import chamfer_distance
        loss, _ = chamfer_distance(
            pred_points.unsqueeze(0),
            gt_points.unsqueeze(0)
        )
        return loss
    except ImportError:
        return chamfer_distance_simple(pred_points, gt_points)


def earth_mover_distance(pred_points, gt_points):
    """
    Approximation of Earth Mover's Distance
    More expensive but better quality than Chamfer
    """
    # Simplified EMD using Hungarian algorithm approximation
    # This is just Chamfer for now, can be improved
    return chamfer_distance_simple(pred_points, gt_points)


def coverage_loss(pred_points, gt_points, threshold=0.01):
    """
    Penalize if predicted points don't cover ground truth surface
    """
    pred_exp = pred_points.unsqueeze(1)  # (N, 1, 3)
    gt_exp = gt_points.unsqueeze(0)  # (1, M, 3)
    
    dist = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)  # (N, M)
    min_dist_to_pred = torch.min(dist, dim=0)[0]  # (M,)
    
    # Percentage of GT points covered (within threshold)
    covered = (min_dist_to_pred < threshold).float().mean()
    
    # Loss: want high coverage (minimize 1 - covered)
    return 1.0 - covered


def geometry_loss(pred_points, gt_points, lambda_chamfer=1.0, lambda_coverage=0.1):
    """
    Combined geometry loss
    
    Args:
        pred_points: (N, 3) predicted point cloud
        gt_points: (M, 3) ground truth vertices
        lambda_chamfer: Weight for Chamfer distance
        lambda_coverage: Weight for coverage loss
    
    Returns:
        total_loss: Weighted sum
        loss_dict: Individual losses
    """
    # Main Chamfer distance
    loss_chamfer = chamfer_distance_pytorch3d(pred_points, gt_points)
    
    # Coverage loss (ensure we cover the GT surface)
    loss_cover = coverage_loss(pred_points, gt_points)
    
    # Total loss
    total_loss = lambda_chamfer * loss_chamfer + lambda_coverage * loss_cover
    
    loss_dict = {
        'chamfer': loss_chamfer.item(),
        'coverage': loss_cover.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict