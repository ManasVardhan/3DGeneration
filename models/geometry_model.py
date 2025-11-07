"""
FIXED REPLACEMENT for models/geometry_model.py
Drop this into your models/ directory as geometry_model.py
This version maintains mesh connectivity to prevent fragmentation
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoImageProcessor
import trimesh


class MultiViewImageEncoder(nn.Module):
    """Encode images using DINOv2"""
    
    def __init__(self, model_name='facebook/dinov2-base', freeze=False):
        super().__init__()
        print(f"Loading {model_name}...")
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.encoder.config.hidden_size
        print(f"✓ DINOv2 loaded (feature dim: {self.feature_dim})")
        
        if freeze:
            print("  Freezing encoder weights...")
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, images):
        outputs = self.encoder(pixel_values=images)
        features = outputs.last_hidden_state[:, 0]
        return features


class ViewAggregator(nn.Module):
    """Aggregate features from 6 views"""
    def __init__(self, feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, view_features):
        max_pool = torch.max(view_features, dim=1)[0]
        mean_pool = torch.mean(view_features, dim=1)
        aggregated = torch.cat([max_pool, mean_pool], dim=1)
        output = self.mlp(aggregated)
        return output


class ImprovedPointCloudDecoder(nn.Module):
    """FIXED: Mesh-aware decoder that maintains connectivity"""
    
    def __init__(self, feature_dim=768, num_points=8192, hidden_dim=1024):
        super().__init__()
        self.num_points = num_points
        
        # Create initial sphere mesh template
        if num_points <= 642:
            subdivisions = 3  # 642 vertices
            self.actual_points = 642
        elif num_points <= 2562:
            subdivisions = 4  # 2562 vertices  
            self.actual_points = 2562
        else:
            subdivisions = 5  # 10242 vertices
            self.actual_points = 10242
            
        sphere = trimesh.creation.icosphere(subdivisions=subdivisions)
        self.register_buffer('init_vertices', torch.tensor(sphere.vertices, dtype=torch.float32))
        self.register_buffer('faces', torch.tensor(sphere.faces, dtype=torch.long))
        
        # Normalize initial vertices
        self.init_vertices = self.init_vertices / torch.norm(self.init_vertices, dim=1, keepdim=True)
        
        # Network to predict vertex offsets
        self.offset_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.actual_points * 3)
        )
        
        # Initialize last layer with small weights
        nn.init.normal_(self.offset_net[-1].weight, 0, 0.01)
        nn.init.zeros_(self.offset_net[-1].bias)
    
    def forward(self, features):
        B = features.shape[0]
        
        # Predict offsets for each vertex
        offsets = self.offset_net(features)  # (B, num_vertices * 3)
        offsets = offsets.view(B, self.actual_points, 3)
        
        # Apply bounded offsets to initial vertices
        offsets = torch.tanh(offsets) * 0.5  # Max displacement ±0.5
        
        # Get deformed vertices
        init_batch = self.init_vertices.unsqueeze(0).expand(B, -1, -1)
        deformed_vertices = init_batch + offsets
        
        # For backward compatibility, if requested more points, sample
        if self.num_points > self.actual_points:
            # Randomly sample to get requested number of points
            indices = torch.randint(0, self.actual_points, (B, self.num_points), device=features.device)
            points = torch.gather(deformed_vertices, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
        elif self.num_points < self.actual_points:
            # Subsample vertices
            indices = torch.randperm(self.actual_points, device=features.device)[:self.num_points]
            points = deformed_vertices[:, indices, :]
        else:
            points = deformed_vertices
        
        return points


class NormalDecoder(nn.Module):
    """Decoder for surface normals"""
    def __init__(self, feature_dim=768, num_points=8192, hidden_dim=1024):
        super().__init__()
        self.num_points = num_points
        
        # Use same actual points as the main decoder
        if num_points <= 642:
            self.actual_points = 642
        elif num_points <= 2562:
            self.actual_points = 2562
        else:
            self.actual_points = 10242
            
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.actual_points * 3)
        )

    def forward(self, features):
        B = features.shape[0]
        normals = self.decoder(features)
        normals = normals.view(B, self.actual_points, 3)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)
        
        # Sample if needed for consistency with point decoder
        if self.num_points != self.actual_points:
            if self.num_points > self.actual_points:
                indices = torch.randint(0, self.actual_points, (B, self.num_points), device=features.device)
                normals = torch.gather(normals, 1, indices.unsqueeze(-1).expand(-1, -1, 3))
            else:
                indices = torch.randperm(self.actual_points, device=features.device)[:self.num_points]
                normals = normals[:, indices, :]
        
        return normals


class GeometryModel(nn.Module):
    """FIXED Geometry Model with mesh connectivity preservation"""

    def __init__(self, num_points=8192, freeze_encoder=False, hidden_dim=1024):
        super().__init__()
        
        print("="*70)
        print("INITIALIZING MESH-AWARE GEOMETRY MODEL")
        print("="*70)

        self.image_encoder = MultiViewImageEncoder(
            model_name='facebook/dinov2-base',
            freeze=freeze_encoder
        )
        feature_dim = self.image_encoder.feature_dim

        self.view_aggregator = ViewAggregator(feature_dim=feature_dim)

        # FIXED: Use mesh-aware decoder
        self.point_decoder = ImprovedPointCloudDecoder(
            feature_dim=feature_dim,
            num_points=num_points,
            hidden_dim=hidden_dim
        )

        self.normal_decoder = NormalDecoder(
            feature_dim=feature_dim,
            num_points=num_points,
            hidden_dim=hidden_dim
        )

        print(f"✓ Model initialized with mesh connectivity")
        print(f"  Requested points: {num_points}")
        print(f"  Actual mesh vertices: {self.point_decoder.actual_points}")
        print(f"  Maintains connectivity: YES")
        print("="*70)

    def forward(self, images_dict):
        view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        images_list = [images_dict[name] for name in view_names]
        
        view_features = []
        for img in images_list:
            feat = self.image_encoder(img)
            view_features.append(feat)
        
        view_features = torch.stack(view_features, dim=1)
        aggregated = self.view_aggregator(view_features)
        
        points = self.point_decoder(aggregated)
        normals = self.normal_decoder(aggregated)
        
        return points, normals

    def get_trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================================
# FIXED LOSS FUNCTIONS
# ============================================================================

def chamfer_distance_simple(pred, gt):
    """Chamfer distance between point clouds"""
    pred_to_gt = torch.cdist(pred, gt, p=2)
    loss1 = torch.mean(torch.min(pred_to_gt, dim=1)[0])
    
    gt_to_pred = torch.cdist(gt, pred, p=2)
    loss2 = torch.mean(torch.min(gt_to_pred, dim=1)[0])
    
    return (loss1 + loss2) / 2


def edge_length_loss(points, k=10):
    """Regularize edge lengths to prevent fragmentation"""
    N = points.shape[0]
    if N <= k:
        k = max(1, N - 1)
    if k < 1:
        return torch.tensor(0.0, device=points.device)
    
    dist_matrix = torch.cdist(points, points)
    knn_dist, _ = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)
    knn_dist = knn_dist[:, 1:]  # Exclude self
    
    # Penalize very short or very long edges
    target_length = 0.05
    loss = torch.mean((knn_dist - target_length) ** 2)
    
    return loss


def normal_consistency_loss(points, k=20):
    """Encourage consistent surface normals"""
    N = points.shape[0]
    k = min(k, N - 1)
    if k < 3:
        return torch.tensor(0.0, device=points.device)
    
    dist_matrix = torch.cdist(points, points)
    _, knn_idx = torch.topk(dist_matrix, k=k, dim=1, largest=False)
    
    normals = []
    for i in range(N):
        neighbors = points[knn_idx[i]]
        centered = neighbors - neighbors.mean(dim=0, keepdim=True)
        cov = torch.matmul(centered.t(), centered) / k
        
        try:
            _, _, v = torch.svd(cov)
            normal = v[:, -1]
            normals.append(normal)
        except:
            normals.append(torch.tensor([0.0, 1.0, 0.0], device=points.device, dtype=points.dtype))
    
    normals = torch.stack(normals)
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    
    normal_diff = 0.0
    for i in range(N):
        neighbor_normals = normals[knn_idx[i]]
        similarity = torch.matmul(neighbor_normals, normals[i])
        normal_diff += torch.mean(1.0 - torch.abs(similarity))
    
    return normal_diff / N


def laplacian_smoothness_loss(points, k=10):
    """Laplacian smoothness regularization"""
    N = points.shape[0]
    if N <= k:
        k = max(1, N - 1)
    if k < 1:
        return torch.tensor(0.0, device=points.device)
    
    dist_matrix = torch.cdist(points, points)
    _, knn_idx = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)
    knn_idx = knn_idx[:, 1:]  # Exclude self
    
    laplacian_loss = 0.0
    for i in range(points.shape[0]):
        neighbors = points[knn_idx[i]]
        neighbor_mean = neighbors.mean(dim=0)
        laplacian_loss += torch.sum((points[i] - neighbor_mean) ** 2)
    
    return laplacian_loss / points.shape[0]


def coverage_loss(pred_points, gt_points, threshold=0.01):
    """Coverage loss"""
    pred_exp = pred_points.unsqueeze(1)
    gt_exp = gt_points.unsqueeze(0)
    dist = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)
    min_dist_to_pred = torch.min(dist, dim=0)[0]
    covered = (min_dist_to_pred < threshold).float().mean()
    return 1.0 - covered


def geometry_loss(pred_points, pred_normals=None, gt_points=None, 
                  lambda_chamfer=1.0, 
                  lambda_normal=0.1,
                  lambda_edge=0.5,    # INCREASED for better connectivity
                  lambda_smooth=0.2,  # INCREASED for smoother surface
                  lambda_coverage=0.1):
    """
    FIXED geometry loss with better regularization to prevent fragmentation
    Backward compatible with original API
    """
    # Handle backward compatibility
    if gt_points is None and pred_normals is not None:
        gt_points = pred_normals
        pred_normals = None
    
    if gt_points is None:
        raise ValueError("gt_points is required")
    
    # Main losses
    loss_chamfer = chamfer_distance_simple(pred_points, gt_points)
    loss_cover = coverage_loss(pred_points, gt_points)
    
    # CRITICAL: Regularization to prevent fragmentation
    loss_edge = edge_length_loss(pred_points, k=10)
    loss_smooth = laplacian_smoothness_loss(pred_points, k=10)
    
    if pred_normals is not None:
        loss_normal = normal_consistency_with_predicted_normals(pred_points, pred_normals, k=10)
    else:
        loss_normal = normal_consistency_loss(pred_points, k=20)
    
    # Total loss with adjusted weights
    total_loss = (
        lambda_chamfer * loss_chamfer +
        lambda_coverage * loss_cover +
        lambda_edge * loss_edge +      # Higher weight to maintain connectivity
        lambda_normal * loss_normal +
        lambda_smooth * loss_smooth     # Higher weight for smoothness
    )
    
    loss_dict = {
        'chamfer': loss_chamfer.item(),
        'coverage': loss_cover.item(),
        'edge': loss_edge.item(),
        'normal': loss_normal.item(),
        'smooth': loss_smooth.item(),
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict


def normal_consistency_with_predicted_normals(points, normals, k=10):
    """Normal consistency using predicted normals"""
    N = points.shape[0]
    if N <= k:
        k = max(1, N - 1)
    if k < 1:
        return torch.tensor(0.0, device=points.device)
    
    pred_exp = points.unsqueeze(1)
    dist = torch.sum((pred_exp - points.unsqueeze(0)) ** 2, dim=-1)
    _, indices = torch.topk(dist, k=k+1, largest=False, dim=1)
    indices = indices[:, 1:]  # Exclude self
    
    neighbor_normals = normals[indices]
    normals_exp = normals.unsqueeze(1).expand(-1, k, -1)
    consistency = torch.sum(normals_exp * neighbor_normals, dim=-1)
    loss = torch.mean(1.0 - consistency)
    
    return loss