"""
Stage 1: Geometry Model (IMPROVED VERSION)
Multi-View to 3D Point Cloud/Mesh Prediction

KEY IMPROVEMENTS:
1. Better mesh extraction (Poisson reconstruction, not convex hull)
2. Advanced loss functions (edge length, normal consistency, Laplacian)
3. Higher point count support
4. Better point cloud decoder
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
        outputs = self.encoder(pixel_values=images)
        features = outputs.last_hidden_state[:, 0]  # (B, 768)
        return features


class ViewAggregator(nn.Module):
    """
    Aggregate features from 6 views using symmetric pooling operations.
    This approach is PERMUTATION INVARIANT - output is identical regardless of view order.
    """

    def __init__(self, feature_dim=768):
        super().__init__()
        self.feature_dim = feature_dim

        # MLP to process aggregated features (max + mean concatenated)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, view_features):
        """
        Permutation-invariant view aggregation.

        Args:
            view_features: (B, num_views, feature_dim) - features from each view

        Returns:
            aggregated: (B, feature_dim) - aggregated features
        """
        # Symmetric aggregation operations
        max_pool = torch.max(view_features, dim=1)[0]    # (B, feature_dim)
        mean_pool = torch.mean(view_features, dim=1)     # (B, feature_dim)

        # Concatenate and process
        aggregated = torch.cat([max_pool, mean_pool], dim=1)  # (B, 2*feature_dim)
        output = self.mlp(aggregated)  # (B, feature_dim)

        return output


class ImprovedPointCloudDecoder(nn.Module):
    """
    IMPROVED Point Cloud Decoder with:
    - Deeper network for better detail
    - Residual connections
    - Better normalization
    """
    
    def __init__(self, feature_dim=768, num_points=8192, hidden_dim=1024):
        super().__init__()
        
        self.num_points = num_points
        
        # Initial projection
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual blocks for better gradient flow
        self.res_block1 = self._make_res_block(hidden_dim, hidden_dim * 2)
        self.res_block2 = self._make_res_block(hidden_dim * 2, hidden_dim * 2)
        self.res_block3 = self._make_res_block(hidden_dim * 2, hidden_dim * 2)
        
        # Final decoder to points
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_points * 3)
        )
    
    def _make_res_block(self, in_dim, out_dim):
        """Create a residual block"""
        if in_dim != out_dim:
            skip = nn.Linear(in_dim, out_dim)
        else:
            skip = nn.Identity()
        
        return nn.ModuleDict({
            'skip': skip,
            'block': nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(out_dim, out_dim),
                nn.LayerNorm(out_dim)
            )
        })
    
    def _apply_res_block(self, x, res_block):
        """Apply residual block"""
        skip = res_block['skip'](x)
        out = res_block['block'](x)
        return torch.relu(skip + out)
    
    def forward(self, features):
        """
        Args:
            features: (B, feature_dim)
        Returns:
            points: (B, num_points, 3) - 3D coordinates
        """
        B = features.shape[0]
        
        # Project and process through residual blocks
        x = self.proj(features)
        x = self._apply_res_block(x, self.res_block1)
        x = self._apply_res_block(x, self.res_block2)
        x = self._apply_res_block(x, self.res_block3)
        
        # Decode to flat point coordinates
        points_flat = self.decoder(x)  # (B, num_points * 3)
        
        # Reshape to point cloud
        points = points_flat.view(B, self.num_points, 3)
        
        # Normalize to [-1, 1] range
        points = torch.tanh(points)
        
        return points


class GeometryModel(nn.Module):
    """
    IMPROVED Geometry Model: 6 Images → 3D Point Cloud
    
    Architecture:
        6 Images → DINOv2 Encoder → View Aggregation → Point Cloud Decoder
    """
    
    def __init__(self, 
                 num_points=8192,  # Increased default
                 freeze_encoder=False,
                 hidden_dim=1024):
        super().__init__()
        
        print("="*70)
        print("INITIALIZING IMPROVED GEOMETRY MODEL")
        print("="*70)
        
        # Image encoder (DINOv2)
        self.image_encoder = MultiViewImageEncoder(
            model_name='facebook/dinov2-base',
            freeze=freeze_encoder
        )
        feature_dim = self.image_encoder.feature_dim
        
        # View aggregator
        self.view_aggregator = ViewAggregator(feature_dim=feature_dim)
        
        # IMPROVED Point cloud decoder
        self.point_decoder = ImprovedPointCloudDecoder(
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
    
    def forward(self, images_dict):
        """
        Forward pass: 6 images → 3D point cloud

        Args:
            images_dict: Dict with keys ['front', 'back', 'left', 'right', 'top', 'bottom']
                         Each value is (B, 3, H, W)

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
        aggregated = self.view_aggregator(view_features)  # (B, feature_dim)

        # Decode to point cloud
        points = self.point_decoder(aggregated)  # (B, num_points, 3)

        return points
    
    def extract_mesh(self, points, method='poisson'):
        """
        IMPROVED: Convert point cloud to mesh using proper surface reconstruction
        
        Args:
            points: (N, 3) numpy array or (B, N, 3) tensor
            method: 'poisson' (recommended), 'ball_pivot', or 'alpha_shape'
        
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
        
        try:
            if method == 'poisson':
                # PROPER Poisson reconstruction (requires Open3D)
                try:
                    import open3d as o3d
                    
                    # Create point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    # Estimate normals (CRITICAL for good reconstruction)
                    pcd.estimate_normals(
                        search_param=o3d.geometry.KDTreeSearchParamHybrid(
                            radius=0.1, max_nn=30
                        )
                    )
                    
                    # Orient normals consistently
                    pcd.orient_normals_consistent_tangent_plane(30)
                    
                    # Poisson reconstruction with higher depth for more detail
                    mesh_o3d, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                        pcd, depth=10, width=0, scale=1.1, linear_fit=False
                    )
                    
                    # Remove low density vertices (noise)
                    vertices_to_remove = densities < np.quantile(densities, 0.1)
                    mesh_o3d.remove_vertices_by_mask(vertices_to_remove)
                    
                    # Clean up mesh
                    mesh_o3d.remove_degenerate_triangles()
                    mesh_o3d.remove_duplicated_triangles()
                    mesh_o3d.remove_duplicated_vertices()
                    mesh_o3d.remove_non_manifold_edges()
                    
                    vertices = np.asarray(mesh_o3d.vertices)
                    faces = np.asarray(mesh_o3d.triangles)
                    
                    print(f"  ✓ Poisson reconstruction: {len(vertices)} vertices, {len(faces)} faces")
                    
                    return vertices, faces
                    
                except ImportError:
                    print("  ⚠️ Open3D not available, install with: pip install open3d")
                    print("  Falling back to alpha shape...")
                    method = 'alpha_shape'
            
            if method == 'ball_pivot':
                # Ball pivoting algorithm
                try:
                    import open3d as o3d
                    
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    pcd.estimate_normals()
                    
                    # Compute average distance between points
                    distances = pcd.compute_nearest_neighbor_distance()
                    avg_dist = np.mean(distances)
                    radius = 1.5 * avg_dist
                    
                    # Ball pivoting
                    radii = [radius, radius * 2, radius * 4]
                    mesh_o3d = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                        pcd, o3d.utility.DoubleVector(radii)
                    )
                    
                    vertices = np.asarray(mesh_o3d.vertices)
                    faces = np.asarray(mesh_o3d.triangles)
                    
                    print(f"  ✓ Ball pivoting: {len(vertices)} vertices, {len(faces)} faces")
                    
                    return vertices, faces
                    
                except ImportError:
                    print("  ⚠️ Open3D not available")
                    method = 'alpha_shape'
            
            if method == 'alpha_shape':
                # Alpha shape (better than convex hull, but still limited)
                cloud = trimesh.PointCloud(points)
                
                try:
                    # Try to create alpha shape
                    mesh = trimesh.creation.alpha_shape(cloud, alpha=0.1)
                    print(f"  ✓ Alpha shape: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
                except:
                    print("  ⚠️ Alpha shape failed, using convex hull as last resort")
                    mesh = cloud.convex_hull
                
                vertices = mesh.vertices
                faces = mesh.faces
                
                return vertices, faces
            
        except Exception as e:
            print(f"  ⚠️ Mesh extraction failed ({e}), returning point cloud")
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
# IMPROVED Loss Functions
# ============================================================================

def chamfer_distance_simple(pred_points, gt_points):
    """
    Simplified Chamfer Distance (no external dependencies)
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


def edge_length_loss(points, k=10):
    """
    NEW: Regularize edge lengths between nearest neighbors
    Prevents points from collapsing or spreading too much
    
    Args:
        points: (N, 3) point cloud
        k: number of nearest neighbors to consider
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(points, points)  # (N, N)
    
    # Get k nearest neighbors for each point
    knn_dist, _ = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)  # (N, k+1)
    knn_dist = knn_dist[:, 1:]  # Exclude self (distance 0)
    
    # Target edge length (adjust based on your scale)
    target_length = 0.05
    
    # Penalize deviation from target length
    loss = torch.mean((knn_dist - target_length) ** 2)
    
    return loss


def normal_consistency_loss(points, k=20):
    """
    NEW: Encourage consistent surface normals
    Helps create smooth surfaces with proper orientation
    
    Args:
        points: (N, 3) point cloud
        k: number of neighbors for normal estimation
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(points, points)  # (N, N)
    
    # Get k nearest neighbors
    _, knn_idx = torch.topk(dist_matrix, k=k, dim=1, largest=False)  # (N, k)
    
    # For each point, estimate normal from neighbors
    normals = []
    for i in range(points.shape[0]):
        neighbors = points[knn_idx[i]]  # (k, 3)
        
        # Center the neighbors
        centered = neighbors - neighbors.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov = torch.matmul(centered.T, centered) / k
        
        # Get smallest eigenvector (normal direction)
        try:
            _, _, v = torch.svd(cov)
            normal = v[:, -1]  # Smallest eigenvector
            normals.append(normal)
        except:
            # Fallback if SVD fails
            normals.append(torch.tensor([0.0, 1.0, 0.0], device=points.device))
    
    normals = torch.stack(normals)  # (N, 3)
    
    # Normalize
    normals = normals / (torch.norm(normals, dim=1, keepdim=True) + 1e-8)
    
    # Consistency loss: normals of neighbors should be similar
    normal_diff = 0.0
    for i in range(points.shape[0]):
        neighbor_normals = normals[knn_idx[i]]  # (k, 3)
        # Cosine similarity (want high similarity = low loss)
        similarity = torch.matmul(neighbor_normals, normals[i])  # (k,)
        normal_diff += torch.mean(1.0 - torch.abs(similarity))
    
    return normal_diff / points.shape[0]


def laplacian_smoothness_loss(points, k=10):
    """
    NEW: Laplacian smoothness regularization
    Encourages smooth surfaces by penalizing high curvature
    
    Args:
        points: (N, 3) point cloud
        k: number of neighbors
    """
    # Compute pairwise distances
    dist_matrix = torch.cdist(points, points)  # (N, N)
    
    # Get k nearest neighbors
    _, knn_idx = torch.topk(dist_matrix, k=k+1, dim=1, largest=False)  # (N, k+1)
    knn_idx = knn_idx[:, 1:]  # Exclude self
    
    # Laplacian: each point should be close to average of neighbors
    laplacian_loss = 0.0
    for i in range(points.shape[0]):
        neighbors = points[knn_idx[i]]  # (k, 3)
        neighbor_mean = neighbors.mean(dim=0)
        laplacian_loss += torch.sum((points[i] - neighbor_mean) ** 2)
    
    return laplacian_loss / points.shape[0]


def coverage_loss(pred_points, gt_points, threshold=0.01):
    """
    Penalize if predicted points don't cover ground truth surface
    """
    pred_exp = pred_points.unsqueeze(1)  # (N, 1, 3)
    gt_exp = gt_points.unsqueeze(0)  # (1, M, 3)
    
    dist = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)  # (N, M)
    min_dist_to_pred = torch.min(dist, dim=0)[0]  # (M,)
    
    # Percentage of GT points covered
    covered = (min_dist_to_pred < threshold).float().mean()
    
    return 1.0 - covered


def geometry_loss(pred_points, pred_normals=None, gt_points=None, 
                  lambda_chamfer=1.0, 
                  lambda_normal=0.1,
                  lambda_edge=0.05,
                  lambda_smooth=0.02,
                  lambda_coverage=0.1):
    """
    IMPROVED Combined geometry loss with multiple regularization terms
    BACKWARD COMPATIBLE with old API
    
    Args:
        pred_points: (N, 3) predicted point cloud
        pred_normals: (N, 3) predicted surface normals OR gt_points for backward compatibility
        gt_points: (M, 3) ground truth vertices (optional if using old API)
        lambda_*: weights for each loss component
    
    Returns:
        total_loss: Weighted sum
        loss_dict: Individual losses
    
    Usage:
        # New API (recommended):
        loss = geometry_loss(pred_points, pred_normals, gt_points)
        
        # Old API (backward compatible):
        loss = geometry_loss(pred_points, gt_points)
    """
    # Backward compatibility: handle old API geometry_loss(pred_points, gt_points)
    if gt_points is None and pred_normals is not None:
        # Old API: geometry_loss(pred_points, gt_points)
        # Second argument is actually gt_points, not normals
        gt_points = pred_normals
        pred_normals = None
    
    if gt_points is None:
        raise ValueError("gt_points is required. Use: geometry_loss(pred_points, pred_normals, gt_points) or geometry_loss(pred_points, gt_points)")
    
    # Main Chamfer distance
    loss_chamfer = chamfer_distance_simple(pred_points, gt_points)
    
    # Coverage loss
    loss_cover = coverage_loss(pred_points, gt_points)
    
    # NEW: Edge length regularization
    loss_edge = edge_length_loss(pred_points, k=10)
    
    # NEW: Normal consistency (uses predicted normals if available, else estimates)
    if pred_normals is not None:
        # Use the predicted normals for consistency loss
        loss_normal = normal_consistency_with_predicted_normals(pred_points, pred_normals, k=10)
    else:
        # Fallback: estimate normals from point cloud
        loss_normal = normal_consistency_loss(pred_points, k=20)
    
    # NEW: Laplacian smoothness
    loss_smooth = laplacian_smoothness_loss(pred_points, k=10)
    
    # Total loss
    total_loss = (
        lambda_chamfer * loss_chamfer +
        lambda_coverage * loss_cover +
        lambda_edge * loss_edge +
        lambda_normal * loss_normal +
        lambda_smooth * loss_smooth
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
    """
    Normal consistency loss using predicted normals
    Ensures neighboring points have similar normal directions
    
    Args:
        points: (N, 3) point cloud
        normals: (N, 3) predicted normals (already normalized)
        k: number of nearest neighbors
    """
    N = points.shape[0]
    
    # Compute pairwise distances
    pred_exp = points.unsqueeze(1)
    dist = torch.sum((pred_exp - points.unsqueeze(0)) ** 2, dim=-1)
    
    # Find k nearest neighbors for each point
    _, indices = torch.topk(dist, k=k+1, largest=False, dim=1)
    indices = indices[:, 1:]  # Exclude self
    
    # Get normals of neighbors
    neighbor_normals = normals[indices]  # (N, k, 3)
    
    # Compute consistency (dot product should be close to 1)
    normals_exp = normals.unsqueeze(1).expand(-1, k, -1)
    consistency = torch.sum(normals_exp * neighbor_normals, dim=-1)  # (N, k)
    
    # Loss: want high consistency (minimize 1 - consistency)
    loss = torch.mean(1.0 - consistency)
    
    return loss