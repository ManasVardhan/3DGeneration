"""
Stage 1: Geometry Model
Multi-View TripoSR for 3D Shape Prediction
"""

import torch
import torch.nn as nn

try:
    from TripoSR.tsr.system import TSR
except ImportError:
    print("ERROR: TripoSR not installed!")
    print("Install with: pip install git+https://github.com/VAST-AI-Research/TripoSR.git")
    exit(1)


class MultiViewTripoSR(nn.Module):
    """
    Adapt TripoSR to handle 6 views instead of 1
    
    Architecture:
        6 Images (512×512) 
        → DINOv2 Encoder (per view)
        → Multi-View Aggregation (attention)
        → TripoSR Transformer
        → Triplane Generator
        → Mesh Extraction (marching cubes)
    """
    
    def __init__(self, pretrained_model_path="stabilityai/TripoSR", freeze_encoder=False):
        super().__init__()
        
        print(f"Loading TripoSR from {pretrained_model_path}...")
        
        # Load pretrained TripoSR
        self.base_model = TSR.from_pretrained(
            pretrained_model_path,
            config_name="config.yaml",
            weight_name="model.ckpt"
        )
        print("✓ TripoSR loaded successfully")
        
        # Get TripoSR components
        self.image_tokenizer = self.base_model.image_tokenizer  # DINOv2 encoder
        self.tokenizer = self.base_model.tokenizer  # Transformer
        self.backbone = self.base_model.backbone  # Triplane decoder
        self.post_processor = self.base_model.post_processor  # Mesh extraction
        self.renderer = self.base_model.renderer
        
        # Freeze image encoder if requested
        if freeze_encoder:
            print("Freezing image encoder (DINOv2)...")
            for param in self.image_tokenizer.parameters():
                param.requires_grad = False
        
        # Multi-view aggregation layer
        self.view_aggregator = nn.MultiheadAttention(
            embed_dim=1024,  # TripoSR token dimension
            num_heads=8,
            batch_first=True
        )
        
        # Learnable query token for aggregation
        self.query_token = nn.Parameter(torch.randn(1, 1, 1024))
        
        # View angle positional encoding
        self.view_pos_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 1024)
        )
    
    def encode_views(self, images_dict, view_angles):
        """
        Encode 6 views with TripoSR's image encoder
        
        Args:
            images_dict: Dict with 6 views, each (B, 3, H, W)
            view_angles: (B, 6, 2) - azimuth, elevation in degrees
        
        Returns:
            tokens: (B, 6, num_tokens, 1024)
        """
        B = images_dict['front'].shape[0]
        view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        all_tokens = []
        
        for i, view_name in enumerate(view_names):
            # Encode this view with DINOv2
            img = images_dict[view_name]  # (B, 3, H, W)
            tokens = self.image_tokenizer(img)  # (B, num_tokens, 1024)
            
            # Add positional encoding for view angle
            angle = view_angles[:, i, :]  # (B, 2)
            pos_emb = self.view_pos_encoder(angle).unsqueeze(1)  # (B, 1, 1024)
            tokens = tokens + pos_emb  # Broadcast to all tokens
            
            all_tokens.append(tokens)
        
        # Stack all views: (B, 6, num_tokens, 1024)
        all_tokens = torch.stack(all_tokens, dim=1)
        
        return all_tokens
    
    def aggregate_views(self, view_tokens):
        """
        Aggregate 6 views into single representation using attention
        
        Args:
            view_tokens: (B, 6, num_tokens, 1024)
        
        Returns:
            aggregated: (B, num_tokens, 1024)
        """
        B, num_views, num_tokens, dim = view_tokens.shape
        
        # Reshape for attention: (B, 6*num_tokens, 1024)
        tokens_flat = view_tokens.reshape(B, num_views * num_tokens, dim)
        
        # Use attention to aggregate
        query = self.query_token.expand(B, num_tokens, -1)  # (B, num_tokens, 1024)
        aggregated, _ = self.view_aggregator(query, tokens_flat, tokens_flat)
        
        return aggregated  # (B, num_tokens, 1024)
    
    def forward(self, images_dict, view_angles):
        """
        Full forward pass: 6 images → 3D triplane
        
        Args:
            images_dict: Dict with keys ['front', 'back', 'left', 'right', 'top', 'bottom']
                         Each value is (B, 3, 512, 512)
            view_angles: (B, 6, 2) - (azimuth, elevation) for each view
        
        Returns:
            triplane: Triplane representation for mesh extraction
        """
        # Step 1: Encode all 6 views
        view_tokens = self.encode_views(images_dict, view_angles)
        
        # Step 2: Aggregate views
        aggregated_tokens = self.aggregate_views(view_tokens)
        
        # Step 3: Pass through TripoSR's transformer
        latents = self.tokenizer(aggregated_tokens)
        
        # Step 4: Generate triplane
        triplane = self.backbone(latents)
        
        return triplane
    
    def extract_mesh(self, triplane, resolution=256):
        """
        Extract mesh from triplane using marching cubes
        
        Args:
            triplane: Triplane representation from forward()
            resolution: Grid resolution (128, 256, or 512)
        
        Returns:
            vertices: (N, 3) torch.Tensor
            faces: (F, 3) torch.Tensor
        """
        # Use TripoSR's mesh extraction
        with torch.no_grad():
            mesh = self.post_processor(triplane, resolution=resolution)
        
        vertices = mesh['vertices']
        faces = mesh['faces']
        
        return vertices, faces
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters (for optimizer)"""
        return [p for p in self.parameters() if p.requires_grad]
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# ============================================================================
# Loss Functions for Geometry
# ============================================================================

def chamfer_distance_pytorch3d(pred_verts, gt_verts):
    """
    Chamfer distance using PyTorch3D (if available)
    """
    try:
        from pytorch3d.loss import chamfer_distance
        loss, _ = chamfer_distance(
            pred_verts.unsqueeze(0),
            gt_verts.unsqueeze(0)
        )
        return loss
    except ImportError:
        return chamfer_distance_simple(pred_verts, gt_verts)


def chamfer_distance_simple(pred_verts, gt_verts):
    """Simplified Chamfer distance (no PyTorch3D required)"""
    pred_exp = pred_verts.unsqueeze(1)  # (N, 1, 3)
    gt_exp = gt_verts.unsqueeze(0)  # (1, M, 3)
    
    dist = torch.sum((pred_exp - gt_exp) ** 2, dim=-1)  # (N, M)
    
    min_dist_pred = torch.min(dist, dim=1)[0]  # (N,)
    forward_loss = torch.mean(min_dist_pred)
    
    min_dist_gt = torch.min(dist, dim=0)[0]  # (M,)
    backward_loss = torch.mean(min_dist_gt)
    
    return forward_loss + backward_loss


def normal_consistency_loss(pred_verts, pred_faces):
    """Encourage smooth surfaces"""
    try:
        from pytorch3d.structures import Meshes
        from pytorch3d.loss import mesh_normal_consistency
        
        mesh = Meshes(verts=[pred_verts], faces=[pred_faces])
        loss = mesh_normal_consistency(mesh)
        return loss
    except:
        return torch.tensor(0.0, device=pred_verts.device)


def geometry_loss(pred_verts, pred_faces, gt_verts, lambda_chamfer=1.0, lambda_normal=0.01):
    """Combined geometry loss"""
    loss_chamfer = chamfer_distance_pytorch3d(pred_verts, gt_verts)
    loss_normal = normal_consistency_loss(pred_verts, pred_faces)
    
    total_loss = lambda_chamfer * loss_chamfer + lambda_normal * loss_normal
    
    loss_dict = {
        'chamfer': loss_chamfer.item(),
        'normal': loss_normal.item() if isinstance(loss_normal, torch.Tensor) else 0.0,
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict