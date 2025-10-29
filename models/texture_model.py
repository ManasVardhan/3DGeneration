"""
Stage 2: Texture Model
Vertex Color Prediction from Geometry + Images
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VertexColorPredictor(nn.Module):
    """Predict RGB colors for each vertex"""
    
    def __init__(self, feature_dim=1024, use_all_views=True):
        super().__init__()
        
        self.use_all_views = use_all_views
        
        # Image encoder (DINOv2)
        print("Loading DINOv2 for texture prediction...")
        from transformers import AutoModel
        self.image_encoder = AutoModel.from_pretrained('facebook/dinov2-base')
        self.image_feat_dim = 768
        print("✓ DINOv2 loaded")
        
        # Freeze image encoder
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
        # Vertex position encoder
        self.vertex_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512)
        )
        
        # Multi-view aggregation (permutation invariant)
        if use_all_views:
            # Use symmetric pooling instead of attention for permutation invariance
            self.feature_projection = nn.Linear(self.image_feat_dim * 2, self.image_feat_dim)
        
        # Color decoder
        color_input_dim = self.image_feat_dim + 512
        self.color_decoder = nn.Sequential(
            nn.Linear(color_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    
    def encode_images(self, images_dict):
        """Encode images with DINOv2"""
        with torch.no_grad():
            if self.use_all_views:
                features = []
                for view in ['front', 'back', 'left', 'right', 'top', 'bottom']:
                    img = images_dict[view]
                    feat = self.image_encoder(img).last_hidden_state
                    feat = feat.mean(dim=1)
                    features.append(feat)
                
                image_features = torch.stack(features, dim=1)
            else:
                img = images_dict['front']
                feat = self.image_encoder(img).last_hidden_state
                image_features = feat.mean(dim=1)
        
        return image_features
    
    def aggregate_multi_view_features(self, image_features):
        """
        Aggregate features from 6 views using permutation-invariant symmetric pooling.

        Args:
            image_features: (B, 6, 768) - features from 6 views

        Returns:
            aggregated: (B, 768) - aggregated features (same output for any view order)
        """
        if not self.use_all_views:
            return image_features

        # Symmetric aggregation: max + mean pooling (order-independent!)
        max_pool = torch.max(image_features, dim=1)[0]   # (B, 768)
        mean_pool = torch.mean(image_features, dim=1)    # (B, 768)

        # Concatenate and project back to original dimension
        combined = torch.cat([max_pool, mean_pool], dim=1)  # (B, 1536)
        aggregated = self.feature_projection(combined)      # (B, 768)

        return aggregated
    
    def forward(self, vertices, images_dict):
        """Predict colors for vertices"""
        N = vertices.shape[0]
        
        vert_feat = self.vertex_encoder(vertices)
        image_features = self.encode_images(images_dict)
        
        if self.use_all_views:
            image_features = self.aggregate_multi_view_features(image_features)
        
        img_feat_exp = image_features.expand(N, -1)
        combined = torch.cat([img_feat_exp, vert_feat], dim=-1)
        colors = self.color_decoder(combined)
        
        return colors


# ============================================================================
# Loss Functions
# ============================================================================

def color_loss_l1_l2(pred_colors, gt_colors, lambda_l1=0.1, lambda_l2=1.0):
    """Combined L1 + L2 loss"""
    l2_loss = torch.mean((pred_colors - gt_colors) ** 2)
    l1_loss = torch.mean(torch.abs(pred_colors - gt_colors))
    return lambda_l2 * l2_loss + lambda_l1 * l1_loss


def perceptual_color_loss(pred_colors, gt_colors):
    """Perceptual color loss"""
    weights = torch.tensor([0.299, 0.587, 0.114], device=pred_colors.device)
    diff = pred_colors - gt_colors
    weighted_diff = diff * weights.unsqueeze(0)
    loss = torch.mean(torch.sum(weighted_diff ** 2, dim=-1))
    return loss


def color_smoothness_loss(colors, faces):
    """Encourage smooth color transitions"""
    if faces is None or len(faces) == 0:
        return torch.tensor(0.0, device=colors.device)
    
    colors_v0 = colors[faces[:, 0]]
    colors_v1 = colors[faces[:, 1]]
    colors_v2 = colors[faces[:, 2]]
    
    diff_01 = torch.mean((colors_v0 - colors_v1) ** 2)
    diff_12 = torch.mean((colors_v1 - colors_v2) ** 2)
    diff_20 = torch.mean((colors_v2 - colors_v0) ** 2)
    
    return (diff_01 + diff_12 + diff_20) / 3.0


def texture_loss(pred_colors, gt_colors, faces=None, 
                 lambda_l1l2=1.0, lambda_perceptual=0.5, lambda_smooth=0.1):
    """Combined texture loss"""
    loss_l1l2 = color_loss_l1_l2(pred_colors, gt_colors)
    loss_perceptual = perceptual_color_loss(pred_colors, gt_colors)
    loss_smooth = color_smoothness_loss(pred_colors, faces) if faces is not None else 0.0
    
    total_loss = (
        lambda_l1l2 * loss_l1l2 +
        lambda_perceptual * loss_perceptual +
        lambda_smooth * loss_smooth
    )
    
    loss_dict = {
        'l1l2': loss_l1l2.item(),
        'perceptual': loss_perceptual.item(),
        'smooth': loss_smooth.item() if isinstance(loss_smooth, torch.Tensor) else 0.0,
        'total': total_loss.item()
    }
    
    return total_loss, loss_dict


def match_vertex_counts(pred_colors, gt_colors, pred_verts, gt_verts):
    """Match number of vertices between prediction and GT"""
    num_pred = pred_colors.shape[0]
    num_gt = gt_colors.shape[0]
    
    if num_pred == num_gt:
        return pred_colors, gt_colors
    elif num_pred > num_gt:
        indices = torch.randperm(num_pred)[:num_gt]
        return pred_colors[indices], gt_colors
    else:
        indices = torch.randperm(num_gt)[:num_pred]
        return pred_colors, gt_colors[indices]