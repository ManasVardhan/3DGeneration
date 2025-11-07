"""
FIXED Configuration for mesh-aware training
Drop this in as config.py replacement
"""

import torch
from pathlib import Path

class ImprovedConfig:
    """FIXED configuration that prevents mesh fragmentation"""
    
    # ========================================================================
    # Data Paths (keep your existing paths)
    # ========================================================================
    obj_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/OBJs"
    images_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/input_images"
    
    # Output directories
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    output_dir = "output"
    
    # ========================================================================
    # Model Architecture Settings
    # ========================================================================
    num_points = 8192  # Will use closest mesh size (actual: 10242 vertices)
                       # The model maintains mesh connectivity internally
    
    hidden_dim = 1024
    feature_dim = 768  # DINOv2 base
    num_views = 6
    
    # ========================================================================
    # Stage 1: Geometry Training Settings  
    # ========================================================================
    batch_size_stage1 = 1
    num_epochs_stage1 = 100
    
    # CRITICAL: Lower learning rate for stable mesh deformation
    learning_rate_stage1 = 5e-5  # Much lower than original!
    
    weight_decay_stage1 = 0.01
    freeze_image_encoder = False  # Fine-tune for better results
    
    # ========================================================================
    # FIXED Loss Function Weights - Prevent fragmentation
    # ========================================================================
    lambda_chamfer = 1.0      # Surface matching
    lambda_coverage = 0.1     # Coverage
    
    # INCREASED regularization to maintain connectivity
    lambda_edge = 0.5         # Edge length (prevent collapse/explosion)
    lambda_normal = 0.1       # Normal consistency  
    lambda_smooth = 0.2       # Laplacian smoothing (surface quality)
    
    # ========================================================================
    # Stage 2: Texture Training Settings
    # ========================================================================
    batch_size_stage2 = 1
    num_epochs_stage2 = 30
    learning_rate_stage2 = 1e-4
    weight_decay_stage2 = 0.01
    
    lambda_color_l1l2 = 1.0
    lambda_perceptual = 0.5
    lambda_smooth_texture = 0.1
    
    # ========================================================================
    # Data Loading Settings
    # ========================================================================
    image_size = 512
    num_workers = 0  # Mac MPS requires 0
    
    # ========================================================================
    # Device Settings
    # ========================================================================
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========================================================================
    # Logging & Checkpointing
    # ========================================================================
    log_interval = 5
    save_interval = 10
    
    # ========================================================================
    # Early Stopping
    # ========================================================================
    patience = 20
    min_delta = 5e-5
    
    # ========================================================================
    # Inference Settings
    # ========================================================================
    geometry_checkpoint = "checkpoints/geometry_best.pth"
    texture_checkpoint = "checkpoints/texture_best.pth"
    
    mesh_extraction_method = 'poisson'  # Better than convex hull
    inference_resolution = 512
    
    # ========================================================================
    # Advanced Settings
    # ========================================================================
    grad_clip_norm = 1.0
    use_amp = False
    seed = 42
    val_split = 0.1
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*70)
        print("MESH-AWARE CONFIGURATION (Fragmentation Fix)")
        print("="*70)
        print(f"Data:")
        print(f"  OBJ dir:    {cls.obj_dir}")
        print(f"  Images dir: {cls.images_dir}")
        print(f"\nCritical Changes to Prevent Fragmentation:")
        print(f"  ✓ Lower learning rate: {cls.learning_rate_stage1}")
        print(f"  ✓ Edge loss weight: {cls.lambda_edge} (maintains connectivity)")
        print(f"  ✓ Smooth loss weight: {cls.lambda_smooth} (surface quality)")
        print(f"  ✓ Gradient clipping: {cls.grad_clip_norm}")
        print(f"\nStage 1 (Geometry):")
        print(f"  Epochs:       {cls.num_epochs_stage1}")
        print(f"  Batch size:   {cls.batch_size_stage1}")
        print(f"  Points:       {cls.num_points}")
        print(f"\nDevice: {cls.device}")
        print("="*70)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        Path(cls.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(cls.log_dir).mkdir(exist_ok=True, parents=True)
        Path(cls.output_dir).mkdir(exist_ok=True, parents=True)


# Create singleton config instance
config = ImprovedConfig()

# Set random seed
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)