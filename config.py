"""
IMPROVED Configuration for Multi-View to 3D Mesh Training
Key improvements over original:
- Higher point count (8192 instead of 4096)
- Better learning rate schedule
- Additional loss weights
- Unfrozen encoder for better quality
"""

import torch
from pathlib import Path

class ImprovedConfig:
    """IMPROVED Training configuration"""
    
    # ========================================================================
    # Data Paths
    # ========================================================================
    obj_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/OBJs"
    images_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/input_images"
    
    # Output directories
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    output_dir = "output"
    
    # ========================================================================
    # IMPROVED Model Architecture Settings
    # ========================================================================
    num_points = 8192  # INCREASED from 4096 for better detail
                       # More points = more geometric detail
    
    hidden_dim = 1024  # Keep same
    
    # ========================================================================
    # IMPROVED Stage 1: Geometry Training Settings
    # ========================================================================
    
    freeze_image_encoder = False  # CHANGED: Fine-tune for better results
                                 # Set to True if running out of memory
    
    feature_dim = 768
    num_views = 6
    
    # Training hyperparameters
    batch_size_stage1 = 1  # Keep same (memory constraints)
    
    num_epochs_stage1 = 100  # INCREASED from 50 - need more training
    
    # IMPROVED LEARNING RATE SCHEDULE
    learning_rate_stage1 = 2e-4  # INCREASED from 5e-5
                                 # Start higher, use better decay
    
    lr_schedule = 'cosine_restart'  # NEW: Use cosine with warm restarts
                                    # Options: 'cosine', 'cosine_restart', 'plateau'
    
    lr_min = 1e-6  # NEW: Minimum learning rate (instead of 5e-8)
    
    warmup_epochs = 5  # NEW: Learning rate warmup
    
    weight_decay_stage1 = 0.01
    
    # ========================================================================
    # IMPROVED Loss Function Weights
    # ========================================================================
    # Main losses
    lambda_chamfer = 1.0
    lambda_coverage = 0.1
    
    # NEW: Additional regularization losses
    lambda_edge = 0.05        # Edge length regularization
    lambda_normal = 0.01      # Normal consistency
    lambda_smooth = 0.005     # Laplacian smoothness
    
    # ========================================================================
    # Stage 2: Texture Training Settings (unchanged)
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
    # Early Stopping (more patient)
    # ========================================================================
    patience = 20  # INCREASED from 15 - allow more time to improve
    min_delta = 5e-5  # DECREASED - more sensitive to improvements
    
    # ========================================================================
    # Inference Settings
    # ========================================================================
    geometry_checkpoint = "checkpoints/geometry_best.pth"
    texture_checkpoint = "checkpoints/texture_best.pth"
    
    mesh_extraction_method = 'poisson'  # NEW: Use Poisson instead of convex hull
                                       # Options: 'poisson', 'ball_pivot', 'alpha_shape'
    
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
        print("IMPROVED CONFIGURATION")
        print("="*70)
        print(f"Data:")
        print(f"  OBJ dir:    {cls.obj_dir}")
        print(f"  Images dir: {cls.images_dir}")
        print(f"\nKey Improvements:")
        print(f"  ✓ Point count: 4096 → {cls.num_points}")
        print(f"  ✓ Learning rate: 5e-5 → {cls.learning_rate_stage1}")
        print(f"  ✓ LR schedule: {cls.lr_schedule}")
        print(f"  ✓ Min LR: {cls.lr_min}")
        print(f"  ✓ Fine-tune encoder: {not cls.freeze_image_encoder}")
        print(f"  ✓ Mesh extraction: {cls.mesh_extraction_method}")
        print(f"\nStage 1 (Geometry):")
        print(f"  Epochs:       {cls.num_epochs_stage1}")
        print(f"  Batch size:   {cls.batch_size_stage1}")
        print(f"  Learning rate: {cls.learning_rate_stage1}")
        print(f"  Warmup epochs: {cls.warmup_epochs}")
        print(f"\nLoss weights:")
        print(f"  Chamfer:      {cls.lambda_chamfer}")
        print(f"  Coverage:     {cls.lambda_coverage}")
        print(f"  Edge length:  {cls.lambda_edge}")
        print(f"  Normal:       {cls.lambda_normal}")
        print(f"  Smoothness:   {cls.lambda_smooth}")
        print(f"\nStage 2 (Texture):")
        print(f"  Epochs:       {cls.num_epochs_stage2}")
        print(f"  Batch size:   {cls.batch_size_stage2}")
        print(f"  Learning rate: {cls.learning_rate_stage2}")
        print(f"\nDevice: {cls.device}")
        print("="*70)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        Path(cls.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(cls.log_dir).mkdir(exist_ok=True, parents=True)
        Path(cls.output_dir).mkdir(exist_ok=True, parents=True)
        print(f"✓ Created directories:")
        print(f"  {cls.checkpoint_dir}/")
        print(f"  {cls.log_dir}/")
        print(f"  {cls.output_dir}/")


# Create singleton config instance
config = ImprovedConfig()

# Set random seed
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)


# ============================================================================
# Quick Test Config (for debugging)
# ============================================================================

class QuickTestConfig(ImprovedConfig):
    """Very quick training for testing changes"""
    num_epochs_stage1 = 5
    num_epochs_stage2 = 5
    num_points = 2048
    freeze_image_encoder = True
    learning_rate_stage1 = 1e-4


# ============================================================================
# Usage
# ============================================================================

if __name__ == "__main__":
    print("\nImproved Configuration:")
    config.print_config()
    print("\n")
    config.create_directories()