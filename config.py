"""
UPDATED Configuration for Multi-Scale Mesh Training
Use this to replace your current config.py
"""

import torch
from pathlib import Path

class FixedConfig:
    """Configuration for multi-scale mesh-aware training"""
    
    # ========================================================================
    # Data Paths - UPDATE THESE TO YOUR PATHS
    # ========================================================================
    obj_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/OBJs"
    images_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/input_images"
    
    # Output directories
    checkpoint_dir = "checkpoints_fixed"
    log_dir = "logs_fixed"
    output_dir = "output_fixed"
    
    # ========================================================================
    # Model Architecture Settings
    # ========================================================================
    # NOTE: Multi-scale model uses fixed vertex counts (162, 642, 2562)
    # This parameter is kept for backward compatibility but not used
    num_points = 2562  # Final mesh will have 2562 vertices
    
    hidden_dim = 1024
    feature_dim = 768  # DINOv2 base
    num_views = 6
    
    # ========================================================================
    # Stage 1: Geometry Training Settings  
    # ========================================================================
    batch_size_stage1 = 1
    num_epochs_stage1 = 150  # More epochs needed for multi-scale
    
    # CRITICAL: Much lower learning rate for stable mesh deformation
    learning_rate_stage1 = 1e-5  # 10x LOWER than before!
    
    weight_decay_stage1 = 0.01
    freeze_image_encoder = False  # Fine-tune for better results
    
    # ========================================================================
    # UPDATED Loss Function Weights - MUCH STRONGER REGULARIZATION
    # ========================================================================
    lambda_chamfer = 1.0      # Surface matching
    
    # CRITICAL: Much higher regularization to prevent fragmentation
    lambda_edge = 2.0         # 4x STRONGER (was 0.5)
    lambda_smooth = 1.0       # 5x STRONGER (was 0.2)
    lambda_normal = 0.5       # Normal consistency
    
    # Coverage removed - handled by multi-scale supervision
    
    # ========================================================================
    # Stage 2: Texture Training Settings (Unchanged)
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
    # Early Stopping - More patient for multi-scale training
    # ========================================================================
    patience = 25  # Increased from 20
    min_delta = 1e-5  # More sensitive
    
    # ========================================================================
    # Inference Settings
    # ========================================================================
    geometry_checkpoint = "checkpoints_fixed/multiscale_best.pth"
    texture_checkpoint = "checkpoints/texture_best.pth"
    
    mesh_extraction_method = 'direct'  # Use mesh directly from model
    inference_resolution = 512
    
    # ========================================================================
    # Advanced Settings - UPDATED FOR STABILITY
    # ========================================================================
    grad_clip_norm = 0.5  # Tighter clipping (was 1.0)
    use_amp = False
    seed = 42
    val_split = 0.1
    
    # Warmup settings for learning rate
    warmup_epochs = 10  # Gradual LR warmup
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*70)
        print("MULTI-SCALE MESH TRAINING CONFIGURATION")
        print("="*70)
        print(f"Data:")
        print(f"  OBJ dir:    {cls.obj_dir}")
        print(f"  Images dir: {cls.images_dir}")
        print(f"\nCRITICAL Changes to Prevent Fragmentation:")
        print(f"  ✓ Multi-scale architecture: 162 → 642 → 2562 vertices")
        print(f"  ✓ Learning rate: {cls.learning_rate_stage1} (10x LOWER)")
        print(f"  ✓ Edge loss: {cls.lambda_edge} (4x STRONGER)")
        print(f"  ✓ Smooth loss: {cls.lambda_smooth} (5x STRONGER)")
        print(f"  ✓ Gradient clip: {cls.grad_clip_norm} (tighter)")
        print(f"  ✓ Warmup epochs: {cls.warmup_epochs}")
        print(f"\nStage 1 (Geometry):")
        print(f"  Epochs:       {cls.num_epochs_stage1}")
        print(f"  Batch size:   {cls.batch_size_stage1}")
        print(f"  Final verts:  {cls.num_points}")
        print(f"  Patience:     {cls.patience}")
        print(f"\nDevice: {cls.device}")
        print("="*70)
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        Path(cls.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        Path(cls.log_dir).mkdir(exist_ok=True, parents=True)
        Path(cls.output_dir).mkdir(exist_ok=True, parents=True)
    
    @classmethod
    def get_training_summary(cls):
        """Get a summary of key training parameters"""
        return {
            'learning_rate': cls.learning_rate_stage1,
            'epochs': cls.num_epochs_stage1,
            'regularization': {
                'edge': cls.lambda_edge,
                'smooth': cls.lambda_smooth,
                'normal': cls.lambda_normal
            },
            'grad_clip': cls.grad_clip_norm,
            'warmup_epochs': cls.warmup_epochs
        }


# Create singleton config instance
config = FixedConfig()

# Set random seed
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)

# Create directories on import
config.create_directories()


# ============================================================================
# NOTES FOR DEBUGGING
# ============================================================================
"""
If you still get fragmentation after 50 epochs:

1. LOWER LEARNING RATE EVEN MORE:
   learning_rate_stage1 = 5e-6  # Half current value

2. INCREASE REGULARIZATION:
   lambda_edge = 5.0    # Much stronger
   lambda_smooth = 2.0  # Much stronger

3. TIGHTEN GRADIENT CLIPPING:
   grad_clip_norm = 0.1  # Very tight

4. TRAIN LONGER:
   num_epochs_stage1 = 200
   patience = 40

Monitor these during training:
- Edge loss should be < 0.001 (if > 0.01, mesh is breaking)
- Smooth loss should decrease steadily
- Chamfer loss decreasing WITHOUT edge loss increasing = good sign
"""