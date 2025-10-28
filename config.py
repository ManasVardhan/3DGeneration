"""
Configuration file for Multi-View to 3D Mesh Training
Edit this file to customize training settings
"""

import torch
from pathlib import Path

class Config:
    """Training configuration"""
    
    # ========================================================================
    # Data Paths
    # ========================================================================
    # UPDATE THESE PATHS TO YOUR DATA LOCATION
    obj_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/Completed"
    images_dir = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/input_images"
    
    # Output directories
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    output_dir = "output"
    
    # ========================================================================
    # Model Architecture Settings
    # ========================================================================
    # Geometry model
    num_points = 4096  # Number of 3D points to predict
                       # 2048: Faster, lower quality
                       # 4096: Balanced (recommended)
                       # 8192: Slower, higher quality
    
    hidden_dim = 1024  # Hidden layer size
    
    # Stage 1: Geometry Training Settings
    # ========================================================================
    # Model architecture
    freeze_image_encoder = False  # Set True to freeze DINOv2 (faster, less memory)
                                 # False: Fine-tune everything (better quality)
                                 # True: Only train decoder (faster)
    
    feature_dim = 768  # DINOv2-base feature dimension (don't change)
    num_views = 6       # Number of views (don't change)
    
    # Training hyperparameters
    batch_size_stage1 = 1  # TripoSR is memory-heavy
                          # Mac M1/M2: Use 1
                          # RTX 3090: Can use 2-4
                          # A100: Can use 4-8
    
    num_epochs_stage1 = 50  # Number of training epochs
                           # 50: Good balance
                           # 100: Better quality, longer training
                           # 30: Quick test
    
    learning_rate_stage1 = 5e-5  # Lower for fine-tuning
                                # Don't increase above 1e-4
    
    weight_decay_stage1 = 0.01  # Regularization
    
    # ========================================================================
    # Stage 2: Texture Training Settings
    # ========================================================================
    # Training hyperparameters
    batch_size_stage2 = 1  # Same as stage 1
    
    num_epochs_stage2 = 30  # Texture usually needs fewer epochs
                           # 30: Good balance
                           # 50: Better quality
                           # 20: Quick test
    
    learning_rate_stage2 = 1e-4  # Higher than stage 1
                                # Training from scratch
    
    weight_decay_stage2 = 0.01
    
    # ========================================================================
    # Data Loading Settings
    # ========================================================================
    image_size = 512  # TripoSR expects 512x512 (don't change)
    
    num_workers = 0  # Number of worker threads for data loading
                    # Mac MPS: MUST be 0
                    # CUDA: Can use 2-4 for faster loading
    
    # ========================================================================
    # Device Settings
    # ========================================================================
    # Automatically detect best available device
    device = "mps" if torch.backends.mps.is_available() else \
             "cuda" if torch.cuda.is_available() else "cpu"
    
    # Or manually set:
    # device = "mps"   # Mac M1/M2/M3
    # device = "cuda"  # NVIDIA GPU
    # device = "cpu"   # CPU only (very slow)
    
    # ========================================================================
    # Logging & Checkpointing
    # ========================================================================
    log_interval = 5   # Log training progress every N batches
                      # Lower = more frequent logging
                      # 1: Log every batch (verbose)
                      # 10: Less frequent
    
    save_interval = 10  # Save checkpoint every N epochs
                       # 5: Save more frequently
                       # 10: Balanced
                       # 20: Save less frequently
    
    # ========================================================================
    # Early Stopping
    # ========================================================================
    patience = 15      # Stop if no improvement for N epochs
                      # 10: Stop early
                      # 15: Balanced
                      # 20: Train longer
    
    min_delta = 1e-4  # Minimum improvement to count as progress
                      # Smaller = more sensitive
    
    # ========================================================================
    # Loss Function Weights
    # ========================================================================
    # Geometry loss weights
    lambda_chamfer = 1.0      # Main geometry loss
    lambda_normal = 0.01      # Normal consistency (smoothness)
    
    # Texture loss weights
    lambda_color_l1l2 = 1.0   # Main color matching
    lambda_perceptual = 0.5   # Perceptual similarity
    lambda_smooth = 0.1       # Color smoothness
    
    # ========================================================================
    # Inference Settings
    # ========================================================================
    # Default model paths for inference
    geometry_checkpoint = "checkpoints/triposr_mv_best.pth"
    texture_checkpoint = "checkpoints/texture_best.pth"
    
    # Inference resolution
    inference_resolution = 256  # Can set higher for inference
                               # 256: Fast
                               # 512: Higher quality
    
    # ========================================================================
    # Advanced Settings (Usually don't need to change)
    # ========================================================================
    # Gradient clipping
    grad_clip_norm = 1.0  # Prevent exploding gradients
    
    # Mixed precision training (experimental)
    use_amp = False  # Automatic Mixed Precision
                    # Can enable on CUDA for faster training
                    # Not supported on MPS
    
    # Reproducibility
    seed = 42  # Random seed for reproducibility
    
    # ========================================================================
    # Validation Settings
    # ========================================================================
    val_split = 0.1  # Fraction of data for validation (not implemented yet)
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("="*70)
        print("CONFIGURATION")
        print("="*70)
        print(f"Data:")
        print(f"  OBJ dir:    {cls.obj_dir}")
        print(f"  Images dir: {cls.images_dir}")
        print(f"\nStage 1 (Geometry):")
        print(f"  Epochs:       {cls.num_epochs_stage1}")
        print(f"  Batch size:   {cls.batch_size_stage1}")
        print(f"  Learning rate: {cls.learning_rate_stage1}")
        print(f"  Freeze encoder: {cls.freeze_image_encoder}")
        print(f"\nStage 2 (Texture):")
        print(f"  Epochs:       {cls.num_epochs_stage2}")
        print(f"  Batch size:   {cls.batch_size_stage2}")
        print(f"  Learning rate: {cls.learning_rate_stage2}")
        print(f"\nDevice:")
        print(f"  {cls.device}")
        print(f"\nMesh resolution: {cls.mesh_resolution}")
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
config = Config()

# Set random seed for reproducibility
torch.manual_seed(config.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(config.seed)


# ============================================================================
# Configuration Presets
# ============================================================================

class FastTrainingConfig(Config):
    """Quick training for testing"""
    num_epochs_stage1 = 10
    num_epochs_stage2 = 10
    mesh_resolution = 128
    freeze_image_encoder = True
    batch_size_stage1 = 1
    batch_size_stage2 = 1


class HighQualityConfig(Config):
    """High quality, longer training"""
    num_epochs_stage1 = 100
    num_epochs_stage2 = 50
    mesh_resolution = 512
    freeze_image_encoder = False
    learning_rate_stage1 = 3e-5  # Even lower for careful fine-tuning


class MemoryEfficientConfig(Config):
    """For limited GPU memory"""
    batch_size_stage1 = 1
    batch_size_stage2 = 1
    mesh_resolution = 128
    freeze_image_encoder = True
    num_workers = 0


# ============================================================================
# Usage Examples
# ============================================================================

if __name__ == "__main__":
    # Print default configuration
    print("\nDefault Configuration:")
    config.print_config()
    
    # Create directories
    print("\n")
    config.create_directories()
    
    # Example: Use a preset
    print("\n\nFast Training Preset:")
    fast_config = FastTrainingConfig()
    fast_config.print_config()