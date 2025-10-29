# MIto3D: Multi-View Image to 3D Mesh Generation

A PyTorch-based pipeline for generating textured 3D meshes from multi-view images. This system uses a two-stage approach: geometry reconstruction followed by texture prediction.

## Overview

This project converts 6 orthogonal view images (front, back, left, right, top, bottom) into a complete 3D mesh with colors/textures.

### Architecture

**Stage 1: Geometry Model**
- Input: 6 multi-view images
- Encoder: DINOv2 (facebook/dinov2-base) - 768-dim features
- Aggregation: Permutation-invariant DeepSets-style pooling (max + mean)
- Decoder: MLP → 3D point cloud (default 4096 points)
- Output: 3D mesh vertices and faces

**Stage 2: Texture Model**
- Input: Predicted geometry + 6 multi-view images
- Encoder: DINOv2 for image features
- Decoder: MLP → RGB colors per vertex
- Output: Textured 3D mesh

### Key Features

- **Permutation Invariant**: Random view shuffling during training ensures the model learns content-based patterns, not position-based
- **Image-Only Interface**: No camera angles needed - the model learns purely from visual information
- **Memory Efficient**: Configurable batch size, frozen encoder option, and LayerNorm for stability
- **Production Ready**: Comprehensive test suite, checkpoint management, and error handling

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA GPU (recommended) or Mac M1/M2/M3 with MPS

### Setup

```bash
# Clone this repo
git clone https://github.com/Manas-Vardhan/3DGeneration.git
cd 3DGeneration

# Create virtual environment
python3 -m venv generation
source generation/bin/activate  # On Windows: generation\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
torch>=2.0.0
transformers>=4.30.0
trimesh>=3.21.0
pillow>=9.5.0
numpy>=1.24.0
```

## Data Structure

**Note:** Due to large file sizes, the OBJ files are not included in this repository. You need to provide your own training data.

Your data should be organized as follows:

```
data/
├── OBJs/
│   ├── 1/
│   │   └── shoe_model.obj
│   ├── 2/
│   │   └── shoe_model.obj
│   └── ...
└── input_images/
    ├── 1_front.png
    ├── 1_back.png
    ├── 1_left.png
    ├── 1_right.png
    ├── 1_top.png
    ├── 1_bottom.png
    ├── 2_front.png
    └── ...
```

**Important:**
- Each OBJ file should be in a subdirectory named with the shoe ID
- Images should be named `{shoe_id}_{view}.{png|jpg|jpeg}`
- All 6 views are required for each shoe: front, back, left, right, top, bottom
- OBJ files are gitignored due to large file sizes

### Setting Up Your Data

1. Create the required directories:
```bash
mkdir -p data/OBJs data/input_images
```

2. Place your OBJ files in subdirectories:
```bash
data/OBJs/1/shoe_model.obj
data/OBJs/2/shoe_model.obj
...
```

3. Place your multi-view images:
```bash
data/input_images/1_front.png
data/input_images/1_back.png
...
```

4. Verify your setup:
```bash
python test_training_startup.py
```

## Configuration

Edit [config.py](config.py) to customize training settings:

```python
# Data paths
obj_dir = "data/OBJs"
images_dir = "data/input_images"

# Model settings
num_points = 4096              # Number of predicted 3D points
freeze_image_encoder = False   # Set True to freeze DINOv2

# Training settings
batch_size_stage1 = 1          # Geometry training batch size
batch_size_stage2 = 1          # Texture training batch size
num_epochs_stage1 = 50         # Geometry training epochs
num_epochs_stage2 = 30         # Texture training epochs
learning_rate_stage1 = 5e-5    # Geometry learning rate
learning_rate_stage2 = 1e-4    # Texture learning rate

# Device
device = "mps"  # or "cuda" or "cpu"
```

### Configuration Presets

```python
# Fast training for testing
config = FastTrainingConfig()

# High quality, longer training
config = HighQualityConfig()

# For limited GPU memory
config = MemoryEfficientConfig()
```

## Usage

### 1. Test Your Setup

Before training, verify everything is configured correctly:

```bash
python test_training_startup.py
```

This runs 10 tests covering:
- Imports and dependencies
- Configuration validation
- Dataset loading and X-Y mapping verification
- Model initialization (geometry + texture)
- Forward pass and loss computation
- Training step execution
- Checkpoint saving/loading

Expected output:
```
🎉 ALL TESTS PASSED! Safe to deploy to cloud! 🎉
```

### 2. Test Permutation Invariance

Verify that the models are truly permutation invariant:

```bash
python test_permutation_invariance.py
```

This ensures that shuffling input view order produces identical outputs.

### 3. Train Stage 1 (Geometry)

Train the geometry reconstruction model:

```bash
python train_geometry.py
```

**What happens:**
- Loads multi-view images and ground truth meshes
- Randomly shuffles view order each epoch (permutation invariance)
- Predicts 3D point clouds from images
- Optimizes using Chamfer distance loss
- Saves checkpoints to `checkpoints/geometry_best.pth`

**Training time:** ~2-3 hours for 50 epochs on Mac M1 (106 shoes)

### 4. Train Stage 2 (Texture)

After geometry training completes, train the texture model:

```bash
python train_texture.py
```

**What happens:**
- Loads geometry model (frozen)
- Predicts 3D geometry from images
- Predicts RGB colors for each vertex
- Optimizes using L1/L2 + perceptual loss
- Saves checkpoints to `checkpoints/texture_best.pth`

**Training time:** ~1-2 hours for 30 epochs on Mac M1 (106 shoes)

### 5. Generate 3D Meshes

Use the trained models to generate meshes from new images:

```bash
python inference.py
```

**Output:**
- Generates textured 3D mesh (.obj + .mtl + texture image)
- Saves to `output/` directory

## Model Architecture Details

### Geometry Model

```
Input: 6 images (B, 3, 512, 512)
    ↓
DINOv2 Encoder (frozen or fine-tuned)
    ↓
Per-view features (B, 6, 768)
    ↓
ViewAggregator (DeepSets pooling)
    - Max pooling across views
    - Mean pooling across views
    - Concatenate [max, mean]
    - MLP projection
    ↓
Aggregated features (B, 768)
    ↓
PointCloudDecoder (MLP with LayerNorm)
    - FC(768 → 1024) + ReLU + LayerNorm
    - FC(1024 → 2048) + ReLU + LayerNorm
    - FC(2048 → 2048) + ReLU + LayerNorm
    - FC(2048 → num_points * 3)
    ↓
Output: Point cloud (B, num_points, 3)
```

**Parameters:**
- Total: ~120M (with DINOv2 trainable)
- Trainable: ~3M (with DINOv2 frozen)

### Texture Model

```
Input: Vertices (N, 3) + 6 images (1, 3, 512, 512)
    ↓
Geometry Model (frozen)
    ↓
Predicted vertices (N, 3)
    ↓
DINOv2 per-view features (6, 768)
    ↓
ViewAggregator
    ↓
Aggregated features (768,)
    ↓
TextureDecoder per vertex
    - Vertex position (3)
    - Aggregated image features (768)
    - MLP → RGB (3)
    ↓
Output: Vertex colors (N, 3)
```

**Parameters:**
- Total: ~2M trainable

## Loss Functions

### Geometry Loss

```python
loss = λ_chamfer * chamfer_distance(pred, gt) +
       λ_coverage * coverage_loss(pred, gt)
```

- **Chamfer Distance**: Bidirectional nearest-neighbor distance between predicted and ground truth point clouds
- **Coverage Loss**: Ensures predicted points cover the entire ground truth surface

### Texture Loss

```python
loss = λ_l1l2 * (L1 + L2)(pred_colors, gt_colors) +
       λ_perceptual * perceptual_loss(pred, gt) +
       λ_smooth * smoothness_loss(pred_colors)
```

- **L1/L2 Loss**: Direct color matching
- **Perceptual Loss**: Feature-space similarity using DINOv2
- **Smoothness Loss**: Encourages smooth color transitions

## Project Structure

```
.
├── README.md
├── requirements.txt
├── config.py                      # Configuration settings
├── models/
│   ├── geometry_model.py          # Stage 1: Geometry prediction
│   └── texture_model.py           # Stage 2: Texture prediction
├── train_geometry.py              # Stage 1 training script
├── train_texture.py               # Stage 2 training script
├── inference.py                   # Generate meshes from trained models
├── load_data.py                   # Dataset and data loading
├── test_training_startup.py       # Comprehensive test suite
├── test_permutation_invariance.py # Permutation invariance tests
├── utils/
│   ├── prepare_dataset.py         # Legacy dataset utilities
│   └── rename_to_IDs.py           # Data preprocessing utilities
├── data/
│   ├── OBJs/                      # Ground truth 3D models
│   └── input_images/              # Multi-view images
├── checkpoints/                   # Saved model checkpoints
├── logs/                          # Training logs
└── output/                        # Generated meshes
```

## Key Implementation Details

### Permutation Invariance

The model uses symmetric pooling operations (max + mean) to ensure output is identical regardless of view order. During training, views are randomly shuffled each iteration:

```python
def _shuffle_views(self, images_dict):
    """Randomly permute view order for permutation invariance"""
    view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
    perm = list(range(6))
    random.shuffle(perm)

    stacked = torch.stack([images_dict[name] for name in view_names], dim=1)
    shuffled = stacked[:, perm, :, :, :]

    return {view_names[i]: shuffled[:, i] for i in range(6)}
```

### BatchNorm vs LayerNorm

The project uses LayerNorm instead of BatchNorm because:
- Works with batch size = 1 (critical for large models on limited memory)
- More stable for point cloud generation tasks
- No dependency on batch statistics

### Memory Optimization

For limited GPU memory:
- Set `freeze_image_encoder = True` to freeze DINOv2 (saves ~86M parameters)
- Use `batch_size = 1`
- Reduce `num_points` to 2048 or 1024
- Enable gradient checkpointing (if needed)

## Troubleshooting

### Out of Memory (OOM)

```python
# In config.py
freeze_image_encoder = True  # Freeze DINOv2
batch_size_stage1 = 1
num_points = 2048           # Reduce from 4096
```

### Dataset Loading Errors

```bash
# Verify data structure
python test_training_startup.py

# The test will report any X-Y mapping issues
```

### Training Not Converging

- Reduce learning rate: `learning_rate_stage1 = 1e-5`
- Increase epochs: `num_epochs_stage1 = 100`
- Try unfreezing encoder: `freeze_image_encoder = False`
- Check data quality (ensure images are well-aligned)

### Model Output is Poor Quality

- Increase number of points: `num_points = 8192`
- Train longer: `num_epochs_stage1 = 100`
- Unfreeze encoder for fine-tuning
- Ensure high-quality input images (512x512 minimum)
- Check ground truth mesh quality

## Performance

**Hardware:** Mac M1 (8GB RAM)
- **Stage 1 Training:** ~2-3 hours (50 epochs, 106 shoes)
- **Stage 2 Training:** ~1-2 hours (30 epochs, 106 shoes)
- **Inference:** ~5-10 seconds per shoe

**Hardware:** NVIDIA RTX 3090
- **Stage 1 Training:** ~30-45 minutes (50 epochs, 106 shoes)
- **Stage 2 Training:** ~15-30 minutes (30 epochs, 106 shoes)
- **Inference:** ~1-2 seconds per shoe

## Built With

- [PyTorch](https://pytorch.org/) - Deep learning framework
- [Transformers](https://huggingface.co/docs/transformers/) - DINOv2 model
- [Trimesh](https://trimsh.org/) - 3D mesh processing
- [Pillow](https://python-pillow.org/) - Image processing

## Authors

**Manas Vardhan** - [Github](https://github.com/Manas-Vardhan)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

- **DINOv2**: Meta AI's vision transformer for feature extraction
- **DeepSets**: Permutation-invariant set functions framework
- **Trimesh**: Comprehensive 3D mesh processing library
