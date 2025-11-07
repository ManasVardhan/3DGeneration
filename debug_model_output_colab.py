"""
Debug Model Output - Visualize predicted point cloud vs ground truth
Copy this into Colab to diagnose what's going wrong
"""

# ============================================================================
# DEBUG CELL - Diagnose Model Predictions
# ============================================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import trimesh

# Add your project to path
import sys
sys.path.append('/content/3DGeneration')

from config import config
from models.geometry_model import GeometryModel

# Configuration
SHOE_ID = "2"  # ← Change this
CHECKPOINT_PATH = "/content/3DGeneration/checkpoints/geometry_improved_best.pth"
IMAGES_DIR = "/content/3DGeneration/data/input_images"
OBJ_DIR = "/content/3DGeneration/data/OBJs"

print("="*70)
print("DEBUGGING MODEL OUTPUT")
print("="*70)

# ============================================================================
# 1. Load Model
# ============================================================================

print("\n[1/5] Loading model...")
model = GeometryModel(
    num_points=config.num_points,
    freeze_encoder=True
).to(config.device)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=config.device)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded from epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Training loss: {checkpoint.get('loss', 'unknown'):.6f}")
else:
    model.load_state_dict(checkpoint)

model.eval()
print("✓ Model loaded")

# ============================================================================
# 2. Load Test Images
# ============================================================================

print(f"\n[2/5] Loading images for shoe {SHOE_ID}...")
images_dir = Path(IMAGES_DIR)
views = ['front', 'back', 'left', 'right', 'top', 'bottom']

transform = transforms.Compose([
    transforms.Resize((config.image_size, config.image_size)),
    transforms.ToTensor(),
])

images_dict = {}
for view in views:
    img_path = None
    for ext in ['.png', '.jpeg', '.jpg']:
        test_path = images_dir / f"{SHOE_ID}_{view}{ext}"
        if test_path.exists():
            img_path = test_path
            break

    if img_path is None:
        print(f"❌ Missing {view} image")
        sys.exit(1)

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(config.device)
    images_dict[view] = img_tensor

print("✓ Images loaded")

# ============================================================================
# 3. Load Ground Truth Mesh
# ============================================================================

print(f"\n[3/5] Loading ground truth mesh...")
obj_paths = list(Path(OBJ_DIR).rglob("*.obj"))
gt_mesh_path = None

for obj_path in obj_paths:
    if obj_path.parent.name == SHOE_ID:
        gt_mesh_path = obj_path
        break

if gt_mesh_path is None:
    print(f"❌ No ground truth mesh found for shoe {SHOE_ID}")
    sys.exit(1)

gt_mesh = trimesh.load(gt_mesh_path, process=False, force='mesh')
gt_vertices = gt_mesh.vertices.copy()

# Normalize (same as training)
centroid = gt_vertices.mean(axis=0)
gt_vertices = gt_vertices - centroid
max_dist = np.abs(gt_vertices).max()
if max_dist > 0:
    gt_vertices = gt_vertices / max_dist

print(f"✓ Ground truth loaded: {len(gt_vertices)} vertices")

# ============================================================================
# 4. Run Model Inference
# ============================================================================

print(f"\n[4/5] Running model inference...")
with torch.no_grad():
    pred_points, pred_normals = model(images_dict)

pred_points_np = pred_points[0].cpu().numpy()  # (num_points, 3)
pred_normals_np = pred_normals[0].cpu().numpy()  # (num_points, 3)

print(f"✓ Predicted {len(pred_points_np)} points")
print(f"  Point cloud range: [{pred_points_np.min():.3f}, {pred_points_np.max():.3f}]")
print(f"  Mean: {pred_points_np.mean(axis=0)}")
print(f"  Std: {pred_points_np.std(axis=0)}")

# ============================================================================
# 5. Visualize Comparison
# ============================================================================

print(f"\n[5/5] Generating visualizations...")

# ========================================================================
# VISUALIZATION 1: Side-by-Side Comparison
# ========================================================================

fig = plt.figure(figsize=(20, 8))

views_3d = [
    ("Front", 0, 0),
    ("Side", 0, 90),
    ("Top", 90, 0),
]

for idx, (view_name, elev, azim) in enumerate(views_3d):
    # Ground truth
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    ax.scatter(gt_vertices[:, 0], gt_vertices[:, 1], gt_vertices[:, 2],
               c='blue', s=1, alpha=0.3, label='Ground Truth')
    ax.set_title(f'Ground Truth - {view_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

    # Predicted
    ax = fig.add_subplot(2, 3, idx + 4, projection='3d')
    ax.scatter(pred_points_np[:, 0], pred_points_np[:, 1], pred_points_np[:, 2],
               c='red', s=1, alpha=0.3, label='Predicted')
    ax.set_title(f'Predicted - {view_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

plt.suptitle(f'Ground Truth vs Predicted - Shoe {SHOE_ID}', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ========================================================================
# VISUALIZATION 2: Overlay
# ========================================================================

fig = plt.figure(figsize=(20, 5))

for idx, (view_name, elev, azim) in enumerate(views_3d):
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')

    # Plot both
    ax.scatter(gt_vertices[:, 0], gt_vertices[:, 1], gt_vertices[:, 2],
               c='blue', s=2, alpha=0.3, label='Ground Truth')
    ax.scatter(pred_points_np[:, 0], pred_points_np[:, 1], pred_points_np[:, 2],
               c='red', s=2, alpha=0.3, label='Predicted')

    ax.set_title(f'Overlay - {view_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])

plt.suptitle(f'Ground Truth (Blue) vs Predicted (Red) - Shoe {SHOE_ID}',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# ========================================================================
# VISUALIZATION 3: Point Distribution Analysis
# ========================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# X, Y, Z distributions
for idx, (axis_name, axis_idx) in enumerate([('X', 0), ('Y', 1), ('Z', 2)]):
    ax = axes[0, idx]
    ax.hist(gt_vertices[:, axis_idx], bins=50, alpha=0.5, label='GT', color='blue')
    ax.hist(pred_points_np[:, axis_idx], bins=50, alpha=0.5, label='Pred', color='red')
    ax.set_xlabel(f'{axis_name} coordinate')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{axis_name}-axis Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Distance from origin
gt_dist = np.linalg.norm(gt_vertices, axis=1)
pred_dist = np.linalg.norm(pred_points_np, axis=1)

axes[1, 0].hist(gt_dist, bins=50, alpha=0.5, label='GT', color='blue')
axes[1, 0].hist(pred_dist, bins=50, alpha=0.5, label='Pred', color='red')
axes[1, 0].set_xlabel('Distance from origin')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distance Distribution')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Point density comparison
axes[1, 1].text(0.1, 0.8, 'Ground Truth:', fontsize=12, fontweight='bold')
axes[1, 1].text(0.1, 0.7, f'  Points: {len(gt_vertices):,}', fontsize=10)
axes[1, 1].text(0.1, 0.6, f'  Range: [{gt_vertices.min():.3f}, {gt_vertices.max():.3f}]', fontsize=10)
axes[1, 1].text(0.1, 0.5, f'  Mean: {gt_vertices.mean(axis=0)}', fontsize=9)

axes[1, 1].text(0.1, 0.3, 'Predicted:', fontsize=12, fontweight='bold', color='red')
axes[1, 1].text(0.1, 0.2, f'  Points: {len(pred_points_np):,}', fontsize=10)
axes[1, 1].text(0.1, 0.1, f'  Range: [{pred_points_np.min():.3f}, {pred_points_np.max():.3f}]', fontsize=10)
axes[1, 1].text(0.1, 0.0, f'  Mean: {pred_points_np.mean(axis=0)}', fontsize=9)
axes[1, 1].axis('off')
axes[1, 1].set_title('Statistics')

# Normal analysis
normal_magnitudes = np.linalg.norm(pred_normals_np, axis=1)
axes[1, 2].hist(normal_magnitudes, bins=50, color='green', alpha=0.7)
axes[1, 2].set_xlabel('Normal magnitude')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].set_title('Predicted Normal Magnitudes\n(should be ~1.0)')
axes[1, 2].axvline(1.0, color='red', linestyle='--', label='Target (1.0)')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ========================================================================
# Compute Metrics
# ========================================================================

print("\n" + "="*70)
print("DIAGNOSTIC METRICS")
print("="*70)

# Chamfer distance (simplified)
from scipy.spatial import cKDTree

tree_gt = cKDTree(gt_vertices)
tree_pred = cKDTree(pred_points_np)

# Forward: pred -> GT
dist_pred_to_gt, _ = tree_gt.query(pred_points_np)
forward_chamfer = dist_pred_to_gt.mean()

# Backward: GT -> pred
dist_gt_to_pred, _ = tree_pred.query(gt_vertices)
backward_chamfer = dist_gt_to_pred.mean()

chamfer = (forward_chamfer + backward_chamfer) / 2

print(f"\nChamfer Distance:")
print(f"  Forward (pred→GT): {forward_chamfer:.6f}")
print(f"  Backward (GT→pred): {backward_chamfer:.6f}")
print(f"  Total: {chamfer:.6f}")

# Coverage analysis
coverage_threshold = 0.1
coverage = (dist_gt_to_pred < coverage_threshold).mean()
print(f"\nCoverage Analysis:")
print(f"  Threshold: {coverage_threshold}")
print(f"  Coverage: {coverage*100:.2f}% of GT points within threshold")

# Point distribution
print(f"\nPoint Distribution:")
print(f"  GT centroid: {gt_vertices.mean(axis=0)}")
print(f"  Pred centroid: {pred_points_np.mean(axis=0)}")
print(f"  Centroid shift: {np.linalg.norm(gt_vertices.mean(axis=0) - pred_points_np.mean(axis=0)):.6f}")

print(f"\n  GT std: {gt_vertices.std(axis=0)}")
print(f"  Pred std: {pred_points_np.std(axis=0)}")

print(f"\n  GT range: [{gt_vertices.min():.3f}, {gt_vertices.max():.3f}]")
print(f"  Pred range: [{pred_points_np.min():.3f}, {pred_points_np.max():.3f}]")

# Check for collapse
pred_spread = pred_points_np.std()
gt_spread = gt_vertices.std()
print(f"\nCollapse Detection:")
print(f"  GT spread (std): {gt_spread:.6f}")
print(f"  Pred spread (std): {pred_spread:.6f}")
print(f"  Spread ratio (pred/GT): {pred_spread/gt_spread:.3f}")
if pred_spread / gt_spread < 0.5:
    print(f"  ⚠️  WARNING: Points may be collapsing!")

# Normal quality
print(f"\nNormal Quality:")
print(f"  Mean magnitude: {normal_magnitudes.mean():.6f} (should be ~1.0)")
print(f"  Std magnitude: {normal_magnitudes.std():.6f}")

print("="*70)

print("\n✓ Diagnostic complete!")
print("\nWhat to look for:")
print("  1. Are predicted points forming a shoe shape or just a blob?")
print("  2. Is the point spread similar to GT? (spread ratio should be ~1.0)")
print("  3. Is coverage low? (<50% means points aren't covering the surface)")
print("  4. Are points collapsing to the center? (check centroid and spread)")
