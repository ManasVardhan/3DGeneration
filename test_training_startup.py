"""
Smoke Test for Training Pipeline
Tests that training starts without errors (for cloud deployment validation)

This test performs a quick sanity check of:
1. Data loading (with minimal data)
2. Model initialization
3. First training step forward/backward pass
4. Checkpoint saving

Run this before deploying to cloud to catch issues early!
"""

import torch
import sys
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """Test 1: All imports work"""
    print("="*70)
    print("TEST 1: IMPORT CHECKS")
    print("="*70)

    try:
        print("  Importing PyTorch...", end=" ")
        import torch
        print(f"✓ (v{torch.__version__})")

        print("  Importing transformers...", end=" ")
        from transformers import AutoModel, AutoImageProcessor
        print("✓")

        print("  Importing project modules...", end=" ")
        from config import config
        from models.geometry_model import GeometryModel, geometry_loss
        from models.texture_model import VertexColorPredictor, texture_loss
        from load_data import ShoeDataset, custom_collate_fn
        print("✓")

        print("\n✓ All imports successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}\n")
        traceback.print_exc()
        return False


def test_config():
    """Test 2: Config loads and paths exist"""
    print("="*70)
    print("TEST 2: CONFIGURATION")
    print("="*70)

    try:
        from config import config

        print(f"  Device: {config.device}")
        print(f"  Batch size (Stage 1): {config.batch_size_stage1}")
        print(f"  Batch size (Stage 2): {config.batch_size_stage2}")
        print(f"  Num points: {config.num_points}")
        print(f"  Freeze encoder: {config.freeze_image_encoder}")

        # Check data paths
        print(f"\n  Checking data paths...")
        obj_dir = Path(config.obj_dir)
        images_dir = Path(config.images_dir)

        print(f"    OBJ dir: {obj_dir}")
        if not obj_dir.exists():
            print(f"    ⚠️  Warning: OBJ directory does not exist!")
        else:
            obj_files = list(obj_dir.glob("*.obj"))
            print(f"    ✓ Found {len(obj_files)} OBJ files")

        print(f"    Images dir: {images_dir}")
        if not images_dir.exists():
            print(f"    ⚠️  Warning: Images directory does not exist!")
        else:
            image_files = list(images_dir.glob("*_front.*"))
            print(f"    ✓ Found {len(image_files)} front view images")

        print("\n✓ Configuration loaded\n")
        return True

    except Exception as e:
        print(f"\n✗ Config failed: {e}\n")
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test 3: Dataset loads without errors"""
    print("="*70)
    print("TEST 3: DATASET LOADING")
    print("="*70)

    try:
        from config import config
        from load_data import ShoeDataset, custom_collate_fn
        from torch.utils.data import DataLoader

        print("  Creating dataset...")
        dataset = ShoeDataset(
            obj_dir=config.obj_dir,
            images_dir=config.images_dir,
            verify_mappings=True,
            image_size=224  # Use smaller size for testing (faster)
        )
        print(f"  ✓ Dataset created: {len(dataset)} samples")

        if len(dataset) == 0:
            print("  ⚠️  Warning: Dataset is empty!")
            return False

        print(f"\n  Loading first sample...")
        sample = dataset[0]
        print(f"    Images: {list(sample['images'].keys())}")
        print(f"    Vertices shape: {sample['vertices'].shape}")
        print(f"    Faces shape: {sample['faces'].shape}")
        print(f"    Colors shape: {sample['vertex_colors'].shape}")
        print(f"    Angles shape: {sample['angles'].shape}")
        print(f"    Shoe ID: {sample['shoe_id']}")

        print(f"\n  Creating dataloader...")
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        print(f"  ✓ Dataloader created: {len(dataloader)} batches")

        print(f"\n  Testing batch iteration...")
        batch = next(iter(dataloader))
        print(f"    Batch keys: {list(batch.keys())}")
        print(f"    Images keys: {list(batch['images'].keys())}")
        for view in ['front', 'back', 'left', 'right', 'top', 'bottom']:
            print(f"      {view}: {batch['images'][view].shape}")

        print("\n✓ Dataset loading successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Dataset loading failed: {e}\n")
        traceback.print_exc()
        return False


def test_geometry_model_init():
    """Test 4: Geometry model initializes"""
    print("="*70)
    print("TEST 4: GEOMETRY MODEL INITIALIZATION")
    print("="*70)

    try:
        from config import config
        from models.geometry_model import GeometryModel

        print(f"  Initializing geometry model on {config.device}...")
        print(f"    (This may take a minute to download DINOv2 if first time)")

        model = GeometryModel(
            num_points=config.num_points,
            freeze_encoder=config.freeze_image_encoder,
            hidden_dim=config.hidden_dim
        ).to(config.device)

        total_params, trainable_params = model.count_parameters()
        print(f"\n  ✓ Model initialized")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Trainable parameters: {trainable_params:,}")

        print("\n✓ Geometry model initialization successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Geometry model init failed: {e}\n")
        traceback.print_exc()
        return False


def test_geometry_forward_pass():
    """Test 5: Geometry model forward pass works"""
    print("="*70)
    print("TEST 5: GEOMETRY MODEL FORWARD PASS")
    print("="*70)

    try:
        from config import config
        from models.geometry_model import GeometryModel, geometry_loss

        print(f"  Creating model...")
        model = GeometryModel(
            num_points=1024,  # Use smaller for faster test
            freeze_encoder=True,  # Freeze for faster test
            hidden_dim=512
        ).to(config.device)
        model.eval()

        print(f"  Creating dummy input...")
        batch_size = 1
        img_size = 224
        views = ['front', 'back', 'left', 'right', 'top', 'bottom']

        images_dict = {
            view: torch.randn(batch_size, 3, img_size, img_size).to(config.device)
            for view in views
        }

        print(f"  Running forward pass...")
        with torch.no_grad():
            output = model(images_dict)

        print(f"    Output shape: {output.shape}")
        print(f"    Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")

        print(f"\n  Testing loss computation...")
        gt_points = torch.randn(1024, 3).to(config.device)
        loss, loss_dict = geometry_loss(output[0], gt_points)
        print(f"    Loss: {loss.item():.6f}")
        print(f"    Loss components: {loss_dict}")

        print("\n✓ Forward pass successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}\n")
        traceback.print_exc()
        return False


def test_geometry_training_step():
    """Test 6: Geometry model training step (forward + backward)"""
    print("="*70)
    print("TEST 6: GEOMETRY MODEL TRAINING STEP")
    print("="*70)

    try:
        from config import config
        from models.geometry_model import GeometryModel, geometry_loss
        from load_data import ShoeDataset, custom_collate_fn
        from torch.utils.data import DataLoader
        import torch.optim as optim

        print(f"  Loading dataset...")
        dataset = ShoeDataset(
            obj_dir=config.obj_dir,
            images_dir=config.images_dir,
            verify_mappings=True,
            image_size=224  # Use smaller size for testing (faster, less memory)
        )

        if len(dataset) == 0:
            print("  ⚠️  Skipping: No data available")
            return True

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )

        print(f"  Creating model...")
        model = GeometryModel(
            num_points=512,  # Small for quick test
            freeze_encoder=True,
            hidden_dim=256
        ).to(config.device)
        model.train()

        print(f"  Creating optimizer...")
        optimizer = optim.AdamW(
            model.get_trainable_parameters(),
            lr=1e-4
        )

        print(f"  Getting first batch...")
        batch = next(iter(dataloader))

        print(f"  Running training step...")
        images = {k: v.to(config.device) for k, v in batch['images'].items()}
        angles = batch['angles'].to(config.device)
        gt_vertices = batch['vertices'][0].to(config.device)

        # Sample GT vertices
        num_samples = 512
        num_verts = gt_vertices.shape[0]
        if num_verts >= num_samples:
            indices = torch.randperm(num_verts)[:num_samples]
        else:
            indices = torch.randint(0, num_verts, (num_samples,))
        gt_vertices_sample = gt_vertices[indices]

        # Forward pass
        print(f"    Forward...")
        images_single = {k: v[0:1] for k, v in images.items()}
        pred_points = model(images_single)

        # Loss
        print(f"    Loss computation...")
        loss, loss_dict = geometry_loss(pred_points[0], gt_vertices_sample)
        print(f"      Loss: {loss.item():.6f}")

        # Backward
        print(f"    Backward...")
        optimizer.zero_grad()
        loss.backward()

        # Optimizer step
        print(f"    Optimizer step...")
        torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
        optimizer.step()

        print("\n✓ Training step successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Training step failed: {e}\n")
        traceback.print_exc()
        return False


def test_texture_model_init():
    """Test 7: Texture model initializes"""
    print("="*70)
    print("TEST 7: TEXTURE MODEL INITIALIZATION")
    print("="*70)

    try:
        from config import config
        from models.texture_model import VertexColorPredictor

        print(f"  Initializing texture model...")
        model = VertexColorPredictor(
            feature_dim=config.feature_dim,
            use_all_views=True
        ).to(config.device)

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  ✓ Model initialized")
        print(f"    Trainable parameters: {trainable_params:,}")

        print("\n✓ Texture model initialization successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Texture model init failed: {e}\n")
        traceback.print_exc()
        return False


def test_texture_forward_pass():
    """Test 8: Texture model forward pass"""
    print("="*70)
    print("TEST 8: TEXTURE MODEL FORWARD PASS")
    print("="*70)

    try:
        from config import config
        from models.texture_model import VertexColorPredictor, texture_loss

        print(f"  Creating model...")
        model = VertexColorPredictor(
            feature_dim=768,
            use_all_views=True
        ).to(config.device)
        model.eval()

        print(f"  Creating dummy input...")
        batch_size = 1
        img_size = 224
        num_vertices = 100
        views = ['front', 'back', 'left', 'right', 'top', 'bottom']

        images_dict = {
            view: torch.randn(batch_size, 3, img_size, img_size).to(config.device)
            for view in views
        }
        vertices = torch.randn(num_vertices, 3).to(config.device)

        print(f"  Running forward pass...")
        with torch.no_grad():
            colors = model(vertices, images_dict)

        print(f"    Output shape: {colors.shape}")
        print(f"    Output range: [{colors.min().item():.3f}, {colors.max().item():.3f}]")

        print(f"\n  Testing loss computation...")
        gt_colors = torch.rand(num_vertices, 3).to(config.device)
        loss, loss_dict = texture_loss(colors, gt_colors)
        print(f"    Loss: {loss.item():.6f}")
        print(f"    Loss components: {loss_dict}")

        print("\n✓ Texture forward pass successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Texture forward pass failed: {e}\n")
        traceback.print_exc()
        return False


def test_checkpoint_saving():
    """Test 9: Checkpoint saving works"""
    print("="*70)
    print("TEST 9: CHECKPOINT SAVING")
    print("="*70)

    try:
        from config import config
        from models.geometry_model import GeometryModel
        import torch
        from pathlib import Path

        print(f"  Creating model...")
        model = GeometryModel(
            num_points=512,
            freeze_encoder=True,
            hidden_dim=256
        ).to(config.device)

        print(f"  Creating checkpoint directory...")
        checkpoint_dir = Path("test_checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        print(f"  Saving checkpoint...")
        checkpoint_path = checkpoint_dir / "test_checkpoint.pth"
        torch.save({
            'epoch': 1,
            'model_state_dict': model.state_dict(),
            'loss': 0.1234
        }, checkpoint_path)
        print(f"    ✓ Saved to {checkpoint_path}")

        print(f"  Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        print(f"    ✓ Loaded (epoch: {checkpoint['epoch']}, loss: {checkpoint['loss']})")

        print(f"\n  Cleaning up...")
        checkpoint_path.unlink()
        checkpoint_dir.rmdir()
        print(f"    ✓ Test checkpoint removed")

        print("\n✓ Checkpoint save/load successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Checkpoint save/load failed: {e}\n")
        traceback.print_exc()
        return False


def test_full_training_script_import():
    """Test 10: Training scripts import without errors"""
    print("="*70)
    print("TEST 10: TRAINING SCRIPTS IMPORT")
    print("="*70)

    try:
        print("  Importing train_geometry.py...", end=" ")
        import train_geometry
        print("✓")

        print("  Importing train_texture.py...", end=" ")
        import train_texture
        print("✓")

        print("\n✓ Training scripts import successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Training scripts import failed: {e}\n")
        traceback.print_exc()
        return False


def main():
    """Run all smoke tests"""
    print("\n" + "#"*70)
    print("# TRAINING PIPELINE SMOKE TEST")
    print("# Run this before deploying to cloud!")
    print("#"*70 + "\n")

    results = {}

    # Run all tests
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Dataset Loading", test_dataset_loading),
        ("Geometry Model Init", test_geometry_model_init),
        ("Geometry Forward Pass", test_geometry_forward_pass),
        ("Geometry Training Step", test_geometry_training_step),
        ("Texture Model Init", test_texture_model_init),
        ("Texture Forward Pass", test_texture_forward_pass),
        ("Checkpoint Saving", test_checkpoint_saving),
        ("Training Scripts Import", test_full_training_script_import),
    ]

    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}\n")
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:8} {test_name}")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    print(f"\n  Total: {total} tests")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if all(results.values()):
        print("\n" + "🎉 ALL TESTS PASSED! Safe to deploy to cloud! 🎉")
        print("="*70 + "\n")
        return 0
    else:
        print("\n" + "⚠️  SOME TESTS FAILED. Fix errors before deploying!")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
