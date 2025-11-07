"""
Smoke Test for CURRENT Point Cloud Training Pipeline
Tests the actual geometry model and training you're using
"""

import torch
import sys
import traceback
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


def test_imports():
    """Test: All imports work"""
    print("="*70)
    print("TEST 1: IMPORTS")
    print("="*70)

    try:
        print("  PyTorch...", end=" ")
        import torch
        print(f"✓ (v{torch.__version__})")

        print("  Transformers...", end=" ")
        from transformers import AutoModel
        print("✓")

        print("  Trimesh...", end=" ")
        import trimesh
        print("✓")

        print("  Project modules...", end=" ")
        from config import config
        from models.geometry_model import GeometryModel, geometry_loss
        from train_geometry import ImprovedGeometryTrainer
        print("✓")

        print("\n✓ All imports successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Import failed: {e}\n")
        traceback.print_exc()
        return False


def test_config():
    """Test: Config loads correctly"""
    print("="*70)
    print("TEST 2: CONFIGURATION")
    print("="*70)

    try:
        from config import config

        print(f"  Device: {config.device}")
        print(f"  Batch size: {config.batch_size_stage1}")
        print(f"  Num points: {config.num_points}")
        print(f"  Learning rate: {config.learning_rate_stage1}")
        print(f"  Epochs: {config.num_epochs_stage1}")

        # Check data paths
        obj_dir = Path(config.obj_dir)
        images_dir = Path(config.images_dir)

        print(f"\n  Data paths:")
        print(f"    OBJ dir: {obj_dir} ({'exists' if obj_dir.exists() else 'NOT FOUND'})")
        print(f"    Images dir: {images_dir} ({'exists' if images_dir.exists() else 'NOT FOUND'})")

        print("\n✓ Configuration OK\n")
        return True

    except Exception as e:
        print(f"\n✗ Config failed: {e}\n")
        traceback.print_exc()
        return False


def test_geometry_model():
    """Test: Geometry model initializes and runs"""
    print("="*70)
    print("TEST 3: GEOMETRY MODEL")
    print("="*70)

    try:
        from config import config
        from models.geometry_model import GeometryModel

        print(f"  Initializing model...")
        model = GeometryModel(
            num_points=config.num_points,
            freeze_encoder=True,
            hidden_dim=config.hidden_dim
        ).to(config.device)

        total_params, trainable_params = model.count_parameters()
        print(f"  ✓ Model created")
        print(f"    Total params: {total_params:,}")
        print(f"    Trainable: {trainable_params:,}")
        print(f"    Points: {config.num_points}")

        # Test forward pass
        print(f"\n  Testing forward pass...")
        model.eval()

        views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        images = {
            view: torch.randn(1, 3, 224, 224).to(config.device)
            for view in views
        }

        with torch.no_grad():
            points, normals = model(images)

        print(f"  ✓ Forward pass successful")
        print(f"    Points shape: {points.shape}")
        print(f"    Normals shape: {normals.shape}")
        print(f"    Points range: [{points.min().item():.3f}, {points.max().item():.3f}]")

        print("\n✓ Geometry model working\n")
        return True

    except Exception as e:
        print(f"\n✗ Geometry model failed: {e}\n")
        traceback.print_exc()
        return False


def test_loss_function():
    """Test: Loss function computes correctly"""
    print("="*70)
    print("TEST 4: LOSS FUNCTION")
    print("="*70)

    try:
        from config import config
        from models.geometry_model import geometry_loss

        print(f"  Creating dummy data...")
        pred_points = torch.randn(8192, 3).to(config.device)
        pred_normals = torch.randn(8192, 3).to(config.device)
        gt_points = torch.randn(8192, 3).to(config.device)

        print(f"  Computing loss...")
        loss, loss_dict = geometry_loss(
            pred_points,
            pred_normals,
            gt_points,
            lambda_chamfer=config.lambda_chamfer,
            lambda_normal=config.lambda_normal,
            lambda_edge=config.lambda_edge,
            lambda_smooth=config.lambda_smooth,
            lambda_coverage=config.lambda_coverage
        )

        print(f"  ✓ Loss computed")
        print(f"    Total loss: {loss.item():.6f}")
        print(f"    Components:")
        for key, value in loss_dict.items():
            print(f"      {key}: {value:.6f}")

        print("\n✓ Loss function working\n")
        return True

    except Exception as e:
        print(f"\n✗ Loss function failed: {e}\n")
        traceback.print_exc()
        return False


def test_training_step():
    """Test: Training step executes"""
    print("="*70)
    print("TEST 5: TRAINING STEP")
    print("="*70)

    try:
        from config import config
        from models.geometry_model import GeometryModel, geometry_loss
        import torch.optim as optim

        print(f"  Creating model...")
        model = GeometryModel(
            num_points=2048,  # Smaller for quick test
            freeze_encoder=True,
            hidden_dim=512
        ).to(config.device)
        model.train()

        print(f"  Creating optimizer...")
        optimizer = optim.AdamW(
            model.get_trainable_parameters(),
            lr=config.learning_rate_stage1
        )

        print(f"  Creating batch...")
        views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        images = {
            view: torch.randn(1, 3, 224, 224).to(config.device)
            for view in views
        }
        gt_vertices = torch.randn(2048, 3).to(config.device)

        print(f"  Running training step...")

        # Forward
        pred_points, pred_normals = model(images)

        # Loss
        loss, loss_dict = geometry_loss(
            pred_points[0],
            pred_normals[0],
            gt_vertices,
            lambda_chamfer=config.lambda_chamfer,
            lambda_normal=config.lambda_normal,
            lambda_edge=config.lambda_edge,
            lambda_smooth=config.lambda_smooth,
            lambda_coverage=config.lambda_coverage
        )

        print(f"    Loss: {loss.item():.6f}")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.get_trainable_parameters(), 1.0)
        print(f"    Grad norm: {grad_norm:.4f}")

        # Step
        optimizer.step()

        print("\n✓ Training step successful\n")
        return True

    except Exception as e:
        print(f"\n✗ Training step failed: {e}\n")
        traceback.print_exc()
        return False


def test_dataset():
    """Test: Dataset loads (if data exists)"""
    print("="*70)
    print("TEST 6: DATASET")
    print("="*70)

    try:
        from config import config
        from load_data import ShoeDataset
        from pathlib import Path

        obj_dir = Path(config.obj_dir)
        images_dir = Path(config.images_dir)

        if not obj_dir.exists() or not images_dir.exists():
            print("  ⚠️  Skipping: Data directories not found")
            return True  # Don't fail

        print(f"  Loading dataset...")
        dataset = ShoeDataset(
            obj_dir=config.obj_dir,
            images_dir=config.images_dir,
            verify_mappings=True,
            image_size=config.image_size
        )

        print(f"  ✓ Dataset loaded")
        print(f"    Size: {len(dataset)} shoes")

        if len(dataset) > 0:
            print(f"\n  Testing sample...")
            sample = dataset[0]
            print(f"    Shoe ID: {sample['shoe_id']}")
            print(f"    Vertices: {sample['vertices'].shape}")
            print(f"    Faces: {sample['faces'].shape}")
            print(f"    Images: {list(sample['images'].keys())}")

        print("\n✓ Dataset working\n")
        return True

    except Exception as e:
        print(f"\n✗ Dataset failed: {e}\n")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "#"*70)
    print("# CURRENT TRAINING PIPELINE SMOKE TEST")
    print("# Point Cloud Geometry Model")
    print("#"*70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Geometry Model", test_geometry_model),
        ("Loss Function", test_loss_function),
        ("Training Step", test_training_step),
        ("Dataset", test_dataset),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}\n")
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("="*70)
    print("SUMMARY")
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
        print("\n🎉 ALL TESTS PASSED! Ready to train!")
        print("="*70 + "\n")
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED")
        print("="*70 + "\n")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
