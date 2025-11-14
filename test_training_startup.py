"""
UPDATED Smoke Test - Compatible with Your load_data.py
Handles both ShoeDataset and MultiViewMeshDataset naming
"""

import sys
import traceback
from pathlib import Path
import warnings

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# 1. Import tests
# ---------------------------------------------------------------------------

def test_imports():
    print("=" * 70)
    print("TEST 1: Imports")
    print("=" * 70)
    try:
        from config import config
        
        # Try both naming conventions
        try:
            from load_data import ShoeDataset, custom_collate_fn
            dataset_class = ShoeDataset
            collate = custom_collate_fn
            print("  ✓ config imported")
            print("  ✓ ShoeDataset & custom_collate_fn imported")
        except ImportError:
            # Fall back to alternate names
            from load_data import MultiViewMeshDataset, collate_fn
            dataset_class = MultiViewMeshDataset
            collate = collate_fn
            print("  ✓ config imported")
            print("  ✓ MultiViewMeshDataset & collate_fn imported")
            print("  ℹ️  Using alternate names (add aliases to load_data.py)")
        
        # Try to import the model
        try:
            from models.geometry_model import ImprovedGeometryModel, multi_scale_geometry_loss
            print("  ✓ ImprovedGeometryModel & multi_scale_geometry_loss imported")
            print("  ✓ Using NEW multi-scale architecture")
            is_new = True
        except ImportError:
            from models.geometry_model import GeometryModel, geometry_loss
            print("  ✓ GeometryModel & geometry_loss imported")
            print("  ⚠️  Using OLD single-scale architecture (will fragment!)")
            is_new = False
            
        print()
        return True, is_new, dataset_class, collate
            
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        traceback.print_exc()
        print()
        return False, False, None, None


# ---------------------------------------------------------------------------
# 2. Config sanity check
# ---------------------------------------------------------------------------

def test_config():
    print("=" * 70)
    print("TEST 2: Config Sanity Check")
    print("=" * 70)
    try:
        from config import config

        print(f"  Device: {config.device}")
        print(f"  OBJ dir:    {config.obj_dir}")
        print(f"  Images dir: {config.images_dir}")
        print(f"  Batch size (stage1): {config.batch_size_stage1}")
        print(f"  Num points:          {getattr(config, 'num_points', 'N/A')}")
        print(f"  LR stage1:           {config.learning_rate_stage1}")
        
        # Check for critical anti-fragmentation settings
        print(f"\n  Anti-fragmentation settings:")
        print(f"    Edge regularization:   {config.lambda_edge}")
        print(f"    Smooth regularization: {config.lambda_smooth}")
        print(f"    Gradient clipping:     {getattr(config, 'grad_clip_norm', 'NOT SET')}")
        
        # Validate settings
        warnings = []
        if config.learning_rate_stage1 > 5e-5:
            warnings.append(f"LR too high ({config.learning_rate_stage1}) - should be ≤1e-5")
        else:
            print(f"  ✓ Learning rate is appropriately low")
            
        if config.lambda_edge < 1.0:
            warnings.append(f"Edge regularization weak ({config.lambda_edge}) - should be ≥2.0")
        else:
            print(f"  ✓ Edge regularization is strong")
            
        if config.lambda_smooth < 0.5:
            warnings.append(f"Smooth regularization weak ({config.lambda_smooth}) - should be ≥1.0")
        else:
            print(f"  ✓ Smooth regularization is strong")

        if warnings:
            print(f"\n  ⚠️  CONFIG WARNINGS:")
            for w in warnings:
                print(f"    - {w}")

        obj_dir = Path(config.obj_dir)
        images_dir = Path(config.images_dir)

        print("\n  Path existence:")
        print(f"    OBJ dir exists?    {obj_dir.exists()}")
        print(f"    Images dir exists? {images_dir.exists()}")

        print("\n  ✓ Config object looks OK\n")
        return True
    except Exception as e:
        print(f"  ✗ Config test failed: {e}")
        traceback.print_exc()
        print()
        return False


# ---------------------------------------------------------------------------
# 3. Dataset + DataLoader test
# ---------------------------------------------------------------------------

def test_dataset_and_loader(dataset_class, collate):
    print("=" * 70)
    print("TEST 3: Dataset & DataLoader")
    print("=" * 70)
    try:
        from config import config

        # Prepare kwargs for dataset initialization
        dataset_kwargs = {
            'obj_dir': config.obj_dir,
            'images_dir': config.images_dir,
            'image_size': config.image_size,
        }
        
        # Add views if dataset expects it
        if hasattr(config, 'views'):
            dataset_kwargs['views'] = config.views
        else:
            # Assume standard views
            dataset_kwargs['views'] = ['front', 'back', 'left', 'right', 'top', 'bottom']

        dataset = dataset_class(**dataset_kwargs)

        print(f"  Dataset size: {len(dataset)}")

        if len(dataset) == 0:
            print("  ⚠️  Dataset is EMPTY – check your paths.")
            return False

        loader = DataLoader(
            dataset,
            batch_size=config.batch_size_stage1,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate,
        )

        batch = next(iter(loader))

        print("  Batch keys:", list(batch.keys()))
        print("  Views:", list(batch["images"].keys()))
        for view, imgs in batch["images"].items():
            print(f"    {view}: {tuple(imgs.shape)}  (B, C, H, W)")

        print(f"  First sample vertices: {batch['vertices'][0].shape}")
        print("  ✓ Dataset & loader work\n")
        return True
    except StopIteration:
        print("  ✗ DataLoader returned no batches (empty dataset).")
        print()
        return False
    except Exception as e:
        print(f"  ✗ Dataset/DataLoader test failed: {e}")
        traceback.print_exc()
        print()
        return False


# ---------------------------------------------------------------------------
# 4. Model forward pass
# ---------------------------------------------------------------------------

def test_model_forward(is_new_architecture, dataset_class, collate):
    print("=" * 70)
    print("TEST 4: Model Forward Pass")
    print("=" * 70)
    try:
        from config import config

        device = torch.device(config.device)

        # Prepare dataset
        dataset_kwargs = {
            'obj_dir': config.obj_dir,
            'images_dir': config.images_dir,
            'image_size': config.image_size,
        }
        
        if hasattr(config, 'views'):
            dataset_kwargs['views'] = config.views
        else:
            dataset_kwargs['views'] = ['front', 'back', 'left', 'right', 'top', 'bottom']

        dataset = dataset_class(**dataset_kwargs)
        
        if len(dataset) == 0:
            print("  ✗ Cannot run forward pass: dataset is empty.")
            return False

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate,
        )
        batch = next(iter(loader))

        # Move images to device
        images = {k: v.to(device) for k, v in batch["images"].items()}

        if is_new_architecture:
            # New multi-scale model
            from models.geometry_model import ImprovedGeometryModel
            
            model = ImprovedGeometryModel(
                freeze_encoder=config.freeze_image_encoder,
                hidden_dim=config.hidden_dim,
            ).to(device)
            model.eval()

            with torch.no_grad():
                pred_points, pred_normals = model(images, return_all_levels=False)
                print(f"  pred_points (fine) shape: {tuple(pred_points.shape)}   (B, N, 3)")
                print(f"  pred_normals shape: {tuple(pred_normals.shape)} (B, N, 3)")
                
                # Test multi-level
                mesh_outputs = model(images, return_all_levels=True)
                print(f"\n  Multi-scale outputs:")
                print(f"    Coarse: {tuple(mesh_outputs['coarse'].shape)} vertices")
                print(f"    Mid:    {tuple(mesh_outputs['mid'].shape)} vertices")
                print(f"    Fine:   {tuple(mesh_outputs['fine'].shape)} vertices")
                
        else:
            # Old single-scale model
            from models.geometry_model import GeometryModel
            
            model = GeometryModel(
                num_points=config.num_points,
                freeze_encoder=config.freeze_image_encoder,
                hidden_dim=config.hidden_dim,
            ).to(device)
            model.eval()

            with torch.no_grad():
                pred_points, pred_normals = model(images)
                print(f"  pred_points shape: {tuple(pred_points.shape)}   (B, N, 3)")
                print(f"  pred_normals shape: {tuple(pred_normals.shape)} (B, N, 3)")

        print("  ✓ Forward pass succeeded\n")
        return True
    except Exception as e:
        print(f"  ✗ Model forward test failed: {e}")
        traceback.print_exc()
        print()
        return False


# ---------------------------------------------------------------------------
# 5. Single training step
# ---------------------------------------------------------------------------

def test_training_step(is_new_architecture, dataset_class, collate):
    print("=" * 70)
    print("TEST 5: Single Training Step")
    print("=" * 70)
    try:
        from config import config

        device = torch.device(config.device)

        # Prepare dataset
        dataset_kwargs = {
            'obj_dir': config.obj_dir,
            'images_dir': config.images_dir,
            'image_size': config.image_size,
        }
        
        if hasattr(config, 'views'):
            dataset_kwargs['views'] = config.views
        else:
            dataset_kwargs['views'] = ['front', 'back', 'left', 'right', 'top', 'bottom']

        dataset = dataset_class(**dataset_kwargs)
        
        if len(dataset) == 0:
            print("  ✗ Cannot run training step: dataset is empty.")
            return False

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate,
        )
        batch = next(iter(loader))

        images = {k: v.to(device) for k, v in batch["images"].items()}
        gt_vertices = batch["vertices"][0].to(device)

        if is_new_architecture:
            from models.geometry_model import ImprovedGeometryModel, multi_scale_geometry_loss
            
            model = ImprovedGeometryModel(
                freeze_encoder=config.freeze_image_encoder,
                hidden_dim=config.hidden_dim,
            ).to(device)
            model.train()

            optimizer = optim.AdamW(
                model.get_trainable_parameters(),
                lr=config.learning_rate_stage1,
                weight_decay=config.weight_decay_stage1,
            )

            mesh_outputs = model(images, return_all_levels=True)

            if gt_vertices.shape[0] > 2562:
                idx = torch.randperm(gt_vertices.shape[0], device=device)[:2562]
                gt_sample = gt_vertices[idx]
            else:
                idx = torch.randint(0, gt_vertices.shape[0], (2562,), device=device)
                gt_sample = gt_vertices[idx]

            loss, loss_dict = multi_scale_geometry_loss(
                mesh_outputs,
                gt_sample,
                lambda_chamfer=config.lambda_chamfer,
                lambda_edge=config.lambda_edge,
                lambda_smooth=config.lambda_smooth,
                lambda_normal=config.lambda_normal,
            )

            print(f"  Loss value: {loss.item():.6f}")
            print(f"    Components:")
            for k, v in loss_dict.items():
                print(f"      {k}: {v:.6f}")
                
        else:
            from models.geometry_model import GeometryModel, geometry_loss
            
            model = GeometryModel(
                num_points=config.num_points,
                freeze_encoder=config.freeze_image_encoder,
                hidden_dim=config.hidden_dim,
            ).to(device)
            model.train()

            optimizer = optim.AdamW(
                model.get_trainable_parameters(),
                lr=config.learning_rate_stage1,
                weight_decay=config.weight_decay_stage1,
            )

            pred_points, pred_normals = model(images)

            if gt_vertices.shape[0] > config.num_points:
                idx = torch.randperm(gt_vertices.shape[0], device=device)[:config.num_points]
                gt_sample = gt_vertices[idx]
            else:
                gt_sample = gt_vertices

            loss, loss_dict = geometry_loss(
                pred_points[0],
                pred_normals[0],
                gt_sample,
                lambda_chamfer=config.lambda_chamfer,
                lambda_normal=config.lambda_normal,
                lambda_edge=config.lambda_edge,
                lambda_smooth=config.lambda_smooth,
                lambda_coverage=getattr(config, 'lambda_coverage', 0.1),
            )

            print(f"  Loss value: {loss.item():.6f}")
            print(f"    Components: {loss_dict}")

        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        grad_clip = getattr(config, "grad_clip_norm", None)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_parameters(), grad_clip
            )
            print(f"  ✓ Gradients clipped at {grad_clip}")
            
        optimizer.step()

        print("  ✓ Backward + optimizer step succeeded\n")
        return True
    except Exception as e:
        print(f"  ✗ Training step test failed: {e}")
        traceback.print_exc()
        print()
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 70)
    print("MULTI-SCALE MESH TRAINING - SMOKE TEST")
    print("=" * 70 + "\n")
    
    # Test imports
    import_success, is_new_arch, dataset_class, collate = test_imports()
    if not import_success:
        print("\n❌ CRITICAL: Import test failed.")
        print("\n🔧 POSSIBLE FIX:")
        print("  Add these lines to the END of your load_data.py:")
        print("    ShoeDataset = MultiViewMeshDataset")
        print("    custom_collate_fn = collate_fn")
        return 1
    
    print()
    
    results = {
        "imports": import_success,
        "config": test_config(),
        "dataset": test_dataset_and_loader(dataset_class, collate),
        "forward": test_model_forward(is_new_arch, dataset_class, collate),
        "train_step": test_training_step(is_new_arch, dataset_class, collate),
    }

    print("=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")
    print("=" * 70)
    
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED!")
        print("✓ Pipeline is correctly configured")
        print("✓ Ready for training")
        if is_new_arch:
            print("✓ Using multi-scale architecture (fragmentation fix active)")
        else:
            print("⚠️  Using old architecture - WILL FRAGMENT!")
            print("   Replace models/geometry_model.py with geometry_model_fixed.py")
        print("\nRun: python train_geometry.py")
        print("=" * 70)
        return 0
    else:
        print("\n⚠️  SOME TESTS FAILED - see output above")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())