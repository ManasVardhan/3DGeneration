"""
Test script to verify permutation invariance of the geometry and texture models.

This script ensures that the model outputs are identical regardless of the order
in which the 6 multi-view images are provided.
"""

import torch
import random
import numpy as np
from models.geometry_model import GeometryModel
from models.texture_model import VertexColorPredictor


def test_geometry_model_permutation_invariance():
    """
    Test that GeometryModel produces identical outputs for permuted inputs.
    """
    print("="*70)
    print("TESTING GEOMETRY MODEL PERMUTATION INVARIANCE")
    print("="*70)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model
    model = GeometryModel(num_points=1024, freeze_encoder=True)
    model.eval()

    # Create dummy input data
    batch_size = 2
    img_size = 224
    views = ['front', 'back', 'left', 'right', 'top', 'bottom']

    # Create random images
    images_dict = {
        view: torch.randn(batch_size, 3, img_size, img_size)
        for view in views
    }

    # Test 1: Original order
    print("\n1. Testing original order...")
    with torch.no_grad():
        output_original = model(images_dict)
    print(f"   Output shape: {output_original.shape}")
    print(f"   Output mean: {output_original.mean().item():.6f}")

    # Test 2: Random permutation
    print("\n2. Testing random permutation...")
    permuted_views = random.sample(views, len(views))
    print(f"   Permuted order: {permuted_views}")

    # Create permuted inputs - swap the actual image data
    images_list_permuted = [images_dict[permuted_views[i]] for i in range(len(views))]
    images_dict_permuted = {views[i]: images_list_permuted[i] for i in range(len(views))}

    with torch.no_grad():
        output_permuted = model(images_dict_permuted)
    print(f"   Output shape: {output_permuted.shape}")
    print(f"   Output mean: {output_permuted.mean().item():.6f}")

    # Compare outputs
    print("\n3. Comparing outputs...")
    max_diff = torch.max(torch.abs(output_original - output_permuted)).item()
    mean_diff = torch.mean(torch.abs(output_original - output_permuted)).item()

    print(f"   Max absolute difference: {max_diff:.8f}")
    print(f"   Mean absolute difference: {mean_diff:.8f}")

    # Check if permutation invariant (allowing small numerical errors)
    tolerance = 1e-5
    is_invariant = max_diff < tolerance

    if is_invariant:
        print(f"\n✓ SUCCESS: Model is permutation invariant (max diff < {tolerance})")
    else:
        print(f"\n✗ FAILURE: Model is NOT permutation invariant (max diff = {max_diff})")

    return is_invariant


def test_texture_model_permutation_invariance():
    """
    Test that VertexColorPredictor produces identical outputs for permuted inputs.
    """
    print("\n" + "="*70)
    print("TESTING TEXTURE MODEL PERMUTATION INVARIANCE")
    print("="*70)

    # Set random seed
    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model
    model = VertexColorPredictor(use_all_views=True)
    model.eval()

    # Create dummy input data
    batch_size = 1  # Texture model processes single meshes
    img_size = 224
    num_vertices = 100
    views = ['front', 'back', 'left', 'right', 'top', 'bottom']

    # Create random images
    images_dict = {
        view: torch.randn(batch_size, 3, img_size, img_size)
        for view in views
    }

    # Create random vertices
    vertices = torch.randn(num_vertices, 3)

    # Test 1: Original order
    print("\n1. Testing original order...")
    with torch.no_grad():
        output_original = model(vertices, images_dict)
    print(f"   Output shape: {output_original.shape}")
    print(f"   Output mean: {output_original.mean().item():.6f}")

    # Test 2: Random permutation
    print("\n2. Testing random permutation...")
    permuted_views = random.sample(views, len(views))
    print(f"   Permuted order: {permuted_views}")

    # Create permuted inputs - swap the actual image data
    images_list_permuted = [images_dict[permuted_views[i]] for i in range(len(views))]
    images_dict_permuted = {views[i]: images_list_permuted[i] for i in range(len(views))}

    with torch.no_grad():
        output_permuted = model(vertices, images_dict_permuted)
    print(f"   Output shape: {output_permuted.shape}")
    print(f"   Output mean: {output_permuted.mean().item():.6f}")

    # Compare outputs
    print("\n3. Comparing outputs...")
    max_diff = torch.max(torch.abs(output_original - output_permuted)).item()
    mean_diff = torch.mean(torch.abs(output_original - output_permuted)).item()

    print(f"   Max absolute difference: {max_diff:.8f}")
    print(f"   Mean absolute difference: {mean_diff:.8f}")

    # Check if permutation invariant
    tolerance = 1e-5
    is_invariant = max_diff < tolerance

    if is_invariant:
        print(f"\n✓ SUCCESS: Model is permutation invariant (max diff < {tolerance})")
    else:
        print(f"\n✗ FAILURE: Model is NOT permutation invariant (max diff = {max_diff})")

    return is_invariant


def test_multiple_permutations():
    """
    Test multiple random permutations to ensure robustness.
    """
    print("\n" + "="*70)
    print("TESTING MULTIPLE RANDOM PERMUTATIONS")
    print("="*70)

    torch.manual_seed(42)
    np.random.seed(42)

    # Initialize model
    model = GeometryModel(num_points=512, freeze_encoder=True)
    model.eval()

    # Create dummy data
    batch_size = 1
    img_size = 224
    views = ['front', 'back', 'left', 'right', 'top', 'bottom']

    images_dict = {
        view: torch.randn(batch_size, 3, img_size, img_size)
        for view in views
    }

    # Get reference output
    with torch.no_grad():
        reference_output = model(images_dict)

    # Test 10 random permutations
    num_tests = 10
    all_invariant = True
    tolerance = 1e-5

    print(f"\nTesting {num_tests} random permutations...")

    for i in range(num_tests):
        # Random permutation
        permuted_views = random.sample(views, len(views))

        # Permute images
        images_list_permuted = [images_dict[permuted_views[j]] for j in range(len(views))]
        images_dict_permuted = {views[j]: images_list_permuted[j] for j in range(len(views))}

        with torch.no_grad():
            output_permuted = model(images_dict_permuted)

        max_diff = torch.max(torch.abs(reference_output - output_permuted)).item()

        status = "✓" if max_diff < tolerance else "✗"
        print(f"  {status} Test {i+1}: max_diff = {max_diff:.8f}")

        if max_diff >= tolerance:
            all_invariant = False

    if all_invariant:
        print(f"\n✓ SUCCESS: All {num_tests} permutations produced identical outputs!")
    else:
        print(f"\n✗ FAILURE: Some permutations produced different outputs")

    return all_invariant


if __name__ == '__main__':
    print("\n" + "#"*70)
    print("# PERMUTATION INVARIANCE TEST SUITE")
    print("#"*70)

    # Run all tests
    results = {}

    try:
        results['geometry'] = test_geometry_model_permutation_invariance()
    except Exception as e:
        print(f"\n✗ Geometry model test failed with error: {e}")
        results['geometry'] = False

    try:
        results['texture'] = test_texture_model_permutation_invariance()
    except Exception as e:
        print(f"\n✗ Texture model test failed with error: {e}")
        results['texture'] = False

    try:
        results['multiple'] = test_multiple_permutations()
    except Exception as e:
        print(f"\n✗ Multiple permutations test failed with error: {e}")
        results['multiple'] = False

    # Summary
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"Geometry Model:        {'✓ PASS' if results.get('geometry') else '✗ FAIL'}")
    print(f"Texture Model:         {'✓ PASS' if results.get('texture') else '✗ FAIL'}")
    print(f"Multiple Permutations: {'✓ PASS' if results.get('multiple') else '✗ FAIL'}")

    all_passed = all(results.values())
    if all_passed:
        print("\n" + "🎉 ALL TESTS PASSED! Models are permutation invariant! 🎉")
    else:
        print("\n" + "⚠️  SOME TESTS FAILED. Please review the implementation.")

    print("="*70 + "\n")
