"""
Inference Script
Generate 3D textured meshes from multi-view images
"""

import torch
import trimesh
import numpy as np
import argparse
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

from config import config
from models.geometry_model import GeometryModel
from models.texture_model import VertexColorPredictor


class MeshGenerator:
    """Generate 3D meshes from multi-view images"""
    
    def __init__(self, geometry_checkpoint, texture_checkpoint):
        """
        Args:
            geometry_checkpoint: Path to trained geometry model
            texture_checkpoint: Path to trained texture model
        """
        self.config = config
        self.device = config.device
        
        print("Loading models...")
        
        # Load geometry model
        self.geometry_model = GeometryModel(
            num_points=config.num_points,
            freeze_encoder=True
        ).to(self.device)
        
        self.geometry_model.load_state_dict(
            torch.load(geometry_checkpoint, map_location=self.device)
        )
        self.geometry_model.eval()
        print("✓ Geometry model loaded")
        
        # Load texture model
        self.texture_model = VertexColorPredictor().to(self.device)
        self.texture_model.load_state_dict(
            torch.load(texture_checkpoint, map_location=self.device)
        )
        self.texture_model.eval()
        print("✓ Texture model loaded")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ])
    
    def load_images(self, image_dir, shoe_id):
        """
        Load 6 view images for a shoe

        Args:
            image_dir: Directory containing images
            shoe_id: Shoe ID (e.g., "17")

        Returns:
            images_dict: Dict of tensors
        """
        views = ['front', 'back', 'left', 'right', 'top', 'bottom']

        images_dict = {}

        for view in views:
            # Try different extensions
            img_path = None
            for ext in ['.png', '.jpeg', '.jpg']:
                test_path = Path(image_dir) / f"{shoe_id}_{view}{ext}"
                if test_path.exists():
                    img_path = test_path
                    break

            if img_path is None:
                raise FileNotFoundError(
                    f"Image not found: {image_dir}/{shoe_id}_{view}.[png|jpeg|jpg]"
                )

            # Load and transform
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)  # (1, 3, H, W)
            images_dict[view] = img_tensor.to(self.device)

        return images_dict
    
    def generate(self, images_dict, resolution=256):
        """
        Generate 3D mesh from images

        Args:
            images_dict: Dict of image tensors
            resolution: Mesh resolution (higher = more detailed)

        Returns:
            vertices: (N, 3) numpy array
            faces: (F, 3) numpy array
            colors: (N, 3) numpy array [0, 1]
        """
        with torch.no_grad():
            print("  Predicting geometry...")
            # Stage 1: Predict geometry
            points = self.geometry_model(images_dict)
            vertices, faces = self.geometry_model.extract_mesh(points)

            print(f"  Generated mesh: {vertices.shape[0]} vertices, {faces.shape[0]} faces")

            print("  Predicting colors...")
            # Stage 2: Predict colors
            vertices_tensor = torch.from_numpy(vertices).float().to(self.device)
            colors = self.texture_model(vertices_tensor, images_dict)

            # Convert to numpy
            vertices_np = vertices
            faces_np = faces
            colors_np = colors.cpu().numpy()
        
        return vertices_np, faces_np, colors_np
    
    def save_mesh(self, vertices, faces, colors, output_path):
        """Save textured mesh to file"""
        # Scale colors to 0-255
        colors_255 = (colors * 255).astype(np.uint8)
        
        # Create mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            vertex_colors=colors_255
        )
        
        # Save
        mesh.export(output_path)
        print(f"✓ Mesh saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate 3D mesh from multi-view images")
    parser.add_argument('--shoe_id', type=str, required=True, help='Shoe ID (e.g., 17)')
    parser.add_argument('--image_dir', type=str, default=None, help='Directory with images')
    parser.add_argument('--output', type=str, default=None, help='Output mesh path')
    parser.add_argument('--resolution', type=int, default=256, help='Mesh resolution (128/256/512)')
    parser.add_argument('--geometry_model', type=str, default=None, help='Geometry checkpoint')
    parser.add_argument('--texture_model', type=str, default=None, help='Texture checkpoint')
    
    args = parser.parse_args()
    
    # Default paths
    image_dir = args.image_dir or config.images_dir
    geometry_checkpoint = args.geometry_model or config.geometry_checkpoint
    texture_checkpoint = args.texture_model or config.texture_checkpoint
    output_path = args.output or f"output/shoe_{args.shoe_id}_generated.obj"
    
    # Create output directory
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    
    print("="*70)
    print("3D MESH GENERATION")
    print("="*70)
    print(f"Shoe ID: {args.shoe_id}")
    print(f"Image Dir: {image_dir}")
    print(f"Resolution: {args.resolution}")
    print(f"Output: {output_path}")
    print("="*70 + "\n")
    
    # Check if models exist
    if not Path(geometry_checkpoint).exists():
        print(f"❌ Geometry model not found: {geometry_checkpoint}")
        print("   Please train the model first: python train_geometry.py")
        return
    
    if not Path(texture_checkpoint).exists():
        print(f"❌ Texture model not found: {texture_checkpoint}")
        print("   Please train the model first: python train_texture.py")
        return
    
    # Initialize generator
    generator = MeshGenerator(geometry_checkpoint, texture_checkpoint)
    
    # Load images
    print(f"Loading images for shoe {args.shoe_id}...")
    try:
        images_dict = generator.load_images(image_dir, args.shoe_id)
        print("✓ Images loaded\n")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # Generate mesh
    print("Generating 3D mesh...")
    vertices, faces, colors = generator.generate(images_dict, resolution=args.resolution)
    print("✓ Generation complete\n")
    
    # Save mesh
    print("Saving mesh...")
    generator.save_mesh(vertices, faces, colors, output_path)
    
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Output: {output_path}")
    print("\nMesh statistics:")
    print(f"  Vertices: {len(vertices):,}")
    print(f"  Faces: {len(faces):,}")
    print(f"  Has colors: Yes")
    print("\nView mesh in:")
    print("  - Blender")
    print("  - MeshLab")
    print("  - https://3dviewer.net/")
    print("="*70)


if __name__ == "__main__":
    main()