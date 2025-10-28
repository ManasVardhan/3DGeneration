"""
Inference Script: Generate 3D textured meshes from multi-view images
"""

import torch
import trimesh
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms

# Import models from training script
from train import GeometryModel, TextureModel, Config


class MeshGenerator:
    """Generate 3D meshes from multi-view images"""
    
    def __init__(self, geometry_checkpoint, texture_checkpoint, config):
        """
        Args:
            geometry_checkpoint: Path to trained geometry model
            texture_checkpoint: Path to trained texture model
            config: Config object
        """
        self.config = config
        self.device = config.device
        
        # Load models
        print("Loading models...")
        self.geometry_model = GeometryModel(
            feature_dim=config.feature_dim,
            num_points=2048
        ).to(self.device)
        
        self.texture_model = TextureModel(
            feature_dim=config.feature_dim,
            num_points=2048
        ).to(self.device)
        
        # Load weights
        self.geometry_model.load_state_dict(torch.load(geometry_checkpoint, map_location=self.device))
        self.texture_model.load_state_dict(torch.load(texture_checkpoint, map_location=self.device))
        
        self.geometry_model.eval()
        self.texture_model.eval()
        
        print("✓ Models loaded successfully")
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
        ])
    
    def load_multi_view_images(self, image_dir, shoe_id):
        """
        Load 6 view images for a shoe
        
        Args:
            image_dir: Directory containing images
            shoe_id: Shoe ID (e.g., "17")
        
        Returns:
            images_dict: Dict of tensors (1, 3, H, W)
            angles: Tensor (1, 6, 2)
        """
        views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        view_angles = {
            'front': (0, 0),
            'back': (180, 0),
            'left': (90, 0),
            'right': (270, 0),
            'top': (0, 90),
            'bottom': (0, -90)
        }
        
        images_dict = {}
        angles_list = []
        
        for view in views:
            # Try different extensions
            img_path = None
            for ext in ['.png', '.jpeg', '.jpg']:
                test_path = Path(image_dir) / f"{shoe_id}_{view}{ext}"
                if test_path.exists():
                    img_path = test_path
                    break
            
            if img_path is None:
                raise FileNotFoundError(f"Image not found for shoe {shoe_id}, view {view}")
            
            # Load and transform
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0)  # (1, 3, H, W)
            images_dict[view] = img_tensor.to(self.device)
            
            angles_list.append(view_angles[view])
        
        angles = torch.tensor(angles_list, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        return images_dict, angles
    
    def generate_mesh(self, images_dict, angles):
        """
        Generate 3D mesh from multi-view images
        
        Args:
            images_dict: Dict of image tensors
            angles: View angles tensor
        
        Returns:
            vertices: (N, 3) numpy array
            colors: (N, 3) numpy array
        """
        with torch.no_grad():
            # Stage 1: Predict geometry
            points = self.geometry_model(images_dict, angles)  # (1, N, 3)
            
            # Stage 2: Predict colors
            colors = self.texture_model(points, images_dict)  # (1, N, 3)
            
            # Convert to numpy
            vertices = points[0].cpu().numpy()
            colors = colors[0].cpu().numpy()
        
        return vertices, colors
    
    def save_point_cloud(self, vertices, colors, output_path):
        """Save as point cloud (OBJ or PLY)"""
        # Scale colors to 0-255
        colors_255 = (colors * 255).astype(np.uint8)
        
        # Create point cloud
        cloud = trimesh.PointCloud(vertices, colors=colors_255)
        
        # Save
        cloud.export(output_path)
        print(f"✓ Point cloud saved: {output_path}")
    
    def save_mesh_with_poisson(self, vertices, colors, output_path):
        """
        Save as mesh using Poisson surface reconstruction
        Requires Open3D
        """
        try:
            import open3d as o3d
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # Poisson surface reconstruction
            print("  Running Poisson surface reconstruction...")
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )
            
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Save
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            print(f"✓ Mesh saved: {output_path}")
            
        except ImportError:
            print("⚠️  Open3D not installed. Saving as point cloud instead.")
            self.save_point_cloud(vertices, colors, output_path)


# ============================================================================
# Usage Examples
# ============================================================================

def generate_single_shoe(shoe_id, image_dir, output_dir):
    """Generate mesh for a single shoe"""
    
    # Configuration
    config = Config()
    
    # Paths to trained models
    geometry_checkpoint = "checkpoints/stage1_best.pth"
    texture_checkpoint = "checkpoints/stage2_best.pth"
    
    # Initialize generator
    generator = MeshGenerator(geometry_checkpoint, texture_checkpoint, config)
    
    # Load images
    print(f"\nGenerating mesh for shoe ID: {shoe_id}")
    images_dict, angles = generator.load_multi_view_images(image_dir, shoe_id)
    print("✓ Images loaded")
    
    # Generate mesh
    print("Generating 3D mesh...")
    vertices, colors = generator.generate_mesh(images_dict, angles)
    print(f"✓ Generated {len(vertices)} points")
    
    # Save output
    Path(output_dir).mkdir(exist_ok=True)
    
    # Save as point cloud
    output_path = Path(output_dir) / f"shoe_{shoe_id}_pointcloud.ply"
    generator.save_point_cloud(vertices, colors, output_path)
    
    # Try to save as