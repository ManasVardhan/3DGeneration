import trimesh
import torch
import numpy as np
from pathlib import Path

class OBJToTrainingTarget:
    """
    Convert 3D OBJ files to training targets (Y values) for your model
    """
    
    def __init__(self, normalize=True, max_vertices=10000):
        """
        Args:
            normalize: Whether to center and normalize mesh to [-1, 1]
            max_vertices: Maximum number of vertices (for memory management)
        """
        self.normalize = normalize
        self.max_vertices = max_vertices
    
    def load_and_process(self, obj_path):
        """
        Load OBJ and convert to training target format
        
        Args:
            obj_path: Path to .obj file
            
        Returns:
            dict with keys:
            - 'vertices': torch.Tensor (N, 3) - 3D positions
            - 'faces': torch.Tensor (F, 3) - triangle indices
            - 'vertex_colors': torch.Tensor (N, 3) - RGB colors [0, 1]
            - 'vertex_normals': torch.Tensor (N, 3) - surface normals
            - 'mesh_stats': dict with statistics
        """
        print(f"Loading: {obj_path}")
        
        # Load mesh (process=False to preserve colors/textures)
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        
        # Extract basic geometry
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        print(f"  Original: {len(vertices)} vertices, {len(faces)} faces")
        
        # Simplify if too many vertices
        if len(vertices) > self.max_vertices:
            print(f"  Simplifying mesh to ~{self.max_vertices} vertices...")
            # Calculate target reduction ratio
            target_faces = int((self.max_vertices / len(vertices)) * len(faces))
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                vertices = mesh.vertices.copy()
                faces = mesh.faces.copy()
                print(f"  Simplified: {len(vertices)} vertices, {len(faces)} faces")
            except Exception as e:
                print(f"  Warning: Simplification failed ({e}), using original mesh")
                # Keep original mesh if simplification fails
        
        # Normalize mesh
        if self.normalize:
            vertices = self._normalize_vertices(vertices)
            print(f"  Normalized to [-1, 1] cube")
        
        # Extract vertex colors
        vertex_colors = self._extract_vertex_colors(mesh, len(vertices))
        
        # Compute vertex normals
        vertex_normals = self._compute_vertex_normals(mesh, vertices, faces)
        
        # Convert to PyTorch tensors
        y_values = {
            'vertices': torch.from_numpy(vertices).float(),
            'faces': torch.from_numpy(faces).long(),
            'vertex_colors': torch.from_numpy(vertex_colors).float(),
            'vertex_normals': torch.from_numpy(vertex_normals).float(),
            'mesh_stats': {
                'num_vertices': len(vertices),
                'num_faces': len(faces),
                'bbox_min': vertices.min(axis=0).tolist(),
                'bbox_max': vertices.max(axis=0).tolist(),
                'has_colors': vertex_colors.max() > 0
            }
        }
        
        print(f"  ✓ Converted to training target")
        print(f"    Vertices: {y_values['vertices'].shape}")
        print(f"    Faces: {y_values['faces'].shape}")
        print(f"    Colors: {y_values['vertex_colors'].shape}")
        print(f"    Normals: {y_values['vertex_normals'].shape}")
        
        return y_values
    
    def _normalize_vertices(self, vertices):
        """
        Center at origin and normalize to fit in [-1, 1] cube
        """
        # Center
        centroid = vertices.mean(axis=0)
        vertices = vertices - centroid
        
        # Scale to [-1, 1]
        max_dist = np.abs(vertices).max()
        if max_dist > 0:
            vertices = vertices / max_dist
        
        return vertices
    
    def _extract_vertex_colors(self, mesh, num_vertices):
        """
        Extract vertex colors from mesh
        Returns RGB values in [0, 1] range
        """
        # Try to get vertex colors
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            print(f"  Found vertex colors")
            return colors
        
        # Try to get from texture/UV mapping
        if hasattr(mesh.visual, 'uv') and hasattr(mesh.visual, 'material'):
            try:
                # Sample texture at UV coordinates
                colors = self._sample_texture_at_vertices(mesh)
                if colors is not None:
                    print(f"  Extracted colors from texture")
                    return colors
            except Exception as e:
                print(f"  Could not extract texture colors: {e}")
        
        # Try to get base material color
        if hasattr(mesh.visual, 'material'):
            try:
                material = mesh.visual.material
                if hasattr(material, 'diffuse'):
                    color = material.diffuse[:3] / 255.0
                    colors = np.tile(color, (num_vertices, 1)).astype(np.float32)
                    print(f"  Using material diffuse color: {color}")
                    return colors
            except:
                pass
        
        # Default: white color
        print(f"  No colors found, using white")
        return np.ones((num_vertices, 3), dtype=np.float32)
    
    def _sample_texture_at_vertices(self, mesh):
        """
        Sample texture image at UV coordinates to get vertex colors
        """
        if not hasattr(mesh.visual, 'uv'):
            return None
        
        if not hasattr(mesh.visual.material, 'image'):
            return None
        
        # Get texture image
        texture_img = np.array(mesh.visual.material.image).astype(np.float32) / 255.0
        h, w = texture_img.shape[:2]
        
        # Get UV coordinates
        uv = mesh.visual.uv
        
        # Convert UV [0, 1] to pixel coordinates
        u_coords = (uv[:, 0] * (w - 1)).astype(np.int32)
        v_coords = ((1 - uv[:, 1]) * (h - 1)).astype(np.int32)  # Flip V
        
        # Clamp to image bounds
        u_coords = np.clip(u_coords, 0, w - 1)
        v_coords = np.clip(v_coords, 0, h - 1)
        
        # Sample texture
        colors = texture_img[v_coords, u_coords, :3]
        
        return colors.astype(np.float32)
    
    def _compute_vertex_normals(self, mesh, vertices, faces):
        """
        Compute smooth vertex normals
        """
        try:
            # Use trimesh's built-in normal computation
            normals = mesh.vertex_normals.copy()
        except:
            # Fallback: compute manually
            normals = self._compute_normals_manual(vertices, faces)
        
        return normals.astype(np.float32)
    
    def _compute_normals_manual(self, vertices, faces):
        """
        Manually compute vertex normals
        """
        normals = np.zeros_like(vertices)
        
        # Compute face normals
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
        
        # Accumulate to vertices
        for i in range(len(faces)):
            normals[faces[i, 0]] += face_normals[i]
            normals[faces[i, 1]] += face_normals[i]
            normals[faces[i, 2]] += face_normals[i]
        
        # Normalize
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        return normals


# ============================================================================
# Convert Your Specific File
# ============================================================================

def convert_single_obj_to_y(obj_path):
    """
    Simple function to convert your OBJ to Y values
    """
    converter = OBJToTrainingTarget(normalize=True, max_vertices=10000)
    y_values = converter.load_and_process(obj_path)
    return y_values


# ============================================================================
# Batch Processing for All Your Shoes
# ============================================================================

def process_all_shoes(base_path, output_dir=None):
    """
    Process all shoe OBJs in your dataset
    
    Args:
        base_path: Path to your data directory (e.g., "data/Completed")
        output_dir: Where to save processed tensors (optional)
    
    Returns:
        List of Y values for all shoes
    """
    from pathlib import Path
    import glob
    
    # Find all OBJ files
    obj_files = glob.glob(str(Path(base_path) / "*/*.obj"))
    print(f"Found {len(obj_files)} OBJ files")
    
    converter = OBJToTrainingTarget(normalize=True)
    all_y_values = []
    
    for obj_file in obj_files:
        try:
            y_values = converter.load_and_process(obj_file)
            all_y_values.append(y_values)
            
            # Optionally save processed data
            if output_dir:
                output_path = Path(output_dir) / f"{Path(obj_file).stem}.pt"
                torch.save(y_values, output_path)
                
        except Exception as e:
            print(f"  ✗ Error processing {obj_file}: {e}")
    
    print(f"\n✅ Processed {len(all_y_values)}/{len(obj_files)} meshes successfully")
    return all_y_values


# ============================================================================
# Create Dataset with X (images) and Y (mesh) pairs
# ============================================================================

from torch.utils.data import Dataset, DataLoader

class ShoeDataset(Dataset):
    """
    Complete dataset with X (multi-view images) and Y (3D mesh)
    """
    
    def __init__(self, obj_dir, images_dir):
        """
        Args:
            obj_dir: Directory containing OBJ files
            images_dir: Directory containing rendered views
                Expected naming: {shoe_id}_front.jpeg, etc.
        """
        self.obj_converter = OBJToTrainingTarget(normalize=True)
        self.images_dir = Path(images_dir)
        
        # Find all OBJs
        self.obj_paths = list(Path(obj_dir).rglob("*.obj"))
        
        # Extract shoe IDs
        self.shoe_ids = [self._extract_shoe_id(p) for p in self.obj_paths]
        
        self.views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        self.view_angles = {
            'front': (0, 0),
            'back': (180, 0),
            'left': (90, 0),
            'right': (270, 0),
            'top': (0, 90),
            'bottom': (0, -90)
        }
    
    def _extract_shoe_id(self, obj_path):
        """Extract shoe ID from path"""
        # Adjust based on your file structure
        # e.g., "data/Completed/4/10_19_2025.obj" -> "4"
        return obj_path.parent.name
    
    def __len__(self):
        return len(self.obj_paths)
    
    def __getitem__(self, idx):
        obj_path = self.obj_paths[idx]
        shoe_id = self.shoe_ids[idx]
        
        # Load Y values (ground truth mesh)
        y_values = self.obj_converter.load_and_process(obj_path)
        
        # Load X values (multi-view images)
        images = {}
        for view in self.views:
            img_path = self.images_dir / f"{shoe_id}_{view}.jpeg"
            
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")
            
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            img = torch.from_numpy(np.array(img)).float() / 255.0
            img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            images[view] = img
        
        # View angles
        angles = torch.tensor(
            [self.view_angles[v] for v in self.views],
            dtype=torch.float32
        )
        
        return {
            # X values (input)
            'images': images,
            'angles': angles,
            
            # Y values (target)
            'vertices': y_values['vertices'],
            'faces': y_values['faces'],
            'vertex_colors': y_values['vertex_colors'],
            'vertex_normals': y_values['vertex_normals'],
            
            # Metadata
            'shoe_id': shoe_id
        }


# ============================================================================
# Custom Collate Function for Variable-Sized Meshes
# ============================================================================

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized meshes
    Instead of stacking, we return lists for meshes
    """
    # Batch images (these are all the same size)
    images_batch = {}
    for view in ['front', 'back', 'left', 'right', 'top', 'bottom']:
        images_batch[view] = torch.stack([item['images'][view] for item in batch])
    
    # Batch angles (same size)
    angles_batch = torch.stack([item['angles'] for item in batch])
    
    # For meshes: keep as lists (variable sizes)
    vertices_batch = [item['vertices'] for item in batch]
    faces_batch = [item['faces'] for item in batch]
    colors_batch = [item['vertex_colors'] for item in batch]
    normals_batch = [item['vertex_normals'] for item in batch]
    
    # Shoe IDs
    shoe_ids = [item['shoe_id'] for item in batch]
    
    return {
        # X values (batched tensors)
        'images': images_batch,
        'angles': angles_batch,
        
        # Y values (lists of tensors - variable sizes!)
        'vertices': vertices_batch,
        'faces': faces_batch,
        'vertex_colors': colors_batch,
        'vertex_normals': normals_batch,
        
        # Metadata
        'shoe_id': shoe_ids
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    '''
    # ========================================
    # Example 1: Convert your specific file
    # ========================================
    print("="*60)
    print("Example 1: Convert Single OBJ to Y values")
    print("="*60)
    
    filename = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/Completed/4/10_19_2025.obj"
    
    y_values = convert_single_obj_to_y(filename)
    
    print("\nY values ready for training:")
    print(f"  Vertices: {y_values['vertices'].shape}")
    print(f"  Faces: {y_values['faces'].shape}")
    print(f"  Colors: {y_values['vertex_colors'].shape}")
    print(f"  Normals: {y_values['vertex_normals'].shape}")
    
    # These are your ground truth targets!
    # Use them in your loss function
    
    
    # ========================================
    # Example 2: Process all shoes
    # ========================================
    print("\n" + "="*60)
    print("Example 2: Process All Shoes")
    print("="*60)
    
    base_path = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/Completed"
    all_y_values = process_all_shoes(base_path, output_dir="processed_meshes")
    
    print(f"\nProcessed {len(all_y_values)} meshes")
    
    '''
    # ========================================
    # Example 3: Create complete dataset
    # ========================================
    print("\n" + "="*60)
    print("Example 3: Create Training Dataset")
    print("="*60)
    
    dataset = ShoeDataset(
        obj_dir=r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/Completed",
        images_dir=r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/input_images"
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        collate_fn=custom_collate_fn  # ← ADD THIS!
    )
    
    # Test loading one batch
    print("\nTesting batch loading...")
    for batch in dataloader:
        print(f"\nBatch loaded:")
        print(f"  X (Input):")
        print(f"    Images: {list(batch['images'].keys())}")
        print(f"    Angles: {batch['angles'].shape}")
        
        print(f"  Y (Target):")
        print(f"    Vertices: {batch['vertices'].shape}")
        print(f"    Faces: {batch['faces'].shape}")
        print(f"    Colors: {batch['vertex_colors'].shape}")
        print(f"    Normals: {batch['vertex_normals'].shape}")
        
        break  # Just show first batch
    
    '''
    # ========================================
    # Example 4: Use in training loop
    # ========================================
    print("\n" + "="*60)
    print("Example 4: Training Loop Example")
    print("="*60)
    
    print("""
# Pseudocode for training:

for batch in dataloader:
    # X: Input multi-view images
    images = batch['images']
    angles = batch['angles']
    
    # Y: Ground truth mesh (targets)
    gt_vertices = batch['vertices']
    gt_faces = batch['faces']
    gt_colors = batch['vertex_colors']
    
    # Forward pass
    pred_vertices, pred_faces = geometry_model(images, angles)
    pred_colors = texture_model(pred_vertices, pred_faces, images, angles)
    
    # Compute loss against Y values
    loss = chamfer_distance(pred_vertices, gt_vertices)
    loss += color_loss(pred_colors, gt_colors)
    
    # Backward pass
    loss.backward()
    optimizer.step()
""")
    
    print("\n✅ All examples completed!")
    '''