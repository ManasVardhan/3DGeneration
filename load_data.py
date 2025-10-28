"""
Dataset Loader for Multi-View to 3D Mesh Training
Loads X (images) and Y (3D meshes) with proper ID mapping
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import numpy as np
import trimesh


# ============================================================================
# OBJ to Training Target Converter
# ============================================================================

class OBJToTrainingTarget:
    """Convert 3D OBJ files to training targets (Y values)"""
    
    def __init__(self, normalize=True, max_vertices=10000):
        self.normalize = normalize
        self.max_vertices = max_vertices
    
    def load_and_process(self, obj_path):
        """Load OBJ and convert to training target format"""
        print(f"Loading: {obj_path}")
        
        # Load mesh
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        
        # Extract geometry
        vertices = mesh.vertices.copy()
        faces = mesh.faces.copy()
        
        print(f"  Original: {len(vertices)} vertices, {len(faces)} faces")
        
        # Simplify if too many vertices
        if len(vertices) > self.max_vertices:
            print(f"  Simplifying mesh to ~{self.max_vertices} vertices...")
            target_faces = int((self.max_vertices / len(vertices)) * len(faces))
            try:
                mesh = mesh.simplify_quadric_decimation(target_faces)
                vertices = mesh.vertices.copy()
                faces = mesh.faces.copy()
                print(f"  Simplified: {len(vertices)} vertices, {len(faces)} faces")
            except Exception as e:
                print(f"  Warning: Simplification failed ({e}), using original mesh")
        
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
        
        return y_values
    
    def _normalize_vertices(self, vertices):
        """Center at origin and normalize to [-1, 1] cube"""
        centroid = vertices.mean(axis=0)
        vertices = vertices - centroid
        
        max_dist = np.abs(vertices).max()
        if max_dist > 0:
            vertices = vertices / max_dist
        
        return vertices
    
    def _extract_vertex_colors(self, mesh, num_vertices):
        """Extract vertex colors from mesh"""
        if hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0
            print(f"  Found vertex colors")
            return colors
        
        if hasattr(mesh.visual, 'uv') and hasattr(mesh.visual, 'material'):
            try:
                colors = self._sample_texture_at_vertices(mesh)
                if colors is not None:
                    print(f"  Extracted colors from texture")
                    return colors
            except Exception as e:
                print(f"  Could not extract texture colors: {e}")
        
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
        
        print(f"  No colors found, using white")
        return np.ones((num_vertices, 3), dtype=np.float32)
    
    def _sample_texture_at_vertices(self, mesh):
        """Sample texture image at UV coordinates"""
        if not hasattr(mesh.visual, 'uv'):
            return None
        
        if not hasattr(mesh.visual.material, 'image'):
            return None
        
        texture_img = np.array(mesh.visual.material.image).astype(np.float32) / 255.0
        h, w = texture_img.shape[:2]
        
        uv = mesh.visual.uv
        
        u_coords = (uv[:, 0] * (w - 1)).astype(np.int32)
        v_coords = ((1 - uv[:, 1]) * (h - 1)).astype(np.int32)
        
        u_coords = np.clip(u_coords, 0, w - 1)
        v_coords = np.clip(v_coords, 0, h - 1)
        
        colors = texture_img[v_coords, u_coords, :3]
        
        return colors.astype(np.float32)
    
    def _compute_vertex_normals(self, mesh, vertices, faces):
        """Compute smooth vertex normals"""
        try:
            normals = mesh.vertex_normals.copy()
        except:
            normals = self._compute_normals_manual(vertices, faces)
        
        return normals.astype(np.float32)
    
    def _compute_normals_manual(self, vertices, faces):
        """Manually compute vertex normals"""
        normals = np.zeros_like(vertices)
        
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        face_normals = np.cross(v1 - v0, v2 - v0)
        face_normals = face_normals / (np.linalg.norm(face_normals, axis=1, keepdims=True) + 1e-8)
        
        for i in range(len(faces)):
            normals[faces[i, 0]] += face_normals[i]
            normals[faces[i, 1]] += face_normals[i]
            normals[faces[i, 2]] += face_normals[i]
        
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-8)
        
        return normals


# ============================================================================
# Dataset with X (images) and Y (mesh) pairs
# ============================================================================

class ShoeDataset(Dataset):
    """Complete dataset with X (multi-view images) and Y (3D mesh)"""
    
    def __init__(self, obj_dir, images_dir, verify_mappings=True):
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
        
        # Verify X-Y mappings
        if verify_mappings:
            self._verify_all_mappings()
    
    def _extract_shoe_id(self, obj_path):
        """Extract shoe ID from path"""
        return obj_path.parent.name
    
    def _verify_all_mappings(self):
        """Verify that all OBJs have corresponding images"""
        print("\n" + "="*60)
        print("VERIFYING X-Y MAPPINGS")
        print("="*60)
        
        missing_images = []
        valid_pairs = 0
        
        for idx, (obj_path, shoe_id) in enumerate(zip(self.obj_paths, self.shoe_ids)):
            missing_views = []
            
            for view in self.views:
                img_path_png = self.images_dir / f"{shoe_id}_{view}.png"
                img_path_jpeg = self.images_dir / f"{shoe_id}_{view}.jpeg"
                img_path_jpg = self.images_dir / f"{shoe_id}_{view}.jpg"
                
                if not (img_path_png.exists() or img_path_jpeg.exists() or img_path_jpg.exists()):
                    missing_views.append(view)
            
            if missing_views:
                missing_images.append({
                    'shoe_id': shoe_id,
                    'obj_path': str(obj_path),
                    'missing_views': missing_views
                })
            else:
                valid_pairs += 1
        
        print(f"\n✓ Valid X-Y pairs: {valid_pairs}/{len(self.obj_paths)}")
        
        if missing_images:
            print(f"\n⚠️  WARNING: {len(missing_images)} shoes have missing images:")
            for item in missing_images[:5]:
                print(f"  Shoe ID: {item['shoe_id']}")
                print(f"    OBJ: {item['obj_path']}")
                print(f"    Missing views: {', '.join(item['missing_views'])}")
            
            if len(missing_images) > 5:
                print(f"  ... and {len(missing_images) - 5} more")
            
            print(f"\n🔧 Removing {len(missing_images)} invalid pairs from dataset")
            valid_indices = [i for i in range(len(self.obj_paths)) 
                           if self.shoe_ids[i] not in [m['shoe_id'] for m in missing_images]]
            self.obj_paths = [self.obj_paths[i] for i in valid_indices]
            self.shoe_ids = [self.shoe_ids[i] for i in valid_indices]
            
            print(f"✓ Dataset cleaned: {len(self.obj_paths)} valid pairs remaining")
        else:
            print("✓ All X-Y mappings verified successfully!")
        
        print("="*60 + "\n")
    
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
            img_path_png = self.images_dir / f"{shoe_id}_{view}.png"
            img_path_jpeg = self.images_dir / f"{shoe_id}_{view}.jpeg"
            img_path_jpg = self.images_dir / f"{shoe_id}_{view}.jpg"
            
            if img_path_png.exists():
                img_path = img_path_png
            elif img_path_jpeg.exists():
                img_path = img_path_jpeg
            elif img_path_jpg.exists():
                img_path = img_path_jpg
            else:
                raise FileNotFoundError(
                    f"Missing image for Shoe ID {shoe_id}, view {view}"
                )
            
            img = Image.open(img_path).convert('RGB')
            img = torch.from_numpy(np.array(img)).float() / 255.0
            img = img.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
            images[view] = img
        
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
            'shoe_id': shoe_id,
            'obj_path': str(obj_path)
        }


# ============================================================================
# Custom Collate Function (for variable-sized meshes)
# ============================================================================

def custom_collate_fn(batch):
    """Handle variable-sized meshes in batches"""
    # Batch images (same size)
    images_batch = {}
    for view in ['front', 'back', 'left', 'right', 'top', 'bottom']:
        images_batch[view] = torch.stack([item['images'][view] for item in batch])
    
    # Batch angles (same size)
    angles_batch = torch.stack([item['angles'] for item in batch])
    
    # Meshes as lists (variable sizes)
    vertices_batch = [item['vertices'] for item in batch]
    faces_batch = [item['faces'] for item in batch]
    colors_batch = [item['vertex_colors'] for item in batch]
    normals_batch = [item['vertex_normals'] for item in batch]
    
    # Metadata
    shoe_ids = [item['shoe_id'] for item in batch]
    obj_paths = [item['obj_path'] for item in batch]
    
    return {
        'images': images_batch,
        'angles': angles_batch,
        'vertices': vertices_batch,
        'faces': faces_batch,
        'vertex_colors': colors_batch,
        'vertex_normals': normals_batch,
        'shoe_id': shoe_ids,
        'obj_path': obj_paths
    }