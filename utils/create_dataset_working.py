import trimesh
import numpy as np
from PIL import Image
import io

filename = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/Blue Vans Shoe/Blue Vans Shoe.obj"
mesh = trimesh.load(filename, force='mesh', process=False)

# Center and normalize
mesh.vertices -= mesh.centroid
scale = np.max(np.abs(mesh.vertices))
mesh.vertices /= scale

print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

def render_view_simple(mesh, rotation_matrix, resolution=(512, 512)):
    """
    Simplified rendering by rotating the mesh instead of the camera
    """
    # Create a copy of the mesh to avoid modifying the original
    mesh_copy = mesh.copy()
    
    # Apply rotation to the mesh
    mesh_copy.apply_transform(rotation_matrix)
    
    # Create scene
    scene = mesh_copy.scene()
    
    # Simple camera setup - just move it back on Z axis
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 2.5  # Move camera back along Z
    scene.camera_transform = camera_pose
    
    # Render
    try:
        png_bytes = scene.save_image(resolution=resolution, visible=True)
        image = Image.open(io.BytesIO(png_bytes))
        return np.array(image)
    except:
        print("Rendering failed, using alternative method...")
        png_bytes = scene.save_image(resolution=resolution)
        image = Image.open(io.BytesIO(png_bytes))
        return np.array(image)

# Fixed rotation matrices for each view
# The shoe needs proper orientation corrections
views = {
    'right': trimesh.transformations.rotation_matrix(0, [0, 0, 1]),  # Image 1: sole view
    'left': trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]) @ trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0]),  # Image 2: inside view
    'back': trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]) @ trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]),  # Image 3: toe view
    'bottom': trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]) @ trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]),  # Image 4: outer side
    'front': trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]) @ trimesh.transformations.rotation_matrix(-np.pi/2, [0, 0, 1]),  # Image 5: heel view
    'top': trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]),  # Image 6: inner side
}

# Render all views
rendered_views = {}
print("\nRendering textured views...")
for view_name, rotation in views.items():
    print(f"  Rendering {view_name} view...")
    img = render_view_simple(mesh, rotation, resolution=(512, 512))
    rendered_views[view_name] = img
    
    # Save individual view
    Image.fromarray(img).save(f'view_{view_name}.png')
    print(f"  Saved view_{view_name}.png")

print("Done!")

# Stack for your dataset in the correct order
X = np.stack([rendered_views[view] for view in ['front', 'back', 'left', 'right', 'top', 'bottom']])
print(f"\nX shape (multi-view images): {X.shape}")