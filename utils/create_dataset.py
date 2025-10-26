# Install required libraries
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from io import BytesIO

# Specify the path to your OBJ file
filename = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data/Blue Vans Shoe/Blue Vans Shoe.obj"  
# Change this to your file path  # Change this to your file path

# Load the OBJ file
mesh = trimesh.load_mesh(filename)

# Center and normalize the mesh
mesh.vertices -= mesh.centroid
scale = np.max(np.abs(mesh.vertices))
mesh.vertices /= scale

print(f"Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")

# ============================================
# GENERATE MULTI-VIEW IMAGES (X)
# ============================================

def render_view(mesh, azimuth, elevation, resolution=(224, 224)):
    """
    Render a view of the mesh using matplotlib
    azimuth: rotation around z-axis (degrees)
    elevation: angle from xy-plane (degrees)
    """
    fig = plt.figure(figsize=(resolution[0]/100, resolution[1]/100), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    # Plot the mesh
    ax.plot_trisurf(mesh.vertices[:, 0],
                    mesh.vertices[:, 1],
                    mesh.vertices[:, 2],
                    triangles=mesh.faces,
                    cmap='viridis',
                    alpha=0.9,
                    edgecolor='none',
                    shade=True,
                    lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45))

    # Set view angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Set equal aspect ratio and remove axes
    max_range = np.array([mesh.vertices[:, 0].max()-mesh.vertices[:, 0].min(),
                          mesh.vertices[:, 1].max()-mesh.vertices[:, 1].min(),
                          mesh.vertices[:, 2].max()-mesh.vertices[:, 2].min()]).max() / 2.0
    mid_x = (mesh.vertices[:, 0].max()+mesh.vertices[:, 0].min()) * 0.5
    mid_y = (mesh.vertices[:, 1].max()+mesh.vertices[:, 1].min()) * 0.5
    mid_z = (mesh.vertices[:, 2].max()+mesh.vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1,1,1])
    ax.axis('off')
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Convert to image array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, facecolor='white')
    plt.close(fig)
    buf.seek(0)

    img = Image.open(buf)
    img = img.resize(resolution, Image.LANCZOS)
    img_array = np.array(img)[:, :, :3]  # Remove alpha channel if present

    return img_array

# Define camera angles for 6 views (azimuth, elevation)
views = {
    'front': (0, 0),      # Looking from front
    'back': (180, 0),     # Looking from back
    'left': (90, 0),      # Looking from left
    'right': (270, 0),    # Looking from right (or -90)
    'top': (0, 90),       # Looking from top
    'bottom': (0, -90)    # Looking from bottom
}

# Render all views
rendered_views = {}
print("\nRendering views...")
for view_name, (azim, elev) in views.items():
    print(f"  Rendering {view_name} view...")
    img = render_view(mesh, azim, elev)
    rendered_views[view_name] = img

# Visualize the 6 views
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (view_name, img) in enumerate(rendered_views.items()):
    axes[idx].imshow(img)
    axes[idx].set_title(f'{view_name.capitalize()} View', fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('multi_view_renders.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# PREPARE X (INPUT): Multi-view images
# ============================================

# Stack all views into a single array
X = np.stack([rendered_views[view] for view in ['front', 'back', 'left', 'right', 'top', 'bottom']])
print(f"\nX shape (multi-view images): {X.shape}")  # (6, 224, 224, 3)

# Alternative: Concatenate views as channels or flatten
X_flattened = X.reshape(6, -1)  # (6, 224*224*3)
print(f"X flattened shape: {X_flattened.shape}")

# ============================================
# PREPARE Y (OUTPUT): 3D representation
# ============================================

# Option 1: Voxel Grid (3D occupancy grid)
def mesh_to_voxels(mesh, resolution=32):
    """Convert mesh to voxel grid"""
    voxels = mesh.voxelized(pitch=2.0/resolution)
    grid = voxels.matrix
    return grid

Y_voxels = mesh_to_voxels(mesh, resolution=32)
print(f"\nY shape (voxel grid): {Y_voxels.shape}")  # (32, 32, 32)

# Option 2: Point Cloud
Y_points = mesh.sample(4096)  # Sample 4096 points from surface
print(f"Y shape (point cloud): {Y_points.shape}")  # (4096, 3)

# Option 3: Mesh vertices and faces (raw representation)
Y_vertices = mesh.vertices
Y_faces = mesh.faces
print(f"Y vertices shape: {Y_vertices.shape}")  # (N, 3)
print(f"Y faces shape: {Y_faces.shape}")  # (M, 3)
