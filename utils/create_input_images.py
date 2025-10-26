import trimesh
import numpy as np
from PIL import Image
import os
import io

# Base paths
base_data_path = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data"
archive_path = os.path.join(base_data_path, "archive")
output_path = os.path.join(base_data_path, "input_images")

# Image resolution
IMAGE_RESOLUTION = (512, 512)  # Change this to (224, 224) or (1024, 1024) as needed

# Create output directory if it doesn't exist
os.makedirs(output_path, exist_ok=True)

def render_view_with_texture(mesh, rotation_matrix, resolution=(512, 512)):
    """
    Render a view of the mesh with textures using trimesh
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
    except Exception as e:
        print(f"      Warning: Rendering with textures failed ({e}), trying alternative...")
        try:
            png_bytes = scene.save_image(resolution=resolution)
            image = Image.open(io.BytesIO(png_bytes))
            return np.array(image)
        except Exception as e2:
            print(f"      Error: Could not render view ({e2})")
            return None

def find_obj_file(folder_path):
    """Find the OBJ file in the folder"""
    for file in os.listdir(folder_path):
        if file.lower().endswith('.obj'):
            return os.path.join(folder_path, file)
    return None

def process_folder(folder_num):
    """Process a single folder and save textured views with naming: {folder_num}_{view}.jpeg"""
    folder_path = os.path.join(archive_path, str(folder_num))
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"⚠️  Folder {folder_num} not found, skipping...")
        return False
    
    # Find OBJ file
    obj_file = find_obj_file(folder_path)
    if obj_file is None:
        print(f"⚠️  No OBJ file found in folder {folder_num}, skipping...")
        return False
    
    print(f"\n📦 Processing folder {folder_num}: {os.path.basename(obj_file)}")
    
    try:
        # Load the OBJ file with textures (process=False preserves materials)
        mesh = trimesh.load(obj_file, force='mesh', process=False)
        
        # Center and normalize the mesh
        mesh.vertices -= mesh.centroid
        scale = np.max(np.abs(mesh.vertices))
        mesh.vertices /= scale
        
        print(f"   Mesh loaded: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"   Visual type: {mesh.visual.kind if hasattr(mesh.visual, 'kind') else 'none'}")
        
        # Define rotation matrices for each view
        views = {
            'front': trimesh.transformations.rotation_matrix(0, [0, 0, 1]),
            'back': trimesh.transformations.rotation_matrix(np.pi, [0, 0, 1]),
            'left': trimesh.transformations.rotation_matrix(-np.pi/2, [0, 0, 1]),
            'right': trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]),
            'top': trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]),
            'bottom': trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0])
        }
        
        # Render and save all views
        views_rendered = 0
        for view_name, rotation in views.items():
            img = render_view_with_texture(mesh, rotation, resolution=IMAGE_RESOLUTION)
            
            if img is not None:
                # Save as: data/input_images/{folder_num}_{view_name}.jpeg
                output_filename = f"{folder_num}_{view_name}.jpeg"
                output_filepath = os.path.join(output_path, output_filename)
                
                # Convert to PIL Image and save as JPEG
                pil_img = Image.fromarray(img)
                # Convert RGBA to RGB if necessary (JPEG doesn't support alpha channel)
                if pil_img.mode == 'RGBA':
                    pil_img = pil_img.convert('RGB')
                pil_img.save(output_filepath, 'JPEG', quality=95)
                
                print(f"   ✓ Saved {output_filename}")
                views_rendered += 1
            else:
                print(f"   ✗ Failed to render {view_name}")
        
        if views_rendered == 6:
            return True
        else:
            print(f"   ⚠️  Only {views_rendered}/6 views rendered successfully")
            return views_rendered > 0
        
    except Exception as e:
        print(f"   ❌ Error processing folder {folder_num}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# Main processing loop
print("="*60)
print("Starting multi-view rendering with TEXTURES for all folders")
print(f"Resolution: {IMAGE_RESOLUTION[0]}x{IMAGE_RESOLUTION[1]} pixels per image")
print(f"Output format: {{folder_num}}_{{view}}.jpeg")
print("="*60)

successful = 0
failed = 0

for folder_num in range(1, 106):  # Folders 1 to 105
    if process_folder(folder_num):
        successful += 1
    else:
        failed += 1

print("\n" + "="*60)
print(f"✅ Processing complete!")
print(f"   Successful: {successful}/{105}")
print(f"   Failed: {failed}/{105}")
print(f"   Output directory: {output_path}")
print("="*60)
