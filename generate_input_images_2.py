"""
Interactive 3D Shoe View Capturer
Manually rotate and capture correct views for each shoe
"""

import tkinter as tk
from tkinter import ttk, messagebox
import trimesh
import numpy as np
from PIL import Image, ImageTk
import io
from pathlib import Path
import json


# ============================================================================
# Configuration
# ============================================================================

BASE_DATA_PATH = r"/Users/manasvardhan/Desktop/3D/3DGeneration/data"
ARCHIVE_PATH = Path(BASE_DATA_PATH) / "Completed"
OUTPUT_PATH = Path(BASE_DATA_PATH) / "input_images_manual"
PROGRESS_FILE = Path(BASE_DATA_PATH) / "capture_progress.json"

# Image resolution for saved views
SAVE_RESOLUTION = (512, 512)
PREVIEW_SIZE = (600, 600)

# Create output directory
OUTPUT_PATH.mkdir(exist_ok=True)


# ============================================================================
# 3D Viewer Class
# ============================================================================

class ShoeViewer:
    def __init__(self, obj_path):
        """Load and prepare shoe mesh for viewing"""
        self.obj_path = obj_path
        
        # Load mesh
        self.mesh = trimesh.load(obj_path, force='mesh', process=False)
        
        # Center and normalize
        self.mesh.vertices -= self.mesh.centroid
        max_extent = np.max(np.abs(self.mesh.vertices))
        if max_extent > 0:
            self.mesh.vertices /= max_extent
        
        # Current rotation
        self.rotation = np.eye(4)
        
        # Mouse interaction state
        self.last_x = 0
        self.last_y = 0
        self.is_dragging = False
    
    def rotate(self, delta_x, delta_y):
        """Rotate mesh based on mouse movement"""
        # Rotation around Y axis (horizontal drag)
        if delta_x != 0:
            rot_y = trimesh.transformations.rotation_matrix(
                np.radians(delta_x * 0.5), [0, 1, 0]
            )
            self.rotation = rot_y @ self.rotation
        
        # Rotation around X axis (vertical drag)
        if delta_y != 0:
            rot_x = trimesh.transformations.rotation_matrix(
                np.radians(delta_y * 0.5), [1, 0, 0]
            )
            self.rotation = rot_x @ self.rotation
    
    def render(self, resolution=(600, 600)):
        """Render current view"""
        # Create transformed mesh
        mesh_copy = self.mesh.copy()
        mesh_copy.apply_transform(self.rotation)
        
        # Create scene
        scene = mesh_copy.scene()
        
        # Camera setup
        camera_pose = np.eye(4)
        camera_pose[2, 3] = 2.5  # Move camera back
        scene.camera_transform = camera_pose
        
        try:
            # Render
            png_bytes = scene.save_image(resolution=resolution, visible=True)
            image = Image.open(io.BytesIO(png_bytes))
            
            # Convert to RGB if needed
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            
            return image
        except Exception as e:
            print(f"Render error: {e}")
            return None
        finally:
            del scene, mesh_copy
    
    def reset_rotation(self):
        """Reset to default view"""
        self.rotation = np.eye(4)


# ============================================================================
# Main GUI Application
# ============================================================================

class ShoeCapturerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Interactive Shoe View Capturer")
        self.root.geometry("800x700")
        
        # State
        self.current_shoe_id = None
        self.viewer = None
        self.captured_views = set()
        self.progress = self.load_progress()
        
        # Get list of shoes
        self.shoe_list = self.get_shoe_list()
        self.current_index = 0
        
        # Setup UI
        self.setup_ui()
        
        # Load first shoe
        if self.shoe_list:
            self.load_shoe(self.shoe_list[0])
    
    def setup_ui(self):
        """Create the user interface"""
        
        # Top controls
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)
        
        # Shoe selector
        ttk.Label(control_frame, text="Shoe:").pack(side=tk.LEFT, padx=5)
        
        self.shoe_var = tk.StringVar()
        self.shoe_combo = ttk.Combobox(
            control_frame, 
            textvariable=self.shoe_var,
            values=self.shoe_list,
            width=10,
            state='readonly'
        )
        self.shoe_combo.pack(side=tk.LEFT, padx=5)
        self.shoe_combo.bind('<<ComboboxSelected>>', self.on_shoe_selected)
        
        # Navigation buttons
        ttk.Button(
            control_frame, 
            text="← Previous", 
            command=self.prev_shoe
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame, 
            text="Next →", 
            command=self.next_shoe
        ).pack(side=tk.LEFT, padx=5)
        
        # Progress label
        self.progress_label = ttk.Label(control_frame, text="")
        self.progress_label.pack(side=tk.RIGHT, padx=10)
        
        # 3D Viewport
        viewport_frame = ttk.LabelFrame(self.root, text="3D View (Drag to Rotate)", padding=10)
        viewport_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.canvas = tk.Canvas(
            viewport_frame, 
            width=PREVIEW_SIZE[0], 
            height=PREVIEW_SIZE[1],
            bg='white'
        )
        self.canvas.pack()
        
        # Mouse bindings
        self.canvas.bind('<ButtonPress-1>', self.on_mouse_press)
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_mouse_release)
        
        # Reset button
        ttk.Button(
            viewport_frame,
            text="Reset View",
            command=self.reset_view
        ).pack(pady=5)
        
        # Capture buttons
        button_frame = ttk.LabelFrame(self.root, text="Capture Views", padding=10)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Create grid of capture buttons
        views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        for i, view in enumerate(views):
            btn = ttk.Button(
                button_frame,
                text=f"Capture {view.upper()}",
                command=lambda v=view: self.capture_view(v),
                width=15
            )
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            
            # Store button reference for styling
            setattr(self, f'btn_{view}', btn)
        
        # Status display
        status_frame = ttk.LabelFrame(self.root, text="Captured Views", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.status_text = tk.Text(status_frame, height=3, state='disabled')
        self.status_text.pack(fill=tk.X)
        
        # Instructions
        instructions = """
INSTRUCTIONS:
1. Drag mouse on 3D view to rotate the shoe
2. Position shoe to show the correct view (e.g., front shows toe)
3. Click the corresponding capture button
4. Repeat for all 6 views
5. Use Next → to move to the next shoe
        """
        
        instr_frame = ttk.LabelFrame(self.root, text="How to Use", padding=5)
        instr_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(instr_frame, text=instructions, justify=tk.LEFT).pack()
    
    def get_shoe_list(self):
        """Get list of shoe IDs"""
        shoes = []
        for folder in sorted(ARCHIVE_PATH.iterdir()):
            if folder.is_dir():
                # Check if OBJ exists
                obj_files = list(folder.glob("*.obj"))
                if obj_files:
                    shoes.append(folder.name)
        return shoes
    
    def load_shoe(self, shoe_id):
        """Load a shoe for viewing"""
        print(f"Loading shoe {shoe_id}...")
        
        self.current_shoe_id = shoe_id
        self.shoe_var.set(shoe_id)
        
        # Find OBJ file
        shoe_folder = ARCHIVE_PATH / shoe_id
        obj_files = list(shoe_folder.glob("*.obj"))
        
        if not obj_files:
            messagebox.showerror("Error", f"No OBJ file found for shoe {shoe_id}")
            return
        
        obj_path = obj_files[0]
        
        # Load viewer
        try:
            self.viewer = ShoeViewer(str(obj_path))
            self.update_display()
            self.update_captured_status()
            self.update_progress_label()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load shoe: {e}")
    
    def update_display(self):
        """Render and display current view"""
        if not self.viewer:
            return
        
        # Render
        image = self.viewer.render(resolution=PREVIEW_SIZE)
        
        if image:
            # Convert to PhotoImage
            self.photo = ImageTk.PhotoImage(image)
            
            # Update canvas
            self.canvas.delete('all')
            self.canvas.create_image(
                PREVIEW_SIZE[0]//2, 
                PREVIEW_SIZE[1]//2, 
                image=self.photo
            )
    
    def on_mouse_press(self, event):
        """Handle mouse press"""
        self.viewer.last_x = event.x
        self.viewer.last_y = event.y
        self.viewer.is_dragging = True
    
    def on_mouse_drag(self, event):
        """Handle mouse drag to rotate"""
        if not self.viewer or not self.viewer.is_dragging:
            return
        
        delta_x = event.x - self.viewer.last_x
        delta_y = event.y - self.viewer.last_y
        
        self.viewer.rotate(delta_x, delta_y)
        self.viewer.last_x = event.x
        self.viewer.last_y = event.y
        
        self.update_display()
    
    def on_mouse_release(self, event):
        """Handle mouse release"""
        if self.viewer:
            self.viewer.is_dragging = False
    
    def reset_view(self):
        """Reset camera view"""
        if self.viewer:
            self.viewer.reset_rotation()
            self.update_display()
    
    def capture_view(self, view_name):
        """Capture current view as specific angle"""
        if not self.viewer or not self.current_shoe_id:
            return
        
        # Render at high resolution
        image = self.viewer.render(resolution=SAVE_RESOLUTION)
        
        if image:
            # Save image
            filename = f"{self.current_shoe_id}_{view_name}.png"
            filepath = OUTPUT_PATH / filename
            image.save(filepath, 'PNG')
            
            # Update captured views
            self.captured_views.add(view_name)
            self.update_captured_status()
            
            # Save progress
            if self.current_shoe_id not in self.progress:
                self.progress[self.current_shoe_id] = []
            if view_name not in self.progress[self.current_shoe_id]:
                self.progress[self.current_shoe_id].append(view_name)
            self.save_progress()
            
            print(f"✓ Captured {view_name}: {filepath}")
            
            # Check if all views captured
            if len(self.captured_views) == 6:
                result = messagebox.askquestion(
                    "Complete",
                    f"All 6 views captured for shoe {self.current_shoe_id}!\n\nMove to next shoe?",
                    icon='info'
                )
                if result == 'yes':
                    self.next_shoe()
    
    def update_captured_status(self):
        """Update the status display"""
        if not self.current_shoe_id:
            return
        
        # Get captured views for current shoe
        if self.current_shoe_id in self.progress:
            self.captured_views = set(self.progress[self.current_shoe_id])
        else:
            self.captured_views = set()
        
        # Update text
        all_views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        status_lines = []
        
        for view in all_views:
            if view in self.captured_views:
                status_lines.append(f"✓ {view.upper()}")
            else:
                status_lines.append(f"○ {view.upper()}")
        
        status_text = "  ".join(status_lines)
        
        self.status_text.config(state='normal')
        self.status_text.delete('1.0', tk.END)
        self.status_text.insert('1.0', status_text)
        self.status_text.config(state='disabled')
        
        # Update button colors
        for view in all_views:
            btn = getattr(self, f'btn_{view}', None)
            if btn:
                if view in self.captured_views:
                    btn.state(['disabled'])
                else:
                    btn.state(['!disabled'])
    
    def update_progress_label(self):
        """Update overall progress"""
        completed_shoes = len([s for s in self.progress if len(self.progress[s]) == 6])
        total_shoes = len(self.shoe_list)
        
        self.progress_label.config(
            text=f"Progress: {completed_shoes}/{total_shoes} shoes complete"
        )
    
    def on_shoe_selected(self, event):
        """Handle shoe selection from dropdown"""
        shoe_id = self.shoe_var.get()
        if shoe_id:
            self.current_index = self.shoe_list.index(shoe_id)
            self.load_shoe(shoe_id)
    
    def next_shoe(self):
        """Load next shoe"""
        if self.current_index < len(self.shoe_list) - 1:
            self.current_index += 1
            self.load_shoe(self.shoe_list[self.current_index])
        else:
            messagebox.showinfo("Complete", "You've reached the last shoe!")
    
    def prev_shoe(self):
        """Load previous shoe"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_shoe(self.shoe_list[self.current_index])
    
    def load_progress(self):
        """Load saved progress"""
        if PROGRESS_FILE.exists():
            with open(PROGRESS_FILE, 'r') as f:
                return json.load(f)
        return {}
    
    def save_progress(self):
        """Save progress to file"""
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(self.progress, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    root = tk.Tk()
    app = ShoeCapturerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()