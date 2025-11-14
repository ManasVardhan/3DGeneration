"""
Data Inspector - Shows EXACT data that model receives
Replicates all preprocessing and displays numerical values for inspection
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import trimesh
import matplotlib.pyplot as plt
from collections import OrderedDict


class DataInspector:
    """Inspect actual model inputs and outputs with all preprocessing"""
    
    def __init__(self):
        self.report = []
        
    def log(self, message):
        """Log and print message"""
        print(message)
        self.report.append(message)
    
    def inspect_complete_pipeline(self, shoe_id=None):
        """Inspect complete data pipeline"""
        self.log("="*70)
        self.log("DATA PIPELINE INSPECTION")
        self.log("="*70)
        
        # Load config
        from config import config
        self.log(f"\n📋 Configuration:")
        self.log(f"   OBJ dir:    {config.obj_dir}")
        self.log(f"   Images dir: {config.images_dir}")
        self.log(f"   Image size: {config.image_size}")
        self.log(f"   Device:     {config.device}")
        
        # Load dataset
        from load_data import ShoeDataset, custom_collate_fn
        
        dataset = ShoeDataset(
            obj_dir=config.obj_dir,
            images_dir=config.images_dir,
            views=['front', 'back', 'left', 'right', 'top', 'bottom'],
            image_size=config.image_size
        )
        
        if len(dataset) == 0:
            self.log("\n❌ Dataset is empty!")
            return
        
        # Pick sample
        if shoe_id is not None:
            idx = None
            for i, sid in enumerate(dataset.shoe_ids):
                if sid == shoe_id:
                    idx = i
                    break
            if idx is None:
                self.log(f"\n❌ Shoe ID '{shoe_id}' not found!")
                return
        else:
            idx = 0
        
        self.log(f"\n📦 Inspecting sample {idx}: {dataset.shoe_ids[idx]}")
        self.log("="*70)
        
        # Get raw sample
        sample = dataset[idx]
        
        # Inspect each component
        self._inspect_images(sample['images'], config)
        self._inspect_vertices(sample['vertices'], sample['obj_path'])
        self._inspect_faces(sample['faces'])
        self._inspect_vertex_colors(sample['vertex_colors'])
        self._inspect_vertex_normals(sample['vertex_normals'])
        
        # Simulate batch processing
        self._inspect_batch_collation([sample])
        
        # Simulate model forward pass
        self._inspect_model_forward(sample, config)
        
        # Save report
        self._save_report()
        
    def _inspect_images(self, images, config):
        """Inspect image tensors"""
        self.log("\n" + "="*70)
        self.log("📸 INPUT X: IMAGES")
        self.log("="*70)
        
        views = ['front', 'back', 'left', 'right', 'top', 'bottom']
        
        for view in views:
            img = images[view]
            self.log(f"\n{view.upper()}:")
            self.log(f"  Shape:     {tuple(img.shape)} (C, H, W)")
            self.log(f"  Dtype:     {img.dtype}")
            self.log(f"  Device:    cpu (before model)")
            self.log(f"  Min value: {img.min().item():.6f}")
            self.log(f"  Max value: {img.max().item():.6f}")
            self.log(f"  Mean:      {img.mean().item():.6f}")
            self.log(f"  Std:       {img.std().item():.6f}")
            
            # Check range
            if img.min() < 0 or img.max() > 1:
                self.log(f"  ⚠️  WARNING: Values outside [0, 1] range!")
            else:
                self.log(f"  ✓ Values in correct [0, 1] range")
            
            # Sample pixel values
            self.log(f"  Sample pixels (R,G,B):")
            self.log(f"    Top-left:     [{img[0, 0, 0]:.4f}, {img[1, 0, 0]:.4f}, {img[2, 0, 0]:.4f}]")
            self.log(f"    Center:       [{img[0, 256, 256]:.4f}, {img[1, 256, 256]:.4f}, {img[2, 256, 256]:.4f}]")
            self.log(f"    Bottom-right: [{img[0, -1, -1]:.4f}, {img[1, -1, -1]:.4f}, {img[2, -1, -1]:.4f}]")
        
        # Overall statistics
        all_imgs = torch.stack([images[v] for v in views])
        self.log(f"\n📊 OVERALL IMAGE STATISTICS:")
        self.log(f"  Total shape:  {tuple(all_imgs.shape)} (Views, C, H, W)")
        self.log(f"  Global mean:  {all_imgs.mean().item():.6f}")
        self.log(f"  Global std:   {all_imgs.std().item():.6f}")
        self.log(f"  Global min:   {all_imgs.min().item():.6f}")
        self.log(f"  Global max:   {all_imgs.max().item():.6f}")
        
        # Check for suspicious patterns
        if all_imgs.std() < 0.01:
            self.log(f"  ⚠️  WARNING: Very low variance - images might be blank/uniform!")
        if all_imgs.mean() < 0.1 or all_imgs.mean() > 0.9:
            self.log(f"  ⚠️  WARNING: Mean very close to 0 or 1 - check brightness!")
        
    def _inspect_vertices(self, vertices, obj_path):
        """Inspect vertex tensor"""
        self.log("\n" + "="*70)
        self.log("🎯 OUTPUT Y: VERTICES (Ground Truth)")
        self.log("="*70)
        
        self.log(f"\nSource: {obj_path}")
        self.log(f"  Shape:        {tuple(vertices.shape)} (N, 3)")
        self.log(f"  Dtype:        {vertices.dtype}")
        self.log(f"  Num vertices: {vertices.shape[0]}")
        
        self.log(f"\n📊 VERTEX STATISTICS:")
        for i, axis in enumerate(['X', 'Y', 'Z']):
            vals = vertices[:, i]
            self.log(f"  {axis}-axis:")
            self.log(f"    Min:  {vals.min().item():.6f}")
            self.log(f"    Max:  {vals.max().item():.6f}")
            self.log(f"    Mean: {vals.mean().item():.6f}")
            self.log(f"    Std:  {vals.std().item():.6f}")
        
        # Check normalization
        distances = torch.norm(vertices, dim=1)
        self.log(f"\n📏 NORMALIZATION CHECK:")
        self.log(f"  Distance from origin (L2 norm):")
        self.log(f"    Min:  {distances.min().item():.6f}")
        self.log(f"    Max:  {distances.max().item():.6f}")
        self.log(f"    Mean: {distances.mean().item():.6f}")
        
        if distances.max() > 2.0:
            self.log(f"  ⚠️  WARNING: Vertices far from origin (not normalized?)")
        elif distances.max() <= 1.01:
            self.log(f"  ✓ Vertices normalized to unit sphere")
        else:
            self.log(f"  ✓ Vertices reasonably scaled")
        
        # Sample vertices
        self.log(f"\n📍 SAMPLE VERTICES:")
        sample_indices = [0, len(vertices)//4, len(vertices)//2, 3*len(vertices)//4, -1]
        for idx in sample_indices:
            v = vertices[idx]
            self.log(f"  Vertex {idx:5d}: [{v[0]:7.4f}, {v[1]:7.4f}, {v[2]:7.4f}] (dist: {torch.norm(v).item():.4f})")
        
        # Check for degenerate vertices
        if (vertices == 0).all(dim=1).any():
            self.log(f"  ⚠️  WARNING: Found vertices at origin (0,0,0)!")
        
        # Check for duplicates
        unique_verts = torch.unique(vertices, dim=0)
        if len(unique_verts) < len(vertices):
            self.log(f"  ⚠️  WARNING: {len(vertices) - len(unique_verts)} duplicate vertices found!")
        else:
            self.log(f"  ✓ All vertices unique")
    
    def _inspect_faces(self, faces):
        """Inspect face tensor"""
        self.log("\n" + "="*70)
        self.log("🔺 OUTPUT Y: FACES (Topology)")
        self.log("="*70)
        
        self.log(f"\n  Shape:     {tuple(faces.shape)} (F, 3)")
        self.log(f"  Dtype:     {faces.dtype}")
        self.log(f"  Num faces: {faces.shape[0]}")
        
        self.log(f"\n📊 FACE STATISTICS:")
        self.log(f"  Min vertex index: {faces.min().item()}")
        self.log(f"  Max vertex index: {faces.max().item()}")
        
        # Sample faces
        self.log(f"\n📍 SAMPLE FACES (vertex indices):")
        sample_indices = [0, len(faces)//4, len(faces)//2, 3*len(faces)//4, -1]
        for idx in sample_indices:
            f = faces[idx]
            self.log(f"  Face {idx:5d}: [{f[0]:5d}, {f[1]:5d}, {f[2]:5d}]")
        
        # Check for degenerate faces
        degenerate = ((faces[:, 0] == faces[:, 1]) | 
                     (faces[:, 1] == faces[:, 2]) | 
                     (faces[:, 0] == faces[:, 2]))
        if degenerate.any():
            self.log(f"  ⚠️  WARNING: {degenerate.sum().item()} degenerate faces (repeated indices)!")
        else:
            self.log(f"  ✓ No degenerate faces")
    
    def _inspect_vertex_colors(self, colors):
        """Inspect vertex colors"""
        self.log("\n" + "="*70)
        self.log("🎨 OUTPUT Y: VERTEX COLORS")
        self.log("="*70)
        
        self.log(f"\n  Shape: {tuple(colors.shape)} (N, 3)")
        self.log(f"  Dtype: {colors.dtype}")
        
        self.log(f"\n📊 COLOR STATISTICS:")
        for i, channel in enumerate(['R', 'G', 'B']):
            vals = colors[:, i]
            self.log(f"  {channel}-channel:")
            self.log(f"    Min:  {vals.min().item():.6f}")
            self.log(f"    Max:  {vals.max().item():.6f}")
            self.log(f"    Mean: {vals.mean().item():.6f}")
        
        # Check range
        if colors.min() < 0 or colors.max() > 1:
            self.log(f"  ⚠️  WARNING: Colors outside [0, 1] range!")
        else:
            self.log(f"  ✓ Colors in correct [0, 1] range")
        
        # Sample colors
        self.log(f"\n📍 SAMPLE COLORS (R, G, B):")
        sample_indices = [0, len(colors)//4, len(colors)//2, 3*len(colors)//4, -1]
        for idx in sample_indices:
            c = colors[idx]
            self.log(f"  Vertex {idx:5d}: [{c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f}]")
        
        # Check if all white/black
        if (colors == 1).all():
            self.log(f"  ⚠️  INFO: All vertices are white (no texture info)")
        elif (colors == 0).all():
            self.log(f"  ⚠️  WARNING: All vertices are black!")
    
    def _inspect_vertex_normals(self, normals):
        """Inspect vertex normals"""
        self.log("\n" + "="*70)
        self.log("📐 OUTPUT Y: VERTEX NORMALS")
        self.log("="*70)
        
        self.log(f"\n  Shape: {tuple(normals.shape)} (N, 3)")
        self.log(f"  Dtype: {normals.dtype}")
        
        # Check normalization
        lengths = torch.norm(normals, dim=1)
        self.log(f"\n📏 NORMAL VECTOR LENGTHS:")
        self.log(f"  Min:  {lengths.min().item():.6f}")
        self.log(f"  Max:  {lengths.max().item():.6f}")
        self.log(f"  Mean: {lengths.mean().item():.6f}")
        
        if (lengths < 0.99).any() or (lengths > 1.01).any():
            self.log(f"  ⚠️  WARNING: Normals not unit length!")
        else:
            self.log(f"  ✓ Normals properly normalized")
        
        # Sample normals
        self.log(f"\n📍 SAMPLE NORMALS (X, Y, Z):")
        sample_indices = [0, len(normals)//4, len(normals)//2, 3*len(normals)//4, -1]
        for idx in sample_indices:
            n = normals[idx]
            length = torch.norm(n).item()
            self.log(f"  Vertex {idx:5d}: [{n[0]:7.4f}, {n[1]:7.4f}, {n[2]:7.4f}] (len: {length:.4f})")
    
    def _inspect_batch_collation(self, samples):
        """Inspect batch collation"""
        self.log("\n" + "="*70)
        self.log("📦 BATCH COLLATION (DataLoader)")
        self.log("="*70)
        
        from load_data import custom_collate_fn
        
        batch = custom_collate_fn(samples)
        
        self.log(f"\nBatch structure:")
        self.log(f"  Keys: {list(batch.keys())}")
        
        self.log(f"\n📸 Batched Images:")
        for view in ['front', 'back', 'left', 'right', 'top', 'bottom']:
            img_batch = batch['images'][view]
            self.log(f"  {view}: {tuple(img_batch.shape)} (B, C, H, W)")
        
        self.log(f"\n🎯 Batched Geometry:")
        self.log(f"  vertices: List of {len(batch['vertices'])} tensors")
        self.log(f"    [0] shape: {tuple(batch['vertices'][0].shape)}")
        self.log(f"  faces: List of {len(batch['faces'])} tensors")
        self.log(f"    [0] shape: {tuple(batch['faces'][0].shape)}")
        
        self.log(f"\n✓ Batch collation successful")
    
    def _inspect_model_forward(self, sample, config):
        """Simulate model forward pass"""
        self.log("\n" + "="*70)
        self.log("🤖 MODEL FORWARD PASS SIMULATION")
        self.log("="*70)
        
        try:
            from models.geometry_model import ImprovedGeometryModel
            is_new = True
        except ImportError:
            from models.geometry_model import GeometryModel
            is_new = False
        
        device = torch.device(config.device)
        
        # Prepare images
        images = {k: v.unsqueeze(0).to(device) for k, v in sample['images'].items()}
        
        self.log(f"\n📥 Model Inputs:")
        self.log(f"  Architecture: {'Multi-scale (NEW)' if is_new else 'Single-scale (OLD)'}")
        self.log(f"  Device: {device}")
        for view in ['front', 'back']:  # Just show 2 for brevity
            self.log(f"  {view}: {tuple(images[view].shape)} on {images[view].device}")
        
        # Create model
        if is_new:
            model = ImprovedGeometryModel(
                freeze_encoder=config.freeze_image_encoder,
                hidden_dim=config.hidden_dim
            ).to(device)
        else:
            model = GeometryModel(
                num_points=config.num_points,
                freeze_encoder=config.freeze_image_encoder,
                hidden_dim=config.hidden_dim
            ).to(device)
        
        model.eval()
        
        self.log(f"\n🔮 Running forward pass...")
        
        with torch.no_grad():
            try:    
                if is_new:
                    mesh_outputs = model(images, return_all_levels=True)
                    
                    self.log(f"\n📤 Model Outputs (Multi-scale):")
                    self.log(f"  Coarse vertices: {tuple(mesh_outputs['coarse'].shape)}")
                    self.log(f"  Mid vertices:    {tuple(mesh_outputs['mid'].shape)}")
                    self.log(f"  Fine vertices:   {tuple(mesh_outputs['fine'].shape)}")
                    self.log(f"  Normals:         {tuple(mesh_outputs['normals'].shape)}")
                    
                    # Inspect fine output
                    pred_verts = mesh_outputs['fine'][0].cpu()
                    self.log(f"\n📊 PREDICTED VERTICES (Fine level):")
                    
                else:
                    pred_verts, pred_normals = model(images)
                    pred_verts = pred_verts[0].cpu()
                    
                    self.log(f"\n📤 Model Outputs:")
                    self.log(f"  Predicted vertices: {tuple(pred_verts.shape)}")
                    self.log(f"  Predicted normals:  {tuple(pred_normals.shape)}")
                    
                    self.log(f"\n📊 PREDICTED VERTICES:")
                
                # Statistics on predictions
                for i, axis in enumerate(['X', 'Y', 'Z']):
                    vals = pred_verts[:, i]
                    self.log(f"  {axis}-axis:")
                    self.log(f"    Min:  {vals.min().item():.6f}")
                    self.log(f"    Max:  {vals.max().item():.6f}")
                    self.log(f"    Mean: {vals.mean().item():.6f}")
                    self.log(f"    Std:  {vals.std().item():.6f}")
                
                # Compare with GT
                gt_verts = sample['vertices']
                self.log(f"\n📏 COMPARISON WITH GROUND TRUTH:")
                self.log(f"  GT vertices:   {gt_verts.shape[0]}")
                self.log(f"  Pred vertices: {pred_verts.shape[0]}")
                
                # Sample both
                if gt_verts.shape[0] > pred_verts.shape[0]:
                    gt_sample = gt_verts[torch.randperm(len(gt_verts))[:len(pred_verts)]]
                else:
                    gt_sample = gt_verts
                
                # Chamfer distance approximation
                dist_matrix = torch.cdist(pred_verts[:100], gt_sample[:100])
                min_dist = dist_matrix.min(dim=1)[0].mean()
                self.log(f"  Approx Chamfer (100 pts): {min_dist.item():.6f}")
                
                self.log(f"\n✓ Forward pass completed successfully")
                
            except Exception as e:
                self.log(f"\n❌ Forward pass failed: {e}")
                import traceback
                self.log(traceback.format_exc())
    
    def _save_report(self):
        """Save inspection report"""
        report_path = Path('data_inspection_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(self.report))
        
        self.log(f"\n{'='*70}")
        self.log(f"📄 Full report saved to: {report_path}")
        self.log(f"{'='*70}")


def main():
    """Main inspection script"""
    import sys
    
    # Parse arguments
    shoe_id = None
    if len(sys.argv) > 1:
        shoe_id = sys.argv[1]
        print(f"Inspecting specific shoe: {shoe_id}\n")
    else:
        print("Inspecting first shoe in dataset\n")
        print("Usage: python inspect_data_pipeline.py [shoe_id]\n")
    
    # Run inspection
    inspector = DataInspector()
    inspector.inspect_complete_pipeline(shoe_id=shoe_id)


if __name__ == "__main__":
    main()