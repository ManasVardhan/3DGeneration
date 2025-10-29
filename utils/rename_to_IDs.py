"""
Rename Shoe Folders and Images from Names to IDs

This script:
1. Scans data/Completed/ for shoe folders (original names)
2. Assigns each shoe a unique numeric ID (1, 2, 3, ...)
3. Renames folders in data/Completed/
4. Renames corresponding images in data/input_images/
5. Creates a mapping file for reference

Usage:
    python rename_shoes_to_ids.py --data_dir data --dry_run  # Preview changes
    python rename_shoes_to_ids.py --data_dir data             # Apply changes
"""

import os
import json
import shutil
from pathlib import Path
import argparse


def scan_shoes(data_dir):
    """
    Scan for all shoe folders in data/Completed/
    
    Returns:
        list of shoe names (folder names)
    """
    completed_dir = Path(data_dir) / 'Completed'
    
    if not completed_dir.exists():
        raise FileNotFoundError(f"Directory not found: {completed_dir}")
    
    # Get all subdirectories
    shoe_names = []
    for item in completed_dir.iterdir():
        if item.is_dir():
            shoe_names.append(item.name)
    
    # Sort alphabetically for consistent ID assignment
    shoe_names.sort()
    
    print(f"Found {len(shoe_names)} shoes in {completed_dir}")
    return shoe_names


def create_mapping(shoe_names, start_id=1):
    """
    Create mapping from shoe names to IDs
    
    Args:
        shoe_names: list of original shoe names
        start_id: starting ID number (default: 1)
    
    Returns:
        dict: {shoe_name: shoe_id}
    """
    mapping = {}
    for i, name in enumerate(shoe_names, start=start_id):
        mapping[name] = i
    
    return mapping


def preview_changes(mapping, data_dir):
    """
    Preview all changes that will be made
    """
    data_dir = Path(data_dir)
    completed_dir = data_dir / 'Completed'
    input_images_dir = data_dir / 'input_images'
    
    print("\n" + "="*70)
    print("PREVIEW OF CHANGES")
    print("="*70 + "\n")
    
    total_folders = 0
    total_images = 0
    
    for shoe_name, shoe_id in sorted(mapping.items(), key=lambda x: x[1]):
        print(f"Shoe '{shoe_name}' → ID {shoe_id}")
        
        # Check folder
        old_folder = completed_dir / shoe_name
        new_folder = completed_dir / str(shoe_id)
        
        if old_folder.exists():
            print(f"  Folder: {old_folder.name}/ → {new_folder.name}/")
            
            # List files in folder
            obj_files = list(old_folder.glob('*.obj'))
            if obj_files:
                for obj_file in obj_files:
                    print(f"    Contains: {obj_file.name}")
            
            total_folders += 1
        else:
            print(f"  Folder: {old_folder} NOT FOUND!")
        
        # Check images
        if input_images_dir.exists():
            old_images = list(input_images_dir.glob(f'{shoe_name}_*.png'))
            
            if old_images:
                print(f"  Images ({len(old_images)}):")
                for old_img in sorted(old_images):
                    # Extract orientation from filename
                    orientation = old_img.stem.replace(f'{shoe_name}_', '')
                    new_img_name = f'{shoe_id}_{orientation}.png'
                    print(f"    {old_img.name} → {new_img_name}")
                    total_images += 1
            else:
                print(f"  Images: No images found for {shoe_name}")
        
        print()
    
    print("="*70)
    print(f"Total: {total_folders} folders, {total_images} images will be renamed")
    print("="*70 + "\n")


def apply_changes(mapping, data_dir, backup=True):
    """
    Apply the renaming changes
    
    Args:
        mapping: dict {shoe_name: shoe_id}
        data_dir: path to data directory
        backup: whether to create backup of mapping
    """
    data_dir = Path(data_dir)
    completed_dir = data_dir / 'Completed'
    input_images_dir = data_dir / 'input_images'
    
    print("\n" + "="*70)
    print("APPLYING CHANGES")
    print("="*70 + "\n")
    
    # Save mapping file
    mapping_file = data_dir / 'shoe_name_to_id_mapping.json'
    
    # Create reverse mapping for reference
    reverse_mapping = {v: k for k, v in mapping.items()}
    mapping_data = {
        'name_to_id': mapping,
        'id_to_name': reverse_mapping
    }
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)
    
    print(f"✓ Saved mapping to {mapping_file}\n")
    
    # Rename in reverse ID order to avoid conflicts
    # (e.g., if folder "1" exists and we want to rename "shoe_x" to "1")
    sorted_items = sorted(mapping.items(), key=lambda x: x[1], reverse=True)
    
    errors = []
    folders_renamed = 0
    images_renamed = 0
    
    for shoe_name, shoe_id in sorted_items:
        print(f"Processing '{shoe_name}' → ID {shoe_id}")
        
        # Rename folder
        old_folder = completed_dir / shoe_name
        new_folder = completed_dir / str(shoe_id)
        
        if old_folder.exists():
            try:
                # Check if target already exists
                if new_folder.exists():
                    # Temporary name to avoid conflict
                    temp_folder = completed_dir / f'temp_{shoe_id}_{shoe_name}'
                    old_folder.rename(temp_folder)
                    old_folder = temp_folder
                
                old_folder.rename(new_folder)
                print(f"  ✓ Renamed folder: {shoe_name}/ → {shoe_id}/")
                folders_renamed += 1
            except Exception as e:
                error_msg = f"  ✗ Error renaming folder {old_folder}: {e}"
                print(error_msg)
                errors.append(error_msg)
        else:
            print(f"  - Folder not found: {old_folder}")
        
        # Rename images
        if input_images_dir.exists():
            old_images = list(input_images_dir.glob(f'{shoe_name}_*.png'))
            
            for old_img in old_images:
                try:
                    # Extract orientation
                    orientation = old_img.stem.replace(f'{shoe_name}_', '')
                    new_img = input_images_dir / f'{shoe_id}_{orientation}.png'
                    
                    # Check if target exists
                    if new_img.exists():
                        # Temporary name
                        temp_img = input_images_dir / f'temp_{shoe_id}_{orientation}.png'
                        old_img.rename(temp_img)
                        old_img = temp_img
                    
                    old_img.rename(new_img)
                    print(f"  ✓ Renamed image: {old_img.name} → {new_img.name}")
                    images_renamed += 1
                except Exception as e:
                    error_msg = f"  ✗ Error renaming image {old_img}: {e}"
                    print(error_msg)
                    errors.append(error_msg)
        
        print()
    
    # Clean up any temporary files
    for temp_folder in completed_dir.glob('temp_*'):
        try:
            actual_id = temp_folder.name.split('_')[1]
            final_folder = completed_dir / actual_id
            temp_folder.rename(final_folder)
            print(f"✓ Cleaned up temp folder: {temp_folder.name} → {actual_id}/")
        except Exception as e:
            print(f"✗ Error cleaning up {temp_folder}: {e}")
    
    if input_images_dir.exists():
        for temp_img in input_images_dir.glob('temp_*.png'):
            try:
                parts = temp_img.stem.split('_')
                actual_id = parts[1]
                orientation = '_'.join(parts[2:])
                final_img = input_images_dir / f'{actual_id}_{orientation}.png'
                temp_img.rename(final_img)
                print(f"✓ Cleaned up temp image: {temp_img.name} → {final_img.name}")
            except Exception as e:
                print(f"✗ Error cleaning up {temp_img}: {e}")
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✓ Folders renamed: {folders_renamed}")
    print(f"✓ Images renamed: {images_renamed}")
    
    if errors:
        print(f"\n⚠ Errors encountered: {len(errors)}")
        for error in errors:
            print(error)
    else:
        print("\n✓ All operations completed successfully!")
    
    print("="*70 + "\n")
    
    return folders_renamed, images_renamed, len(errors)


def verify_changes(mapping, data_dir):
    """
    Verify that all changes were applied correctly
    """
    data_dir = Path(data_dir)
    completed_dir = data_dir / 'Completed'
    input_images_dir = data_dir / 'input_images'
    
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70 + "\n")
    
    issues = []
    
    for shoe_name, shoe_id in mapping.items():
        # Check folder exists with new ID
        new_folder = completed_dir / str(shoe_id)
        if not new_folder.exists():
            issues.append(f"Folder {shoe_id}/ not found (was: {shoe_name})")
        
        # Check old folder doesn't exist
        old_folder = completed_dir / shoe_name
        if old_folder.exists():
            issues.append(f"Old folder {shoe_name}/ still exists!")
        
        # Check images
        if input_images_dir.exists():
            # Check old images don't exist
            old_images = list(input_images_dir.glob(f'{shoe_name}_*.png'))
            if old_images:
                issues.append(f"Old images found: {[img.name for img in old_images]}")
    
    if issues:
        print("⚠ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✓ All changes verified successfully!")
    
    print("="*70 + "\n")
    
    return len(issues) == 0


def main():
    parser = argparse.ArgumentParser(
        description='Rename shoe folders and images from names to numeric IDs'
    )
    parser.add_argument(
        '--data_dir',
        default='data',
        help='Path to data directory (default: data)'
    )
    parser.add_argument(
        '--start_id',
        type=int,
        default=1,
        help='Starting ID number (default: 1)'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Preview changes without applying them'
    )
    parser.add_argument(
        '--verify_only',
        action='store_true',
        help='Only verify existing changes (requires mapping file)'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        return
    
    # Verify only mode
    if args.verify_only:
        mapping_file = data_dir / 'shoe_name_to_id_mapping.json'
        if not mapping_file.exists():
            print(f"Error: Mapping file not found: {mapping_file}")
            print("Run without --verify_only first to create mapping and rename.")
            return
        
        with open(mapping_file) as f:
            mapping_data = json.load(f)
        
        mapping = mapping_data['name_to_id']
        verify_changes(mapping, data_dir)
        return
    
    # Scan shoes
    try:
        shoe_names = scan_shoes(data_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not shoe_names:
        print("No shoes found!")
        return
    
    # Create mapping
    mapping = create_mapping(shoe_names, start_id=args.start_id)
    
    # Preview changes
    preview_changes(mapping, data_dir)
    
    if args.dry_run:
        print("DRY RUN: No changes applied.")
        print("\nTo apply changes, run without --dry_run flag:")
        print(f"  python rename_shoes_to_ids.py --data_dir {args.data_dir}")
        return
    
    # Confirm with user
    print("\n⚠ WARNING: This will rename folders and files!")
    print("A mapping file will be saved for reference.")
    response = input("\nProceed with renaming? (yes/no): ")
    
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Apply changes
    folders_renamed, images_renamed, errors = apply_changes(
        mapping, 
        data_dir, 
        backup=True
    )
    
    # Verify
    if errors == 0:
        verify_changes(mapping, data_dir)
    
    print("\n✓ Process complete!")
    print(f"\nMapping saved to: {data_dir}/shoe_name_to_id_mapping.json")
    print("You can use this file to look up original shoe names.")


if __name__ == '__main__':
    main()