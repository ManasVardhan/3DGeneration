#!/usr/bin/env python3
"""
Script to verify and correct shoe image orientations using Ollama vision model.
"""

import os
import json
import base64
import requests
from pathlib import Path
from collections import defaultdict
import shutil

# Configuration
INPUT_DIR = "data/input_images"
BACKUP_DIR = "data/backup_images"
OLLAMA_MODEL = "qwen3-vl:2b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"
VALID_ORIENTATIONS = ["top", "bottom", "left", "right", "front", "back"]

def encode_image_to_base64(image_path):
    """Encode image to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def query_ollama_vision(image_path, prompt):
    """Query Ollama vision model with an image."""
    try:
        image_base64 = encode_image_to_base64(image_path)
        
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        return result.get("response", "").strip()
    
    except Exception as e:
        print(f"Error querying Ollama for {image_path}: {e}")
        return None

def extract_orientation_from_response(response):
    """Extract orientation from model response."""
    if not response:
        return None
    
    response_lower = response.lower()
    
    # Look for orientation keywords in the response
    for orientation in VALID_ORIENTATIONS:
        if orientation in response_lower:
            return orientation
    
    return None

def parse_filename(filename):
    """Parse filename to extract shoe ID and orientation."""
    if not filename.endswith('.png'):
        return None, None
    
    name_without_ext = filename[:-4]  # Remove .png
    parts = name_without_ext.rsplit('_', 1)
    
    if len(parts) != 2:
        return None, None
    
    shoe_id, orientation = parts
    
    if orientation not in VALID_ORIENTATIONS:
        print(f"Warning: Invalid orientation '{orientation}' in filename: {filename}")
        return shoe_id, orientation
    
    return shoe_id, orientation

def verify_and_correct_orientation(image_path):
    """Verify image orientation and return correct orientation."""
    filename = os.path.basename(image_path)
    shoe_id, current_orientation = parse_filename(filename)
    
    if not shoe_id or not current_orientation:
        print(f"Skipping invalid filename: {filename}")
        return None, None, None
    
    prompt = f"""Look at this shoe image carefully. From what perspective or angle is this shoe photographed?

The possible orientations are:
- top: view from above, looking down at the shoe
- bottom: view from below, showing the sole/bottom of the shoe
- left: view from the left side of the shoe
- right: view from the right side of the shoe
- front: view from the front/toe of the shoe
- back: view from the back/heel of the shoe

Respond with just ONE word indicating the orientation: top, bottom, left, right, front, or back."""

    print(f"Checking {filename} (current: {current_orientation})...", end=" ")
    
    response = query_ollama_vision(image_path, prompt)
    detected_orientation = extract_orientation_from_response(response)
    
    if not detected_orientation:
        print(f"⚠️  Could not detect orientation. Response: {response}")
        return shoe_id, current_orientation, None
    
    if detected_orientation == current_orientation:
        print(f"✓ Correct ({detected_orientation})")
        return shoe_id, current_orientation, current_orientation
    else:
        print(f"✗ Wrong! Detected: {detected_orientation}, Current: {current_orientation}")
        return shoe_id, current_orientation, detected_orientation

def create_backup(input_dir, backup_dir):
    """Create backup of input directory."""
    if os.path.exists(backup_dir):
        print(f"Backup directory already exists: {backup_dir}")
        response = input("Overwrite existing backup? (y/n): ")
        if response.lower() != 'y':
            print("Skipping backup creation.")
            return False
        shutil.rmtree(backup_dir)
    
    print(f"Creating backup: {backup_dir}")
    shutil.copytree(input_dir, backup_dir)
    print("Backup created successfully!")
    return True

def main():
    """Main function to process all images."""
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return
    
    # Create backup
    print("=" * 60)
    print("Creating backup of images...")
    print("=" * 60)
    create_backup(INPUT_DIR, BACKUP_DIR)
    
    # Get all PNG files
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]
    
    if not image_files:
        print(f"No PNG files found in {INPUT_DIR}")
        return
    
    print(f"\nFound {len(image_files)} images to process.\n")
    
    # Process images
    print("=" * 60)
    print("Verifying orientations...")
    print("=" * 60)
    
    corrections = []
    shoe_orientations = defaultdict(set)
    
    for filename in sorted(image_files):
        image_path = os.path.join(INPUT_DIR, filename)
        shoe_id, current_orient, detected_orient = verify_and_correct_orientation(image_path)
        
        if shoe_id and detected_orient:
            shoe_orientations[shoe_id].add(detected_orient)
            
            if current_orient != detected_orient:
                corrections.append({
                    'shoe_id': shoe_id,
                    'old_name': filename,
                    'new_name': f"{shoe_id}_{detected_orient}.png",
                    'old_orientation': current_orient,
                    'new_orientation': detected_orient
                })
    
    # Apply corrections
    if corrections:
        print("\n" + "=" * 60)
        print(f"Found {len(corrections)} images with incorrect orientations")
        print("=" * 60)
        
        for correction in corrections:
            print(f"  {correction['old_name']} → {correction['new_name']}")
        
        response = input("\nApply these corrections? (y/n): ")
        
        if response.lower() == 'y':
            for correction in corrections:
                old_path = os.path.join(INPUT_DIR, correction['old_name'])
                new_path = os.path.join(INPUT_DIR, correction['new_name'])
                
                # Check if target filename already exists
                if os.path.exists(new_path) and old_path != new_path:
                    print(f"Warning: {correction['new_name']} already exists. Skipping {correction['old_name']}")
                    continue
                
                os.rename(old_path, new_path)
                print(f"✓ Renamed: {correction['old_name']} → {correction['new_name']}")
            
            print(f"\n✓ Successfully renamed {len(corrections)} files!")
        else:
            print("Corrections not applied.")
    else:
        print("\n✓ All orientations are correct!")
    
    # Check completeness
    print("\n" + "=" * 60)
    print("Checking completeness (all 6 orientations per shoe)...")
    print("=" * 60)
    
    # Re-scan directory after potential renames
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]
    shoe_orientations = defaultdict(set)
    
    for filename in image_files:
        shoe_id, orientation = parse_filename(filename)
        if shoe_id and orientation in VALID_ORIENTATIONS:
            shoe_orientations[shoe_id].add(orientation)
    
    complete_shoes = []
    incomplete_shoes = []
    
    for shoe_id in sorted(shoe_orientations.keys()):
        orientations = shoe_orientations[shoe_id]
        missing = set(VALID_ORIENTATIONS) - orientations
        
        if len(orientations) == 6:
            complete_shoes.append(shoe_id)
            print(f"✓ Shoe {shoe_id}: Complete (6/6 orientations)")
        else:
            incomplete_shoes.append((shoe_id, missing))
            print(f"✗ Shoe {shoe_id}: Incomplete ({len(orientations)}/6 orientations)")
            print(f"  Missing: {', '.join(sorted(missing))}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total shoes: {len(shoe_orientations)}")
    print(f"Complete shoes: {len(complete_shoes)}")
    print(f"Incomplete shoes: {len(incomplete_shoes)}")
    
    if corrections:
        print(f"Corrections made: {len(corrections)}")
    
    # Save report
    report = {
        "total_shoes": len(shoe_orientations),
        "complete_shoes": complete_shoes,
        "incomplete_shoes": [
            {"shoe_id": shoe_id, "missing_orientations": list(missing)}
            for shoe_id, missing in incomplete_shoes
        ],
        "corrections_made": corrections if corrections else []
    }
    
    report_path = "shoe_orientation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to: {report_path}")
    print(f"Backup saved to: {BACKUP_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()