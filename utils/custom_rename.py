#!/usr/bin/env python3
"""
Advanced custom prompt-based image verification and renaming script.
Supports conditional logic: check visual features AND filename mismatch before renaming.
"""

import os
import json
import base64
import requests
from pathlib import Path
from collections import defaultdict
import shutil
import re

# Configuration
INPUT_DIR = "data/input_images"
BACKUP_DIR = "data/backup_images"
OLLAMA_MODEL = "qwen3-vl:2b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

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

def parse_filename(filename):
    """Parse filename to extract shoe ID and current orientation."""
    if not filename.endswith('.png'):
        return None, None
    
    name_without_ext = filename[:-4]  # Remove .png
    parts = name_without_ext.rsplit('_', 1)
    
    if len(parts) != 2:
        return None, None
    
    shoe_id, orientation = parts
    return shoe_id, orientation

def extract_rename_decision(response):
    """
    Extract renaming decision from model response.
    Returns: (should_rename, new_orientation)
    - should_rename: True if model says to rename
    - new_orientation: the new orientation name (or None)
    """
    if not response:
        return False, None
    
    response_lower = response.lower().strip()
    
    # Check for explicit "rename", "change", "should be", etc.
    rename_indicators = [
        'rename', 'change', 'should be', 'needs to be', 'must be',
        'correct name', 'update to', 'modify to'
    ]
    
    # Check for "no change", "correct", "keep as is", etc.
    no_change_indicators = [
        'no change', 'do not change', "don't change", 'correct as is',
        'keep', 'already correct', 'no rename', 'no need'
    ]
    
    # First check if we should NOT rename
    for indicator in no_change_indicators:
        if indicator in response_lower:
            return False, None
    
    # Check if we should rename
    should_rename = any(indicator in response_lower for indicator in rename_indicators)
    
    if not should_rename:
        return False, None
    
    # Try to extract the new orientation
    orientations = ['top', 'bottom', 'left', 'right', 'front', 'back']
    
    # Look for patterns like "change to bottom", "should be top", etc.
    for orientation in orientations:
        # Pattern: "to/be [orientation]"
        if re.search(rf'\b(to|be)\s+{orientation}\b', response_lower):
            return True, orientation
        # Pattern: "bottom" appears after rename indicators
        if re.search(rf'\b{orientation}\b', response_lower):
            return True, orientation
    
    return True, None  # Should rename but couldn't determine orientation

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
    print("✓ Backup created successfully!")
    return True

def process_image_with_rules(image_path, custom_prompt):
    """Process image with custom rules and determine if renaming is needed."""
    filename = os.path.basename(image_path)
    shoe_id, current_orientation = parse_filename(filename)
    
    if not shoe_id or not current_orientation:
        print(f"⚠️  Skipping invalid filename: {filename}")
        return None
    
    # Replace placeholders in the custom prompt
    prompt = custom_prompt.replace("<ShoeID>", shoe_id)
    prompt = prompt.replace("<current_orientation>", current_orientation)
    prompt = prompt.replace("<filename>", filename)
    
    print(f"\n{'='*70}")
    print(f"Processing: {filename}")
    print(f"  Shoe ID: {shoe_id}")
    print(f"  Current Orientation: {current_orientation}")
    print(f"{'='*70}")
    
    response = query_ollama_vision(image_path, prompt)
    
    if not response:
        print(f"  ⚠️  No response from model")
        return None
    
    print(f"\n  Model Response:")
    print(f"  {'-'*66}")
    print(f"  {response}")
    print(f"  {'-'*66}")
    
    # Extract decision
    should_rename, new_orientation = extract_rename_decision(response)
    
    if should_rename and new_orientation:
        print(f"\n  ✗ DECISION: Rename needed")
        print(f"    Current: {current_orientation}")
        print(f"    Should be: {new_orientation}")
        return {
            'filename': filename,
            'shoe_id': shoe_id,
            'current_orientation': current_orientation,
            'new_orientation': new_orientation,
            'response': response
        }
    elif should_rename and not new_orientation:
        print(f"\n  ⚠️  DECISION: Model suggests rename but unclear what orientation")
        print(f"    Skipping this file")
        return None
    else:
        print(f"\n  ✓ DECISION: No change needed (correct as is)")
        return None

def main():
    """Main function to process images with custom prompt."""
    # Check if input directory exists
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return
    
    print("=" * 70)
    print("CUSTOM CONDITIONAL IMAGE RENAMING")
    print("=" * 70)
    
    # Get custom prompt from user
    print("\nEnter your custom verification prompt with conditional logic.")
    print("\nAvailable placeholders:")
    print("  <ShoeID>              - Replaced with shoe ID (e.g., '001')")
    print("  <current_orientation> - Replaced with current orientation (e.g., 'bottom')")
    print("  <filename>            - Replaced with full filename")
    print("\nYour prompt should describe:")
    print("  1. What visual features to check")
    print("  2. When the filename is incorrect")
    print("  3. What the correct orientation should be")
    print("\nExample prompt:")
    print('  "Look at this image of shoe <ShoeID>.')
    print('   If the entire bottom sole is clearly visible, this should be named bottom.')
    print('   Current filename is <filename> with orientation <current_orientation>.')
    print('   If the sole is visible AND the filename is NOT bottom, rename it to bottom.')
    print('   Otherwise, keep the current name."')
    print("\n" + "-" * 70)
    
    print("\nEnter your prompt (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line == "" and len(lines) > 0:
            break
        lines.append(line)
    
    custom_prompt = "\n".join(lines).strip()
    
    if not custom_prompt:
        print("Error: No prompt provided.")
        return
    
    print("\n" + "=" * 70)
    print("Your prompt:")
    print("-" * 70)
    print(custom_prompt)
    print("=" * 70)
    
    response = input("\nProceed with this prompt? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Create backup
    print("\n" + "=" * 70)
    print("Creating backup...")
    print("=" * 70)
    create_backup(INPUT_DIR, BACKUP_DIR)
    
    # Get all PNG files
    image_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.png')]
    
    if not image_files:
        print(f"\nNo PNG files found in {INPUT_DIR}")
        return
    
    print(f"\n✓ Found {len(image_files)} images to process.")
    
    # Process images
    print("\n" + "=" * 70)
    print("ANALYZING ALL IMAGES...")
    print("=" * 70)
    
    to_rename = []
    
    for filename in sorted(image_files):
        image_path = os.path.join(INPUT_DIR, filename)
        result = process_image_with_rules(image_path, custom_prompt)
        
        if result:
            to_rename.append(result)
    
    # Summary and confirmation
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("=" * 70)
    print(f"Total images analyzed: {len(image_files)}")
    print(f"Files to rename: {len(to_rename)}")
    print(f"Files correct as-is: {len(image_files) - len(to_rename)}")
    
    # Show detailed rename plan
    if to_rename:
        print("\n" + "=" * 70)
        print("PROPOSED RENAMES:")
        print("=" * 70)
        for i, item in enumerate(to_rename, 1):
            old_name = item['filename']
            new_name = f"{item['shoe_id']}_{item['new_orientation']}.png"
            print(f"\n{i}. {old_name} → {new_name}")
            print(f"   Reason: {item['response'][:100]}...")
        
        print("\n" + "=" * 70)
        response = input(f"\nApply these {len(to_rename)} renames? (y/n): ")
        
        if response.lower() == 'y':
            renamed_count = 0
            errors = []
            
            for item in to_rename:
                old_path = os.path.join(INPUT_DIR, item['filename'])
                new_filename = f"{item['shoe_id']}_{item['new_orientation']}.png"
                new_path = os.path.join(INPUT_DIR, new_filename)
                
                # Check if target filename already exists
                if os.path.exists(new_path) and old_path != new_path:
                    error_msg = f"{new_filename} already exists"
                    print(f"⚠️  {error_msg}. Skipping {item['filename']}")
                    errors.append((item['filename'], error_msg))
                    continue
                
                try:
                    os.rename(old_path, new_path)
                    print(f"✓ Renamed: {item['filename']} → {new_filename}")
                    renamed_count += 1
                except Exception as e:
                    error_msg = str(e)
                    print(f"✗ Error renaming {item['filename']}: {error_msg}")
                    errors.append((item['filename'], error_msg))
            
            print(f"\n{'='*70}")
            print(f"✓ Successfully renamed {renamed_count}/{len(to_rename)} files!")
            if errors:
                print(f"⚠️  {len(errors)} files had errors")
            print(f"{'='*70}")
        else:
            print("\nRenaming cancelled. No files were modified.")
    else:
        print("\n✓ All files are correctly named! No changes needed.")
    
    # Save report
    report = {
        "custom_prompt": custom_prompt,
        "total_images": len(image_files),
        "files_to_rename": len(to_rename),
        "rename_details": to_rename
    }
    
    report_path = "rename_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✓ Detailed report saved to: {report_path}")
    print(f"✓ Backup saved to: {BACKUP_DIR}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()