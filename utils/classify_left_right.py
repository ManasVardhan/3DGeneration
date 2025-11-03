"""
Left/Right Shoe Classifier using Claude API

This script:
1. Finds all images named <ShoeID>_left.png and <ShoeID>_right.png
2. Uses Claude to determine which direction the shoe actually faces
3. Renames files to match reality (swap left/right if needed)

Usage:
    export ANTHROPIC_API_KEY="your-key"
    python classify_left_right.py --data_dir /path/to/All_Inputs_Fixed
"""

import anthropic
import base64
import json
import os
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm
import argparse
import shutil


class LeftRightClassifier:
    """Use Claude API to classify left vs right shoe orientations"""
    
    MODELS_TO_TRY = [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-haiku-4-20250320",
        "claude-3-5-sonnet-20241022",
        "claude-3-haiku-20240307",
    ]
    
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Please set ANTHROPIC_API_KEY environment variable:\n"
                "export ANTHROPIC_API_KEY='your-api-key-here'"
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.working_model = None
        
        print("🔍 Finding available Claude model...")
        self._find_working_model()
    
    def _find_working_model(self):
        """Try different models until we find one that works"""
        for model in self.MODELS_TO_TRY:
            try:
                print(f"   Trying {model}...", end=" ")
                
                response = self.client.messages.create(
                    model=model,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}]
                )
                
                self.working_model = model
                print(f"✅ Works!")
                print(f"\n✅ Using model: {model}\n")
                return
            
            except anthropic.NotFoundError:
                print("❌ Not available")
                continue
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        raise RuntimeError("❌ No working Claude model found!")
    
    def encode_image(self, image_path):
        """Encode image to base64 for Claude API"""
        with Image.open(image_path) as img:
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            image_bytes = buffer.getvalue()
            
            return base64.standard_b64encode(image_bytes).decode('utf-8')
    
    def classify_left_right(self, left_path, right_path):
        """
        Classify which image shows left-facing vs right-facing shoe
        
        Args:
            left_path: Path to image labeled as "left"
            right_path: Path to image labeled as "right"
        
        Returns:
            dict: {'left_image_shows': 'left'|'right', 'right_image_shows': 'left'|'right'}
        """
        # Encode images
        left_b64 = self.encode_image(left_path)
        right_b64 = self.encode_image(right_path)
        
        # Create prompt
        prompt = """I have 2 images of the same shoe from different angles.

Image 1 is labeled "left"
Image 2 is labeled "right"

However, these labels may be WRONG. Your job is to determine which direction each shoe ACTUALLY faces.

For each image, determine if the shoe faces:
- "left" (shoe points/faces to the LEFT, toe on left side)
- "right" (shoe points/faces to the RIGHT, toe on right side)

Look carefully at:
- Which direction the toe/front of the shoe points
- The orientation of the shoe's profile
- LEFT means: toe on left, heel on right
- RIGHT means: toe on right, heel on left

Respond with ONLY a JSON object:
{
  "left_image_shows": "left" or "right",
  "right_image_shows": "left" or "right"
}

Be precise. Study the shoe orientation carefully.

JSON response:"""
        
        # Create message with images
        message_content = [
            {
                "type": "text",
                "text": "Image 1 (labeled as 'left'):"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": left_b64
                }
            },
            {
                "type": "text",
                "text": "Image 2 (labeled as 'right'):"
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": right_b64
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
        
        # Call Claude
        try:
            message = self.client.messages.create(
                model=self.working_model,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": message_content
                }]
            )
            
            response_text = message.content[0].text.strip()
            
            # Extract JSON
            json_text = response_text
            
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0].strip()
            
            if '{' in json_text and '}' in json_text:
                start = json_text.index('{')
                end = json_text.rindex('}') + 1
                json_text = json_text[start:end]
            
            try:
                classification = json.loads(json_text)
            except json.JSONDecodeError:
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from: {response_text[:200]}...")
            
            return classification
        
        except Exception as e:
            print(f"❌ Error: {e}")
            return None


def process_left_right_images(data_dir, backup=True, dry_run=False):
    """
    Process all left/right images in directory
    
    Args:
        data_dir: Directory containing images
        backup: Create backup before renaming
        dry_run: Don't actually rename, just show what would happen
    """
    data_dir = Path(data_dir)
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          LEFT/RIGHT SHOE CLASSIFIER WITH CLAUDE API                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Initialize classifier
    classifier = LeftRightClassifier()
    
    # Find all left/right image pairs
    print("📦 Finding left/right image pairs...")
    
    left_files = {}
    right_files = {}
    
    for ext in ['png', 'jpg', 'jpeg']:
        for left_file in data_dir.glob(f'*_left.{ext}'):
            shoe_id = left_file.stem.rsplit('_left', 1)[0]
            left_files[shoe_id] = left_file
        
        for right_file in data_dir.glob(f'*_right.{ext}'):
            shoe_id = right_file.stem.rsplit('_right', 1)[0]
            right_files[shoe_id] = right_file
    
    # Find pairs (shoes with both left and right)
    shoe_ids = set(left_files.keys()) & set(right_files.keys())
    shoe_ids = sorted(shoe_ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    
    print(f"   Found {len(shoe_ids)} shoes with both left & right images")
    print(f"   Found {len(left_files)} left images total")
    print(f"   Found {len(right_files)} right images total")
    print()
    
    if not shoe_ids:
        print("❌ No shoe pairs found!")
        print("   Looking for files like: <ShoeID>_left.png and <ShoeID>_right.png")
        return
    
    # Estimate cost
    cost_per_pair = 0.008  # ~$0.008 per pair (2 images)
    estimated_cost = len(shoe_ids) * cost_per_pair
    
    print(f"💰 Estimated API cost: ${estimated_cost:.2f}")
    print(f"   (${cost_per_pair:.3f} per shoe × {len(shoe_ids)} shoes)")
    print()
    
    if dry_run:
        print("🔍 DRY RUN MODE - No files will be renamed")
        print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    # Create backup directory
    if backup and not dry_run:
        backup_dir = data_dir / 'backup_before_rename'
        backup_dir.mkdir(exist_ok=True)
        print(f"\n📦 Creating backup in: {backup_dir}")
    
    # Process each pair
    print("\n🚀 Classifying shoe orientations...\n")
    
    results = {
        'correct': 0,
        'swapped': 0,
        'errors': 0,
        'details': []
    }
    
    for shoe_id in tqdm(shoe_ids, desc="Processing shoes"):
        left_path = left_files[shoe_id]
        right_path = right_files[shoe_id]
        
        try:
            classification = classifier.classify_left_right(left_path, right_path)
            
            if not classification:
                results['errors'] += 1
                results['details'].append({
                    'shoe_id': shoe_id,
                    'status': 'error',
                    'message': 'Classification failed'
                })
                continue
            
            left_shows = classification.get('left_image_shows', '').lower()
            right_shows = classification.get('right_image_shows', '').lower()
            
            # Check if swap is needed
            needs_swap = (left_shows == 'right' and right_shows == 'left')
            
            if needs_swap:
                results['swapped'] += 1
                action = 'SWAP'
                
                if not dry_run:
                    # Backup originals
                    if backup:
                        shutil.copy2(left_path, backup_dir / left_path.name)
                        shutil.copy2(right_path, backup_dir / right_path.name)
                    
                    # Swap files using temp
                    temp_path = left_path.with_suffix('.temp' + left_path.suffix)
                    
                    # Rename: left → temp, right → left, temp → right
                    left_path.rename(temp_path)
                    right_path.rename(left_path)
                    temp_path.rename(right_path)
                
                results['details'].append({
                    'shoe_id': shoe_id,
                    'status': 'swapped',
                    'left_shows': left_shows,
                    'right_shows': right_shows
                })
            else:
                results['correct'] += 1
                action = 'OK'
                
                results['details'].append({
                    'shoe_id': shoe_id,
                    'status': 'correct',
                    'left_shows': left_shows,
                    'right_shows': right_shows
                })
        
        except Exception as e:
            results['errors'] += 1
            results['details'].append({
                'shoe_id': shoe_id,
                'status': 'error',
                'message': str(e)
            })
    
    # Save detailed results
    results_file = data_dir / 'left_right_classification_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Already correct: {results['correct']}/{len(shoe_ids)}")
    print(f"🔄 Swapped (were wrong): {results['swapped']}/{len(shoe_ids)}")
    print(f"❌ Errors: {results['errors']}/{len(shoe_ids)}")
    print()
    print(f"📊 Detailed results: {results_file}")
    
    if backup and not dry_run and results['swapped'] > 0:
        print(f"📦 Backup of originals: {backup_dir}/")
    
    print("="*70)
    
    # Show some examples of swapped shoes
    if results['swapped'] > 0:
        print("\n📋 Swapped shoes:")
        swapped = [d for d in results['details'] if d['status'] == 'swapped']
        for item in swapped[:10]:
            print(f"   • Shoe {item['shoe_id']}: left showed {item['left_shows']}, right showed {item['right_shows']}")
        if len(swapped) > 10:
            print(f"   ... and {len(swapped) - 10} more")
    
    print("\n✅ DONE!")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify and rename left/right shoe images'
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Directory containing images (e.g., /Users/.../All_Inputs_Fixed)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup (not recommended)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would happen without actually renaming'
    )
    parser.add_argument(
        '--api_key',
        help='Anthropic API key (or use ANTHROPIC_API_KEY env var)'
    )
    
    args = parser.parse_args()
    
    # Set API key if provided
    if args.api_key:
        os.environ['ANTHROPIC_API_KEY'] = args.api_key
    
    process_left_right_images(
        data_dir=args.data_dir,
        backup=not args.no_backup,
        dry_run=args.dry_run
    )