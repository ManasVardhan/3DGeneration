"""
Automatic View Classification Using Claude API (with model fallback)

This version tries multiple Claude models until it finds one that works.
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


class ClaudeViewClassifier:
    """Use Claude API to classify shoe view orientations"""
    
    # List of models to try, in order of preference (Claude 4 naming)
    MODELS_TO_TRY = [
        "claude-sonnet-4-20250514",     # Latest Sonnet 4
        "claude-opus-4-20250514",       # Opus 4  
        "claude-haiku-4-20250320",      # Haiku 4
        "claude-3-5-sonnet-20241022",   # Fallback to Claude 3.5
        "claude-3-haiku-20240307",      # Haiku 3 (cheapest)
    ]
    
    def __init__(self, api_key=None):
        if api_key is None:
            api_key = os.environ.get('ANTHROPIC_API_KEY')
        
        if not api_key:
            raise ValueError(
                "Please set ANTHROPIC_API_KEY environment variable:\n"
                "export ANTHROPIC_API_KEY='your-api-key-here'\n"
                "Get your API key from: https://console.anthropic.com/"
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.view_names = ['front', 'back', 'left', 'right', 'top', 'bottom']
        self.working_model = None
        
        # Find a working model
        print("🔍 Finding available Claude model...")
        self._find_working_model()
    
    def _find_working_model(self):
        """Try different models until we find one that works"""
        for model in self.MODELS_TO_TRY:
            try:
                print(f"   Trying {model}...", end=" ")
                
                # Test with a simple message
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
        
        raise RuntimeError(
            "❌ No working Claude model found!\n"
            "Please check:\n"
            "  1. Your API key is correct\n"
            "  2. Your account has access to Claude models\n"
            "  3. Visit: https://console.anthropic.com/settings/limits"
        )
    
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
    
    def classify_shoe_views(self, shoe_id, images_dir):
        """Use Claude to classify all 6 views of a single shoe"""
        images_dir = Path(images_dir)
        
        # Find all images for this shoe
        image_files = {}
        for view in self.view_names:
            for ext in ['png', 'jpg', 'jpeg']:
                img_path = images_dir / f'{shoe_id}_{view}.{ext}'
                if img_path.exists():
                    image_files[view] = img_path
                    break
        
        if len(image_files) != 6:
            print(f"⚠️  Warning: Found only {len(image_files)}/6 images for shoe {shoe_id}")
        
        # Encode all images
        encoded_images = {}
        for view, path in image_files.items():
            encoded_images[view] = self.encode_image(path)
        
        # Create prompt for Claude
        image_content = []
        for i, (view, image_b64) in enumerate(encoded_images.items(), 1):
            image_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_b64
                }
            })
            image_content.append({
                "type": "text",
                "text": f"Image {i} (labeled as '{view}'):"
            })
        
        prompt = f"""I have 6 images of the same shoe, labeled as: {', '.join(image_files.keys())}

However, these labels may be INCORRECT. Your job is to identify what each image ACTUALLY shows.

For each image, determine if it shows:
- "front" (toe/front of shoe)
- "back" (heel/back of shoe)
- "left" (left side view)
- "right" (right side view)
- "top" (top-down view, see laces/tongue)
- "bottom" (bottom-up view, see sole)

Respond with ONLY a JSON object mapping the image labels to their actual orientations. For example:
{{
  "front": "back",
  "back": "top",
  "left": "front",
  "right": "left",
  "top": "right",
  "bottom": "bottom"
}}

Be precise - look at distinctive features:
- Front: Can see toe box, front of shoe
- Back: Can see heel counter, back of shoe
- Left/Right: Side profile view
- Top: Looking down at shoe, see laces
- Bottom: Looking up at shoe, see sole/tread

JSON response:"""
        
        # Call Claude API with working model
        try:
            message = self.client.messages.create(
                model=self.working_model,
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": image_content + [{"type": "text", "text": prompt}]
                }]
            )
            
            response_text = message.content[0].text.strip()
            
            # Extract JSON - handle various Claude response formats
            json_text = response_text
            
            # Method 1: Look for JSON in markdown code blocks
            if '```json' in response_text:
                json_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Method 2: Extract JSON object with curly braces
            if '{' in json_text and '}' in json_text:
                start = json_text.index('{')
                end = json_text.rindex('}') + 1
                json_text = json_text[start:end]
            
            # Try to parse
            try:
                classification = json.loads(json_text)
            except json.JSONDecodeError:
                # Last resort: use regex to find JSON
                import re
                json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
                if json_match:
                    classification = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON from: {response_text[:200]}...")
            
            return classification
        
        except Exception as e:
            print(f"❌ Error classifying shoe {shoe_id}: {e}")
            return None
    
    def reorder_shoe_images(self, shoe_id, classification, images_dir, output_dir):
        """Reorder images based on Claude's classification"""
        images_dir = Path(images_dir)
        output_dir = Path(output_dir) / str(shoe_id)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create reverse mapping
        reverse_map = {}
        for labeled_view, actual_view in classification.items():
            reverse_map[actual_view] = labeled_view
        
        # Copy images to correct positions
        for actual_view in self.view_names:
            if actual_view not in reverse_map:
                continue
            
            labeled_view = reverse_map[actual_view]
            
            # Find source image
            source_path = None
            for ext in ['png', 'jpg', 'jpeg']:
                test_path = images_dir / f'{shoe_id}_{labeled_view}.{ext}'
                if test_path.exists():
                    source_path = test_path
                    break
            
            if source_path:
                dest_path = output_dir / f'{actual_view}.png'
                
                with Image.open(source_path) as img:
                    img = img.convert('RGB')
                    img.save(dest_path, 'PNG')
        
        return True


def process_all_shoes(images_dir, output_dir='data/reordered_images', api_key=None):
    """Process all shoes using Claude API"""
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║          AUTOMATIC VIEW CLASSIFICATION WITH CLAUDE API              ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Initialize classifier (will auto-detect working model)
    try:
        classifier = ClaudeViewClassifier(api_key=api_key)
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Find all unique shoe IDs
    image_files = list(images_dir.glob('*_*.png')) + \
                  list(images_dir.glob('*_*.jpg')) + \
                  list(images_dir.glob('*_*.jpeg'))
    
    shoe_ids = set()
    for img_file in image_files:
        parts = img_file.stem.rsplit('_', 1)
        if len(parts) == 2:
            shoe_ids.add(parts[0])
    
    shoe_ids = sorted(shoe_ids, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
    
    print(f"📦 Found {len(shoe_ids)} shoes to process")
    print(f"📁 Input: {images_dir}")
    print(f"📁 Output: {output_dir}")
    print()
    
    # Estimate cost
    cost_per_shoe = 0.012
    estimated_cost = len(shoe_ids) * cost_per_shoe
    
    print(f"💰 Estimated API cost: ${estimated_cost:.2f}")
    print(f"   (${cost_per_shoe:.3f} per shoe × {len(shoe_ids)} shoes)")
    print()
    
    response = input("Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Aborted.")
        return
    
    print("\n🚀 Processing shoes...\n")
    
    # Process each shoe
    results = {}
    successful = 0
    failed = 0
    
    for shoe_id in tqdm(shoe_ids, desc="Classifying views"):
        try:
            classification = classifier.classify_shoe_views(shoe_id, images_dir)
            
            if classification:
                success = classifier.reorder_shoe_images(
                    shoe_id, classification, images_dir, output_dir
                )
                
                if success:
                    results[shoe_id] = {
                        'status': 'success',
                        'classification': classification
                    }
                    successful += 1
                else:
                    results[shoe_id] = {
                        'status': 'failed',
                        'error': 'Reordering failed'
                    }
                    failed += 1
            else:
                results[shoe_id] = {
                    'status': 'failed',
                    'error': 'Classification failed'
                }
                failed += 1
        
        except Exception as e:
            results[shoe_id] = {
                'status': 'failed',
                'error': str(e)
            }
            failed += 1
    
    # Save results
    results_file = output_dir / 'classification_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Successful: {successful}/{len(shoe_ids)}")
    print(f"❌ Failed: {failed}/{len(shoe_ids)}")
    print(f"\n📊 Results saved to: {results_file}")
    print(f"📁 Reordered images in: {output_dir}/")
    print("="*70)
    
    if successful > 0:
        print("\n✅ DONE! Now you can train with properly oriented views:")
        print("   python train_geometry.py")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Classify shoe view orientations using Claude API'
    )
    parser.add_argument(
        '--data_dir',
        default='data/input_images',
        help='Directory containing input images'
    )
    parser.add_argument(
        '--output_dir',
        default='data/reordered_images',
        help='Directory for reordered images'
    )
    parser.add_argument(
        '--api_key',
        help='Anthropic API key (or set ANTHROPIC_API_KEY env var)'
    )
    parser.add_argument(
        '--single',
        type=str,
        help='Process single shoe ID for testing'
    )
    
    args = parser.parse_args()
    
    if args.single:
        classifier = ClaudeViewClassifier(api_key=args.api_key)
        
        classification = classifier.classify_shoe_views(args.single, args.data_dir)
        print(f"\nClassification result:")
        print(json.dumps(classification, indent=2))
        
        classifier.reorder_shoe_images(
            args.single, classification, args.data_dir, args.output_dir
        )
        print(f"\n✅ Reordered images saved to: {args.output_dir}/{args.single}/")
    else:
        process_all_shoes(
            images_dir=args.data_dir,
            output_dir=args.output_dir,
            api_key=args.api_key
        )