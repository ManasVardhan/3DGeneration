"""
Verify Shoe Image Completeness

Checks that all shoes have exactly 6 images (front, back, left, right, top, bottom)
and reports any missing views.

Usage:
    python verify_shoe_completeness.py --data_dir /path/to/images
"""

import argparse
from pathlib import Path
from collections import defaultdict
import json


def verify_shoe_completeness(data_dir, export_report=True):
    """
    Verify that all shoes have complete 6-view image sets
    
    Args:
        data_dir: Directory containing shoe images
        export_report: Save detailed report to JSON
    
    Returns:
        dict with verification results
    """
    data_dir = Path(data_dir)
    
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║              SHOE IMAGE COMPLETENESS VERIFICATION                    ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print(f"📁 Checking directory: {data_dir}")
    print()
    
    # Expected views
    required_views = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    # Track images by shoe ID
    shoes = defaultdict(lambda: {'views': set(), 'files': {}})
    
    # Find all image files
    print("🔍 Scanning for images...")
    image_extensions = ['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG']
    
    total_files = 0
    for ext in image_extensions:
        for img_file in data_dir.glob(f'*.{ext}'):
            total_files += 1
            
            # Extract shoe ID and view from filename
            # Expected format: <ShoeID>_<view>.ext
            stem = img_file.stem
            
            if '_' not in stem:
                continue
            
            parts = stem.rsplit('_', 1)
            if len(parts) != 2:
                continue
            
            shoe_id, view = parts
            view = view.lower()
            
            # Check if it's a valid view
            if view in required_views:
                shoes[shoe_id]['views'].add(view)
                shoes[shoe_id]['files'][view] = img_file.name
    
    print(f"   Found {total_files} total image files")
    print(f"   Found {len(shoes)} unique shoes")
    print()
    
    # Analyze completeness
    complete_shoes = []
    incomplete_shoes = []
    
    for shoe_id, data in sorted(shoes.items(), key=lambda x: (0, int(x[0])) if x[0].isdigit() else (1, x[0])):
        views = data['views']
        missing_views = set(required_views) - views
        extra_views = views - set(required_views)
        
        if len(views) == 6 and not missing_views:
            complete_shoes.append({
                'shoe_id': shoe_id,
                'status': 'complete',
                'views': sorted(views),
                'files': data['files']
            })
        else:
            incomplete_shoes.append({
                'shoe_id': shoe_id,
                'status': 'incomplete',
                'found_views': sorted(views),
                'missing_views': sorted(missing_views),
                'extra_views': sorted(extra_views) if extra_views else [],
                'count': len(views),
                'files': data['files']
            })
    
    # Print results
    print("="*70)
    print("RESULTS")
    print("="*70)
    print()
    
    print(f"✅ Complete shoes (6/6 views): {len(complete_shoes)}")
    print(f"⚠️  Incomplete shoes: {len(incomplete_shoes)}")
    print()
    
    if incomplete_shoes:
        print("="*70)
        print("INCOMPLETE SHOES DETAIL")
        print("="*70)
        print()
        
        # Group by number of missing views
        by_missing_count = defaultdict(list)
        for shoe in incomplete_shoes:
            missing_count = len(shoe['missing_views'])
            by_missing_count[missing_count].append(shoe)
        
        for missing_count in sorted(by_missing_count.keys(), reverse=True):
            shoes_list = by_missing_count[missing_count]
            print(f"📊 Missing {missing_count} view(s): {len(shoes_list)} shoes")
            print("-" * 70)
            
            for shoe in shoes_list[:10]:  # Show first 10
                print(f"   Shoe ID: {shoe['shoe_id']}")
                print(f"   Found ({shoe['count']}/6): {', '.join(shoe['found_views'])}")
                if shoe['missing_views']:
                    print(f"   Missing: {', '.join(shoe['missing_views'])}")
                if shoe['extra_views']:
                    print(f"   Extra: {', '.join(shoe['extra_views'])}")
                print()
            
            if len(shoes_list) > 10:
                print(f"   ... and {len(shoes_list) - 10} more shoes with {missing_count} missing view(s)")
                print()
        
        print()
    
    # Summary statistics
    print("="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print()
    
    total_shoes = len(shoes)
    if total_shoes > 0:
        completeness_rate = (len(complete_shoes) / total_shoes) * 100
        print(f"Total shoes found: {total_shoes}")
        print(f"Complete: {len(complete_shoes)} ({completeness_rate:.1f}%)")
        print(f"Incomplete: {len(incomplete_shoes)} ({100-completeness_rate:.1f}%)")
        print()
        
        # View coverage statistics
        view_counts = defaultdict(int)
        for shoe_data in shoes.values():
            for view in shoe_data['views']:
                view_counts[view] += 1
        
        print("View coverage:")
        for view in required_views:
            count = view_counts[view]
            coverage = (count / total_shoes) * 100
            status = "✅" if coverage == 100 else "⚠️ "
            print(f"  {status} {view:8s}: {count:4d}/{total_shoes} ({coverage:5.1f}%)")
        print()
    
    # Export detailed report
    if export_report:
        report = {
            'summary': {
                'total_shoes': len(shoes),
                'complete_shoes': len(complete_shoes),
                'incomplete_shoes': len(incomplete_shoes),
                'completeness_rate': (len(complete_shoes) / len(shoes) * 100) if shoes else 0,
                'required_views': required_views
            },
            'complete': complete_shoes,
            'incomplete': incomplete_shoes,
            'view_statistics': dict(view_counts)
        }
        
        report_file = data_dir / 'shoe_completeness_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Detailed report saved: {report_file}")
        print()
    
    # Action items
    if incomplete_shoes:
        print("="*70)
        print("ACTION ITEMS")
        print("="*70)
        print()
        print("Some shoes are missing views. You have these options:")
        print()
        print("1. Generate missing views:")
        print("   - Re-render from 3D models")
        print("   - Or manually photograph missing angles")
        print()
        print("2. Exclude incomplete shoes from training:")
        print("   - Use only the complete shoes for training")
        print(f"   - This means training with {len(complete_shoes)} shoes")
        print()
        print("3. Train anyway:")
        print("   - Some models can handle missing views")
        print("   - May reduce quality slightly")
        print()
    else:
        print("="*70)
        print("🎉 ALL SHOES COMPLETE!")
        print("="*70)
        print()
        print(f"All {len(complete_shoes)} shoes have complete 6-view image sets.")
        print("You're ready to proceed with training! ✨")
        print()
    
    return {
        'total': len(shoes),
        'complete': complete_shoes,
        'incomplete': incomplete_shoes,
        'view_counts': dict(view_counts)
    }


def list_complete_shoes(data_dir):
    """List only the complete shoe IDs (for filtering in training)"""
    result = verify_shoe_completeness(data_dir, export_report=False)
    
    complete_ids = [shoe['shoe_id'] for shoe in result['complete']]
    
    output_file = Path(data_dir) / 'complete_shoe_ids.txt'
    with open(output_file, 'w') as f:
        for shoe_id in complete_ids:
            f.write(f"{shoe_id}\n")
    
    print(f"📝 Complete shoe IDs saved: {output_file}")
    print(f"   ({len(complete_ids)} shoes)")
    
    return complete_ids


def check_specific_shoe(data_dir, shoe_id):
    """Check a specific shoe's completeness"""
    data_dir = Path(data_dir)
    required_views = ['front', 'back', 'left', 'right', 'top', 'bottom']
    
    print(f"\n🔍 Checking shoe: {shoe_id}")
    print("-" * 70)
    
    found_views = {}
    for view in required_views:
        for ext in ['png', 'jpg', 'jpeg']:
            img_path = data_dir / f'{shoe_id}_{view}.{ext}'
            if img_path.exists():
                found_views[view] = img_path.name
                print(f"   ✅ {view:8s}: {img_path.name}")
                break
        else:
            print(f"   ❌ {view:8s}: NOT FOUND")
    
    missing = set(required_views) - set(found_views.keys())
    
    print()
    if missing:
        print(f"   ⚠️  Missing {len(missing)} view(s): {', '.join(sorted(missing))}")
    else:
        print(f"   ✅ Complete! All 6 views found.")
    print()
    
    return len(found_views) == 6


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Verify shoe image completeness'
    )
    parser.add_argument(
        '--data_dir',
        required=True,
        help='Directory containing shoe images'
    )
    parser.add_argument(
        '--list-complete',
        action='store_true',
        help='Export list of complete shoe IDs to file'
    )
    parser.add_argument(
        '--check-shoe',
        type=str,
        help='Check specific shoe ID'
    )
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip saving JSON report'
    )
    
    args = parser.parse_args()
    
    if args.check_shoe:
        # Check specific shoe
        check_specific_shoe(args.data_dir, args.check_shoe)
    elif args.list_complete:
        # Just list complete shoes
        list_complete_shoes(args.data_dir)
    else:
        # Full verification
        verify_shoe_completeness(args.data_dir, export_report=not args.no_report)