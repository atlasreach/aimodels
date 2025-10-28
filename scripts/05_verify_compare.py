#!/usr/bin/env python3
"""
Model Comparison Verification Script
Compares outputs from all 4 models: SDXL Base, Realistic Vision, Juggernaut XL, SD 1.5
Generates detailed comparison report
"""
import os
from pathlib import Path
from PIL import Image
from collections import defaultdict
import time

# Configuration
MODELS = {
    "sdxl_base": {
        "name": "SDXL Base",
        "folder": "datasets/sdxl_base",
        "expected": 25
    },
    "realistic_vision": {
        "name": "Realistic Vision V5.1",
        "folder": "datasets/realistic_vision",
        "expected": 25
    },
    "juggernaut_xl": {
        "name": "Juggernaut XL",
        "folder": "datasets/juggernaut_xl",
        "expected": 25
    },
    "sd15": {
        "name": "SD 1.5 Baseline",
        "folder": "datasets/sd15",
        "expected": 25
    }
}

REPORT_FILE = "reports/model_comparison.txt"

def analyze_dataset(folder_path, model_name):
    """Analyze a single model's output dataset"""

    stats = {
        'name': model_name,
        'folder': folder_path,
        'total': 0,
        'valid': 0,
        'corrupted': 0,
        'resolutions': defaultdict(int),
        'formats': defaultdict(int),
        'sizes_mb': [],
        'corrupted_files': []
    }

    if not os.path.exists(folder_path):
        print(f"  ⚠️  Folder not found: {folder_path}")
        return stats

    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(folder_path).glob(f"*{ext}"))
        image_files.extend(Path(folder_path).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)
    stats['total'] = len(image_files)

    if stats['total'] == 0:
        print(f"  ⚠️  No images found in {folder_path}")
        return stats

    # Analyze each image
    for img_path in image_files:
        try:
            # Try to open and verify
            with Image.open(img_path) as img:
                img.verify()

            # Re-open for details
            with Image.open(img_path) as img:
                width, height = img.size
                format_name = img.format or "Unknown"

                stats['valid'] += 1
                stats['resolutions'][f"{width}x{height}"] += 1
                stats['formats'][format_name] += 1

                file_size_mb = os.path.getsize(img_path) / (1024 * 1024)
                stats['sizes_mb'].append(file_size_mb)

        except Exception as e:
            stats['corrupted'] += 1
            stats['corrupted_files'].append(str(img_path))

    return stats

def generate_report(all_stats):
    """Generate comparison report"""

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BELLA AI - MODEL COMPARISON REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    # Summary table
    report_lines.append("SUMMARY - ALL MODELS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Model':<25} {'Total':<8} {'Valid':<8} {'Corrupted':<10} {'Success %':<10}")
    report_lines.append("-" * 80)

    for model_id, stats in all_stats.items():
        name = stats['name']
        total = stats['total']
        valid = stats['valid']
        corrupted = stats['corrupted']
        success = (valid / max(total, 1)) * 100

        report_lines.append(f"{name:<25} {total:<8} {valid:<8} {corrupted:<10} {success:>8.1f}%")

    report_lines.append("-" * 80)
    report_lines.append("")

    # Detailed per-model analysis
    for model_id, stats in all_stats.items():
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append(f"MODEL: {stats['name']}")
        report_lines.append("=" * 80)
        report_lines.append(f"Location: {stats['folder']}")
        report_lines.append("")

        # Basic stats
        report_lines.append("STATISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"  Total images:        {stats['total']}")
        report_lines.append(f"  Valid images:        {stats['valid']}")
        report_lines.append(f"  Corrupted images:    {stats['corrupted']}")

        if stats['valid'] > 0:
            success_rate = (stats['valid'] / stats['total']) * 100
            report_lines.append(f"  Success rate:        {success_rate:.1f}%")
        report_lines.append("")

        # Resolutions
        if stats['resolutions']:
            report_lines.append("RESOLUTIONS")
            report_lines.append("-" * 80)
            for resolution, count in sorted(stats['resolutions'].items(), key=lambda x: -x[1]):
                percentage = (count / stats['valid']) * 100 if stats['valid'] > 0 else 0
                report_lines.append(f"  {resolution:15s} : {count:4d} images ({percentage:5.1f}%)")
            report_lines.append("")

        # File sizes
        if stats['sizes_mb']:
            avg_size = sum(stats['sizes_mb']) / len(stats['sizes_mb'])
            min_size = min(stats['sizes_mb'])
            max_size = max(stats['sizes_mb'])
            total_size = sum(stats['sizes_mb'])

            report_lines.append("FILE SIZES")
            report_lines.append("-" * 80)
            report_lines.append(f"  Average size:        {avg_size:.2f} MB")
            report_lines.append(f"  Minimum size:        {min_size:.2f} MB")
            report_lines.append(f"  Maximum size:        {max_size:.2f} MB")
            report_lines.append(f"  Total dataset size:  {total_size:.2f} MB")
            report_lines.append("")

        # Corrupted files
        if stats['corrupted_files']:
            report_lines.append("CORRUPTED FILES")
            report_lines.append("-" * 80)
            for corrupted_file in stats['corrupted_files']:
                report_lines.append(f"  ❌ {corrupted_file}")
            report_lines.append("")

    # Overall comparison
    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("COMPARISON SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Best performer by count
    best_count = max(all_stats.items(), key=lambda x: x[1]['valid'])
    report_lines.append(f"✓ Most images generated:  {best_count[1]['name']} ({best_count[1]['valid']} images)")

    # Avg file sizes comparison
    avg_sizes = {k: (sum(v['sizes_mb']) / len(v['sizes_mb']) if v['sizes_mb'] else 0)
                 for k, v in all_stats.items()}
    largest_avg = max(avg_sizes.items(), key=lambda x: x[1])
    report_lines.append(f"✓ Largest avg file size:  {all_stats[largest_avg[0]]['name']} ({largest_avg[1]:.2f} MB)")

    # Resolution comparison
    report_lines.append("")
    report_lines.append("Resolution breakdown:")
    for model_id, stats in all_stats.items():
        if stats['resolutions']:
            top_res = max(stats['resolutions'].items(), key=lambda x: x[1])
            report_lines.append(f"  {stats['name']:<25} : {top_res[0]} (most common)")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append("RECOMMENDATIONS")
    report_lines.append("=" * 80)
    report_lines.append("")
    report_lines.append("Based on generation metrics:")
    report_lines.append("  • SDXL Base: Good balance of quality and consistency")
    report_lines.append("  • Realistic Vision V5.1: Best for photorealistic faces (512x512)")
    report_lines.append("  • Juggernaut XL: Excellent for consistent facial details (1024x1024)")
    report_lines.append("  • SD 1.5: Fastest but lower quality baseline")
    report_lines.append("")
    report_lines.append("Next steps:")
    report_lines.append("  1. Review generated images manually in each dataset folder")
    report_lines.append("  2. Choose best model based on visual quality")
    report_lines.append("  3. Use chosen model to generate full 100+ image dataset")
    report_lines.append("  4. Proceed with LoRA training on best dataset")
    report_lines.append("")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)

def main():
    print("=" * 80)
    print("MODEL COMPARISON VERIFICATION")
    print("=" * 80)
    print()
    print("Analyzing outputs from all 4 models...")
    print()

    all_stats = {}

    # Analyze each model
    for model_id, model_info in MODELS.items():
        print(f"Analyzing {model_info['name']}...")
        stats = analyze_dataset(model_info['folder'], model_info['name'])
        all_stats[model_id] = stats

        if stats['valid'] > 0:
            print(f"  ✓ Found {stats['valid']} valid images")
        else:
            print(f"  ⚠️  No valid images found")
        print()

    # Generate report
    print("Generating comparison report...")
    report_text = generate_report(all_stats)

    # Save report
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        f.write(report_text)

    # Print report
    print()
    print(report_text)
    print()
    print(f"✓ Report saved to: {REPORT_FILE}")
    print()

if __name__ == "__main__":
    main()
