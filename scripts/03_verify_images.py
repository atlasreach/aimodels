#!/usr/bin/env python3
"""
Verify dataset images for quality and integrity
"""
import os
from pathlib import Path
from PIL import Image
from collections import defaultdict

# Configuration
DATASET_DIR = "datasets/bella_face_aug"
REPORT_FILE = "generated/verification_report.txt"

def verify_images():
    print("=" * 70)
    print("BELLA FACE DATASET VERIFICATION")
    print("=" * 70)
    print(f"Scanning directory: {DATASET_DIR}")
    print()

    # Check if dataset directory exists
    if not os.path.exists(DATASET_DIR):
        print(f"❌ ERROR: Dataset directory not found: {DATASET_DIR}")
        return

    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(DATASET_DIR).glob(f"*{ext}"))
        image_files.extend(Path(DATASET_DIR).glob(f"*{ext.upper()}"))

    image_files = sorted(image_files)
    total_count = len(image_files)

    print(f"✓ Found {total_count} image files")
    print()

    # Statistics
    stats = {
        'total': total_count,
        'valid': 0,
        'corrupted': 0,
        'resolutions': defaultdict(int),
        'formats': defaultdict(int),
        'sizes_mb': [],
        'corrupted_files': []
    }

    # Verify each image
    print("Verifying images...")
    print("-" * 70)

    for idx, img_path in enumerate(image_files, 1):
        try:
            # Try to open and verify image
            with Image.open(img_path) as img:
                # Verify image can be loaded
                img.verify()

            # Re-open to get details (verify() closes the file)
            with Image.open(img_path) as img:
                width, height = img.size
                format_name = img.format
                mode = img.mode

                # Collect statistics
                stats['valid'] += 1
                stats['resolutions'][f"{width}x{height}"] += 1
                stats['formats'][format_name] += 1

                # File size in MB
                file_size_mb = os.path.getsize(img_path) / (1024 * 1024)
                stats['sizes_mb'].append(file_size_mb)

                # Progress indicator
                if idx % 10 == 0 or idx == total_count:
                    progress = idx / total_count * 100
                    print(f"  [{idx:3d}/{total_count}] ({progress:5.1f}%) - {img_path.name}: {width}x{height} {format_name}")

        except Exception as e:
            stats['corrupted'] += 1
            stats['corrupted_files'].append(str(img_path))
            print(f"  ❌ [{idx:3d}/{total_count}] CORRUPTED: {img_path.name}")
            print(f"     Error: {str(e)}")

    print("-" * 70)
    print()

    # Generate report
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("BELLA FACE DATASET VERIFICATION REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")

    # Summary
    report_lines.append("SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"Total images found:      {stats['total']}")
    report_lines.append(f"Valid images:            {stats['valid']}")
    report_lines.append(f"Corrupted images:        {stats['corrupted']}")
    report_lines.append(f"Success rate:            {stats['valid']/max(stats['total'], 1)*100:.1f}%")
    report_lines.append("")

    # Resolution breakdown
    report_lines.append("RESOLUTIONS")
    report_lines.append("-" * 70)
    for resolution, count in sorted(stats['resolutions'].items(), key=lambda x: -x[1]):
        percentage = count / stats['valid'] * 100 if stats['valid'] > 0 else 0
        report_lines.append(f"  {resolution:15s} : {count:4d} images ({percentage:5.1f}%)")
    report_lines.append("")

    # Format breakdown
    report_lines.append("FORMATS")
    report_lines.append("-" * 70)
    for format_name, count in sorted(stats['formats'].items(), key=lambda x: -x[1]):
        percentage = count / stats['valid'] * 100 if stats['valid'] > 0 else 0
        report_lines.append(f"  {format_name:10s} : {count:4d} images ({percentage:5.1f}%)")
    report_lines.append("")

    # File size statistics
    if stats['sizes_mb']:
        avg_size = sum(stats['sizes_mb']) / len(stats['sizes_mb'])
        min_size = min(stats['sizes_mb'])
        max_size = max(stats['sizes_mb'])
        total_size = sum(stats['sizes_mb'])

        report_lines.append("FILE SIZES")
        report_lines.append("-" * 70)
        report_lines.append(f"  Average size:        {avg_size:.2f} MB")
        report_lines.append(f"  Minimum size:        {min_size:.2f} MB")
        report_lines.append(f"  Maximum size:        {max_size:.2f} MB")
        report_lines.append(f"  Total dataset size:  {total_size:.2f} MB")
        report_lines.append("")

    # Corrupted files
    if stats['corrupted_files']:
        report_lines.append("CORRUPTED FILES")
        report_lines.append("-" * 70)
        for corrupted_file in stats['corrupted_files']:
            report_lines.append(f"  ❌ {corrupted_file}")
        report_lines.append("")

    # Status
    report_lines.append("STATUS")
    report_lines.append("-" * 70)
    if stats['corrupted'] == 0:
        report_lines.append("✓ ALL IMAGES VERIFIED SUCCESSFULLY")
        report_lines.append("✓ Dataset is ready for LoRA training")
    else:
        report_lines.append(f"⚠️  WARNING: {stats['corrupted']} corrupted image(s) found")
        report_lines.append("   Please review and remove corrupted files before training")

    report_lines.append("=" * 70)

    # Print report to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save report to file
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        f.write(report_text)

    print()
    print(f"✓ Report saved to: {REPORT_FILE}")
    print()

    return stats

if __name__ == "__main__":
    verify_images()
