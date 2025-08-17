#!/usr/bin/env python3
"""
Replace original CSV files with cleaned versions
This script will replace the original CSV files with their cleaned versions
"""

import os
import shutil
from datetime import datetime

def replace_with_cleaned():
    """Replace original CSV files with their cleaned versions"""
    print("🔄 REPLACING ORIGINAL FILES WITH CLEANED VERSIONS")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define file mappings
    file_mappings = [
        {
            'original': 'dataset_delhi/paired_dataset.csv',
            'cleaned': 'dataset_delhi/paired_dataset_cleaned.csv'
        },
        {
            'original': 'dataset_delhi/unpaired_pairs.csv',
            'cleaned': 'dataset_delhi/unpaired_pairs_cleaned.csv'
        },
        {
            'original': 'dataset_delhi/val_dataset.csv',
            'cleaned': 'dataset_delhi/val_dataset_cleaned.csv'
        },
        {
            'original': 'dataset_delhi/combined_dataset.csv',
            'cleaned': 'dataset_delhi/combined_dataset_cleaned.csv'
        }
    ]
    
    replaced_count = 0
    skipped_count = 0
    
    print(f"\n🔍 Checking cleaned files...")
    
    for mapping in file_mappings:
        original_file = mapping['original']
        cleaned_file = mapping['cleaned']
        
        print(f"\nChecking: {os.path.basename(original_file)}")
        
        # Check if cleaned file exists
        if not os.path.exists(cleaned_file):
            print(f"  ❌ Cleaned file not found: {cleaned_file}")
            print(f"  ⚠️  Please run clean_dataset_csv.py first")
            skipped_count += 1
            continue
        
        # Check if original file exists
        if not os.path.exists(original_file):
            print(f"  ❌ Original file not found: {original_file}")
            skipped_count += 1
            continue
        
        # Get file sizes for comparison
        original_size = os.path.getsize(original_file)
        cleaned_size = os.path.getsize(cleaned_file)
        
        print(f"  Original size: {original_size:,} bytes")
        print(f"  Cleaned size: {cleaned_size:,} bytes")
        
        # Replace the file
        try:
            shutil.copy2(cleaned_file, original_file)
            print(f"  ✅ Successfully replaced with cleaned version")
            replaced_count += 1
        except Exception as e:
            print(f"  ❌ Error replacing file: {e}")
            skipped_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("📊 REPLACEMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Files replaced: {replaced_count}")
    print(f"Files skipped: {skipped_count}")
    print(f"Total processed: {len(file_mappings)}")
    
    if replaced_count > 0:
        print(f"\n🎉 SUCCESS!")
        print(f"  {replaced_count} files have been replaced with cleaned versions")
        print(f"  Your datasets are now ready for use")
    else:
        print(f"\n⚠️  No files were replaced")
        print(f"  Please ensure cleaned files exist by running clean_dataset_csv.py first")

def restore_from_backup(backup_dir):
    """Restore original files from backup"""
    print(f"🔄 RESTORING FROM BACKUP: {backup_dir}")
    print("=" * 80)
    
    if not os.path.exists(backup_dir):
        print(f"❌ Backup directory not found: {backup_dir}")
        return
    
    # Define file mappings
    file_mappings = [
        'paired_dataset.csv',
        'unpaired_pairs.csv',
        'val_dataset.csv',
        'combined_dataset.csv'
    ]
    
    restored_count = 0
    
    for filename in file_mappings:
        backup_file = os.path.join(backup_dir, filename)
        target_file = os.path.join('dataset_delhi', filename)
        
        if os.path.exists(backup_file):
            try:
                shutil.copy2(backup_file, target_file)
                print(f"  ✅ Restored {filename}")
                restored_count += 1
            except Exception as e:
                print(f"  ❌ Error restoring {filename}: {e}")
        else:
            print(f"  ⚠️  Backup file not found: {backup_file}")
    
    print(f"\n📊 RESTORATION SUMMARY:")
    print(f"  Files restored: {restored_count}")
    print(f"  Total files: {len(file_mappings)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--restore":
        if len(sys.argv) > 2:
            backup_dir = sys.argv[2]
            restore_from_backup(backup_dir)
        else:
            print("Usage: python replace_with_cleaned.py --restore <backup_directory>")
            print("Example: python replace_with_cleaned.py --restore backup_20231201_143022")
    else:
        replace_with_cleaned()
