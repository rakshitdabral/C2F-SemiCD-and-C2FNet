#!/usr/bin/env python3
"""
Dataset CSV Cleaner
Removes rows with file path errors or image processing issues from CSV files
Creates cleaned versions of the original datasets
"""

import os
import pandas as pd
import numpy as np
import rasterio
import warnings
from datetime import datetime
import shutil
warnings.filterwarnings('ignore')

def check_file_exists(file_path):
    """Check if a file exists and is accessible"""
    if pd.isna(file_path) or file_path == '' or file_path == 'N/A':
        return False, "Empty, NaN, or N/A path"
    
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if not os.path.isfile(file_path):
        return False, "Path is not a file"
    
    return True, "File exists"

def check_image_processing(file_path, file_type="image"):
    """Check if image can be loaded and processed without NaN/Inf values"""
    try:
        if file_type == "mask":
            # For mask files, use rasterio for single-band TIFF
            with rasterio.open(file_path) as src:
                data = src.read(1)  # Read first band
        else:
            # For image files, use rasterio for multi-band TIFF
            with rasterio.open(file_path) as src:
                data = src.read()  # Read all bands
        
        # Check for NaN values
        if np.isnan(data).any():
            return False, "Contains NaN values"
        
        # Check for Inf values
        if np.isinf(data).any():
            return False, "Contains Inf values"
        
        # Check for all zero values (might indicate corrupted data)
        if np.all(data == 0):
            return False, "All values are zero (possibly corrupted)"
        
        # Check data range
        data_min, data_max = np.min(data), np.max(data)
        if data_min == data_max:
            return False, f"All values are identical ({data_min})"
        
        return True, f"Valid data - Range: [{data_min:.2f}, {data_max:.2f}]"
        
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

def clean_paired_dataset(csv_path, output_path=None):
    """Clean paired_dataset.csv by removing rows with issues"""
    print(f"🧹 Cleaning paired_dataset.csv...")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return None
    
    # Initialize tracking
    valid_rows = []
    removed_rows = []
    total_files = 0
    valid_files = 0
    
    print(f"🔍 Checking each row...")
    
    # Check each row
    for idx, row in df.iterrows():
        row_has_issues = False
        row_issues = []
        
        # Check mask path
        mask_path = row['mask_path']
        exists, msg = check_file_exists(mask_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Mask: {msg}")
        else:
            valid, msg = check_image_processing(mask_path, "mask")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Mask: {msg}")
        
        # Check past image path
        past_path = row['past_image_path']
        exists, msg = check_file_exists(past_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Past Image: {msg}")
        else:
            valid, msg = check_image_processing(past_path, "image")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Past Image: {msg}")
        
        # Check present image path
        present_path = row['present_image_path']
        exists, msg = check_file_exists(present_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Present Image: {msg}")
        else:
            valid, msg = check_image_processing(present_path, "image")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Present Image: {msg}")
        
        # Add row to appropriate list
        if row_has_issues:
            removed_rows.append({
                'row_index': idx,
                'row_number': idx + 1,
                'issues': row_issues,
                'mask_path': mask_path,
                'past_image_path': past_path,
                'present_image_path': present_path
            })
        else:
            valid_rows.append(row)
            valid_files += 3
    
    # Create cleaned dataframe
    cleaned_df = pd.DataFrame(valid_rows)
    
    # Save cleaned CSV
    if output_path is None:
        output_path = csv_path.replace('.csv', '_cleaned.csv')
    
    cleaned_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"  Original rows: {len(df)}")
    print(f"  Valid rows: {len(valid_rows)}")
    print(f"  Removed rows: {len(removed_rows)}")
    print(f"  Success rate: {(len(valid_rows)/len(df))*100:.1f}%")
    print(f"  Cleaned file saved to: {output_path}")
    
    if removed_rows:
        print(f"\n❌ REMOVED ROWS:")
        for removed in removed_rows[:10]:  # Show first 10
            print(f"  Row {removed['row_number']}: {', '.join(removed['issues'])}")
        if len(removed_rows) > 10:
            print(f"  ... and {len(removed_rows) - 10} more rows")
    
    return cleaned_df, removed_rows

def clean_unpaired_pairs(csv_path, output_path=None):
    """Clean unpaired_pairs.csv by removing rows with issues"""
    print(f"🧹 Cleaning unpaired_pairs.csv...")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return None
    
    # Initialize tracking
    valid_rows = []
    removed_rows = []
    
    print(f"🔍 Checking each row...")
    
    # Check each row
    for idx, row in df.iterrows():
        row_has_issues = False
        row_issues = []
        
        # Check past image path
        past_path = row['past_image_path']
        exists, msg = check_file_exists(past_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Past Image: {msg}")
        else:
            valid, msg = check_image_processing(past_path, "image")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Past Image: {msg}")
        
        # Check present image path
        present_path = row['present_image_path']
        exists, msg = check_file_exists(present_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Present Image: {msg}")
        else:
            valid, msg = check_image_processing(present_path, "image")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Present Image: {msg}")
        
        # Add row to appropriate list
        if row_has_issues:
            removed_rows.append({
                'row_index': idx,
                'row_number': idx + 1,
                'issues': row_issues,
                'past_image_path': past_path,
                'present_image_path': present_path
            })
        else:
            valid_rows.append(row)
    
    # Create cleaned dataframe
    cleaned_df = pd.DataFrame(valid_rows)
    
    # Save cleaned CSV
    if output_path is None:
        output_path = csv_path.replace('.csv', '_cleaned.csv')
    
    cleaned_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"  Original rows: {len(df)}")
    print(f"  Valid rows: {len(valid_rows)}")
    print(f"  Removed rows: {len(removed_rows)}")
    print(f"  Success rate: {(len(valid_rows)/len(df))*100:.1f}%")
    print(f"  Cleaned file saved to: {output_path}")
    
    if removed_rows:
        print(f"\n❌ REMOVED ROWS:")
        for removed in removed_rows[:10]:  # Show first 10
            print(f"  Row {removed['row_number']}: {', '.join(removed['issues'])}")
        if len(removed_rows) > 10:
            print(f"  ... and {len(removed_rows) - 10} more rows")
    
    return cleaned_df, removed_rows

def clean_val_dataset(csv_path, output_path=None):
    """Clean val_dataset.csv by removing rows with issues"""
    print(f"🧹 Cleaning val_dataset.csv...")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return None
    
    # Initialize tracking
    valid_rows = []
    removed_rows = []
    
    print(f"🔍 Checking each row...")
    
    # Check each row
    for idx, row in df.iterrows():
        row_has_issues = False
        row_issues = []
        
        # Check mask path
        mask_path = row['mask_path']
        exists, msg = check_file_exists(mask_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Mask: {msg}")
        else:
            valid, msg = check_image_processing(mask_path, "mask")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Mask: {msg}")
        
        # Check past image path
        past_path = row['past_image_path']
        exists, msg = check_file_exists(past_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Past Image: {msg}")
        else:
            valid, msg = check_image_processing(past_path, "image")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Past Image: {msg}")
        
        # Check present image path
        present_path = row['present_image_path']
        exists, msg = check_file_exists(present_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Present Image: {msg}")
        else:
            valid, msg = check_image_processing(present_path, "image")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Present Image: {msg}")
        
        # Add row to appropriate list
        if row_has_issues:
            removed_rows.append({
                'row_index': idx,
                'row_number': idx + 1,
                'issues': row_issues,
                'mask_path': mask_path,
                'past_image_path': past_path,
                'present_image_path': present_path
            })
        else:
            valid_rows.append(row)
    
    # Create cleaned dataframe
    cleaned_df = pd.DataFrame(valid_rows)
    
    # Save cleaned CSV
    if output_path is None:
        output_path = csv_path.replace('.csv', '_cleaned.csv')
    
    cleaned_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"  Original rows: {len(df)}")
    print(f"  Valid rows: {len(valid_rows)}")
    print(f"  Removed rows: {len(removed_rows)}")
    print(f"  Success rate: {(len(valid_rows)/len(df))*100:.1f}%")
    print(f"  Cleaned file saved to: {output_path}")
    
    if removed_rows:
        print(f"\n❌ REMOVED ROWS:")
        for removed in removed_rows[:10]:  # Show first 10
            print(f"  Row {removed['row_number']}: {', '.join(removed['issues'])}")
        if len(removed_rows) > 10:
            print(f"  ... and {len(removed_rows) - 10} more rows")
    
    return cleaned_df, removed_rows

def clean_combined_dataset(csv_path, output_path=None):
    """Clean combined_dataset.csv by removing rows with issues"""
    print(f"🧹 Cleaning combined_dataset.csv...")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return None
    
    # Initialize tracking
    valid_rows = []
    removed_rows = []
    labeled_count = 0
    unlabeled_count = 0
    
    print(f"🔍 Checking each row...")
    
    # Check each row
    for idx, row in df.iterrows():
        row_has_issues = False
        row_issues = []
        has_label = row['with_label']
        mask_path = row['mask']
        
        # Check mask path (only for labeled data)
        if has_label:
            exists, msg = check_file_exists(mask_path)
            if not exists:
                row_has_issues = True
                row_issues.append(f"Mask: {msg}")
            else:
                valid, msg = check_image_processing(mask_path, "mask")
                if not valid:
                    row_has_issues = True
                    row_issues.append(f"Mask: {msg}")
        
        # Check past image path
        past_path = row['past_image_path']
        exists, msg = check_file_exists(past_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Past Image: {msg}")
        else:
            valid, msg = check_image_processing(past_path, "image")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Past Image: {msg}")
        
        # Check present image path
        present_path = row['present_image_path']
        exists, msg = check_file_exists(present_path)
        if not exists:
            row_has_issues = True
            row_issues.append(f"Present Image: {msg}")
        else:
            valid, msg = check_image_processing(present_path, "image")
            if not valid:
                row_has_issues = True
                row_issues.append(f"Present Image: {msg}")
        
        # Add row to appropriate list
        if row_has_issues:
            removed_rows.append({
                'row_index': idx,
                'row_number': idx + 1,
                'issues': row_issues,
                'has_label': has_label,
                'mask_path': mask_path,
                'past_image_path': past_path,
                'present_image_path': present_path
            })
        else:
            valid_rows.append(row)
            if has_label:
                labeled_count += 1
            else:
                unlabeled_count += 1
    
    # Create cleaned dataframe
    cleaned_df = pd.DataFrame(valid_rows)
    
    # Save cleaned CSV
    if output_path is None:
        output_path = csv_path.replace('.csv', '_cleaned.csv')
    
    cleaned_df.to_csv(output_path, index=False)
    
    # Print summary
    print(f"\n📊 CLEANING SUMMARY:")
    print(f"  Original rows: {len(df)}")
    print(f"  Valid rows: {len(valid_rows)}")
    print(f"  Removed rows: {len(removed_rows)}")
    print(f"  Success rate: {(len(valid_rows)/len(df))*100:.1f}%")
    print(f"  Labeled samples: {labeled_count}")
    print(f"  Unlabeled samples: {unlabeled_count}")
    print(f"  Cleaned file saved to: {output_path}")
    
    if removed_rows:
        print(f"\n❌ REMOVED ROWS:")
        for removed in removed_rows[:10]:  # Show first 10
            label_type = "Labeled" if removed['has_label'] else "Unlabeled"
            print(f"  Row {removed['row_number']} ({label_type}): {', '.join(removed['issues'])}")
        if len(removed_rows) > 10:
            print(f"  ... and {len(removed_rows) - 10} more rows")
    
    return cleaned_df, removed_rows

def backup_original_files():
    """Create backup of original CSV files"""
    backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    csv_files = [
        "dataset_delhi/paired_dataset.csv",
        "dataset_delhi/unpaired_pairs.csv", 
        "dataset_delhi/val_dataset.csv",
        "dataset_delhi/combined_dataset.csv"
    ]
    
    print(f"📦 Creating backup in {backup_dir}/")
    for csv_file in csv_files:
        if os.path.exists(csv_file):
            backup_path = os.path.join(backup_dir, os.path.basename(csv_file))
            shutil.copy2(csv_file, backup_path)
            print(f"  ✓ Backed up {csv_file}")
    
    return backup_dir

def main():
    """Main function to clean all CSV files"""
    print("🧹 DATASET CSV CLEANER")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create backup
    backup_dir = backup_original_files()
    
    # Clean all datasets
    results = {}
    
    print(f"\n{'='*80}")
    print("CLEANING DATASETS")
    print(f"{'='*80}")
    
    # Clean paired dataset
    paired_csv = "dataset_delhi/paired_dataset.csv"
    if os.path.exists(paired_csv):
        cleaned_df, removed_rows = clean_paired_dataset(paired_csv)
        results['paired'] = {'cleaned_df': cleaned_df, 'removed_rows': removed_rows}
    else:
        print(f"⚠️  {paired_csv} not found, skipping...")
    
    print(f"\n{'-'*80}")
    
    # Clean unpaired pairs
    unpaired_csv = "dataset_delhi/unpaired_pairs.csv"
    if os.path.exists(unpaired_csv):
        cleaned_df, removed_rows = clean_unpaired_pairs(unpaired_csv)
        results['unpaired'] = {'cleaned_df': cleaned_df, 'removed_rows': removed_rows}
    else:
        print(f"⚠️  {unpaired_csv} not found, skipping...")
    
    print(f"\n{'-'*80}")
    
    # Clean validation dataset
    val_csv = "dataset_delhi/val_dataset.csv"
    if os.path.exists(val_csv):
        cleaned_df, removed_rows = clean_val_dataset(val_csv)
        results['val'] = {'cleaned_df': cleaned_df, 'removed_rows': removed_rows}
    else:
        print(f"⚠️  {val_csv} not found, skipping...")
    
    print(f"\n{'-'*80}")
    
    # Clean combined dataset
    combined_csv = "dataset_delhi/combined_dataset.csv"
    if os.path.exists(combined_csv):
        cleaned_df, removed_rows = clean_combined_dataset(combined_csv)
        results['combined'] = {'cleaned_df': cleaned_df, 'removed_rows': removed_rows}
    else:
        print(f"⚠️  {combined_csv} not found, skipping...")
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("📊 CLEANING SUMMARY REPORT")
    print(f"{'='*80}")
    
    total_removed = 0
    total_original = 0
    
    for dataset_name, result in results.items():
        if result['cleaned_df'] is not None:
            original_count = len(result['cleaned_df']) + len(result['removed_rows'])
            removed_count = len(result['removed_rows'])
            valid_count = len(result['cleaned_df'])
            
            total_original += original_count
            total_removed += removed_count
            
            print(f"\n{dataset_name.upper()} DATASET:")
            print(f"  Original: {original_count} rows")
            print(f"  Valid: {valid_count} rows")
            print(f"  Removed: {removed_count} rows")
            print(f"  Success rate: {(valid_count/original_count)*100:.1f}%")
    
    print(f"\nOVERALL SUMMARY:")
    print(f"  Total original rows: {total_original}")
    print(f"  Total removed rows: {total_removed}")
    print(f"  Overall success rate: {((total_original-total_removed)/total_original)*100:.1f}%")
    print(f"  Backup location: {backup_dir}/")
    
    print(f"\n🎉 CLEANING COMPLETED!")
    print(f"  Cleaned files have '_cleaned' suffix")
    print(f"  Original files backed up in {backup_dir}/")
    print(f"  You can now use the cleaned datasets for training")

if __name__ == "__main__":
    main()
