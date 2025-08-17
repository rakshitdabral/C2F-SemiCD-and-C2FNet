#!/usr/bin/env python3
"""
Testing script for unpaired_pairs.csv
Checks for file path errors and image processing issues (NaN/Inf values)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import rasterio
import warnings
warnings.filterwarnings('ignore')

def check_file_exists(file_path):
    """Check if a file exists and is accessible"""
    if pd.isna(file_path) or file_path == '':
        return False, "Empty or NaN path"
    
    if not os.path.exists(file_path):
        return False, "File does not exist"
    
    if not os.path.isfile(file_path):
        return False, "Path is not a file"
    
    return True, "File exists"

def check_image_processing(file_path):
    """Check if image can be loaded and processed without NaN/Inf values"""
    try:
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

def test_unpaired_pairs(csv_path):
    """Main testing function for unpaired_pairs.csv"""
    print("=" * 80)
    print("TESTING UNPAIRED PAIRS DATASET")
    print("=" * 80)
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Successfully loaded CSV with {len(df)} rows")
        print(f"Columns: {list(df.columns)}")
    except Exception as e:
        print(f"✗ Error reading CSV: {e}")
        return
    
    # Initialize error tracking
    path_errors = []
    processing_errors = []
    total_files = 0
    valid_files = 0
    
    print(f"\n{'='*20} FILE PATH VALIDATION {'='*20}")
    
    # Check each row
    for idx, row in df.iterrows():
        total_files += 2  # past_image, present_image
        
        print(f"\nRow {idx + 1}:")
        
        # Check past image path
        past_path = row['past_image_path']
        exists, msg = check_file_exists(past_path)
        if not exists:
            path_errors.append(f"Row {idx + 1} - Past Image: {msg} - {past_path}")
            print(f"  ✗ Past Image: {msg}")
        else:
            print(f"  ✓ Past Image: File exists")
            valid_files += 1
            
            # Check past image processing
            valid, msg = check_image_processing(past_path)
            if not valid:
                processing_errors.append(f"Row {idx + 1} - Past Image: {msg} - {past_path}")
                print(f"    ✗ Processing: {msg}")
            else:
                print(f"    ✓ Processing: {msg}")
        
        # Check present image path
        present_path = row['present_image_path']
        exists, msg = check_file_exists(present_path)
        if not exists:
            path_errors.append(f"Row {idx + 1} - Present Image: {msg} - {present_path}")
            print(f"  ✗ Present Image: {msg}")
        else:
            print(f"  ✓ Present Image: File exists")
            valid_files += 1
            
            # Check present image processing
            valid, msg = check_image_processing(present_path)
            if not valid:
                processing_errors.append(f"Row {idx + 1} - Present Image: {msg} - {present_path}")
                print(f"    ✗ Processing: {msg}")
            else:
                print(f"    ✓ Processing: {msg}")
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    print(f"Total files checked: {total_files}")
    print(f"Valid files: {valid_files}")
    print(f"Invalid files: {total_files - valid_files}")
    print(f"Success rate: {(valid_files/total_files)*100:.1f}%")
    
    if path_errors:
        print(f"\n{'='*20} PATH ERRORS ({len(path_errors)}) {'='*20}")
        for error in path_errors:
            print(f"  {error}")
    
    if processing_errors:
        print(f"\n{'='*20} PROCESSING ERRORS ({len(processing_errors)}) {'='*20}")
        for error in processing_errors:
            print(f"  {error}")
    
    if not path_errors and not processing_errors:
        print(f"\n🎉 ALL TESTS PASSED! Dataset is ready for use.")
    else:
        print(f"\n⚠️  {len(path_errors) + len(processing_errors)} issues found. Please fix before using dataset.")

if __name__ == "__main__":
    csv_path = "dataset_delhi/unpaired_pairs.csv"
    test_unpaired_pairs(csv_path)
