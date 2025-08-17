# Dataset Testing Suite

This suite contains comprehensive testing scripts to validate your CSV dataset files for file path errors and image processing issues.

## 📁 Files Overview

The testing suite includes scripts for four different CSV files:

1. **`test_paired_dataset.py`** - Tests `paired_dataset.csv` (mask + past + present images)
2. **`test_unpaired_pairs.py`** - Tests `unpaired_pairs.csv` (past + present images only)
3. **`test_val_dataset.py`** - Tests `val_dataset.csv` (validation dataset)
4. **`test_combined_dataset.py`** - Tests `combined_dataset.csv` (labeled + unlabeled samples)
5. **`run_all_dataset_tests.py`** - Master script to run all tests

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_dataset_testing.txt
```

### 2. Run All Tests

```bash
python run_all_dataset_tests.py
```

### 3. Run Individual Tests

```bash
# Test paired dataset
python test_paired_dataset.py

# Test unpaired pairs
python test_unpaired_pairs.py

# Test validation dataset
python test_val_dataset.py

# Test combined dataset
python test_combined_dataset.py
```

## 🔍 What Each Test Checks

### File Path Validation
- ✅ File existence
- ✅ File accessibility
- ✅ Valid file paths (not directories)
- ✅ Non-empty paths

### Image Processing Validation
- ✅ Ability to read TIFF files with rasterio
- ✅ No NaN values in image data
- ✅ No Inf values in image data
- ✅ Non-zero data (not all pixels are zero)
- ✅ Valid data range (not all identical values)
- ✅ Proper data format and structure

### Special Handling
- **Mask files**: Treated as single-band TIFF files
- **Image files**: Treated as multi-band TIFF files
- **Combined dataset**: Handles "N/A" mask paths for unlabeled data

## 📊 Test Output

Each test provides detailed output including:

```
================================================================================
TESTING PAIRED DATASET
================================================================================
✓ Successfully loaded CSV with 100 rows
Columns: ['mask_path', 'past_image_path', 'present_image_path']

==================== FILE PATH VALIDATION ====================

Row 1:
  ✓ Mask: File exists
    ✓ Processing: Valid data - Range: [0.00, 1.00]
  ✓ Past Image: File exists
    ✓ Processing: Valid data - Range: [0.00, 255.00]
  ✓ Present Image: File exists
    ✓ Processing: Valid data - Range: [0.00, 255.00]

==================== SUMMARY ====================
Total files checked: 300
Valid files: 300
Invalid files: 0
Success rate: 100.0%

🎉 ALL TESTS PASSED! Dataset is ready for use.
```

## ⚠️ Common Issues and Solutions

### 1. File Not Found Errors
- **Problem**: Files referenced in CSV don't exist
- **Solution**: Check file paths and ensure all files are in the correct locations

### 2. Permission Errors
- **Problem**: Cannot access files due to permissions
- **Solution**: Check file permissions and ensure read access

### 3. Corrupted Image Files
- **Problem**: Images contain NaN/Inf values or are all zero
- **Solution**: Re-download or regenerate corrupted image files

### 4. Memory Issues
- **Problem**: Large images causing memory problems
- **Solution**: The scripts are optimized to handle large TIFF files efficiently

## 📋 CSV File Formats

### paired_dataset.csv
```csv
mask_path,past_image_path,present_image_path
/path/to/mask.tif,/path/to/past.tif,/path/to/present.tif
```

### unpaired_pairs.csv
```csv
past_image_path,present_image_path
/path/to/past.tif,/path/to/present.tif
```

### val_dataset.csv
```csv
mask_path,past_image_path,present_image_path
/path/to/mask.tif,/path/to/past.tif,/path/to/present.tif
```

### combined_dataset.csv
```csv
mask,past_image_path,present_image_path,with_label
/path/to/mask.tif,/path/to/past.tif,/path/to/present.tif,True
N/A,/path/to/past.tif,/path/to/present.tif,False
```

## 🛠️ Customization

### Adding New Tests
1. Create a new test script following the existing pattern
2. Use the helper functions `check_file_exists()` and `check_image_processing()`
3. Add the test to `run_all_dataset_tests.py`

### Modifying Validation Rules
- Edit the `check_image_processing()` function to add custom validation rules
- Modify data range checks, format validation, etc.

### Performance Optimization
- The scripts use rasterio for efficient TIFF reading
- Large files are processed without loading entire images into memory
- Timeout protection prevents hanging on corrupted files

## 📞 Support

If you encounter issues:

1. Check that all dependencies are installed correctly
2. Verify file paths in your CSV files
3. Ensure TIFF files are not corrupted
4. Check Python version compatibility (3.7+ recommended)

## 🔧 Dependencies

- **pandas**: CSV file reading and data manipulation
- **numpy**: Numerical operations and array handling
- **Pillow**: Image processing (backup for rasterio)
- **rasterio**: Efficient TIFF file reading and processing

## 📝 License

This testing suite is provided as-is for dataset validation purposes.
