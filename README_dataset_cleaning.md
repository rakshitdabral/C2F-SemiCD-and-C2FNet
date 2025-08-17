# Dataset CSV Cleaning Suite

This suite provides tools to automatically clean your CSV dataset files by removing rows with file path errors or image processing issues.

## 📁 Files Overview

1. **`clean_dataset_csv.py`** - Main cleaning script that removes problematic rows
2. **`replace_with_cleaned.py`** - Script to replace original files with cleaned versions
3. **`README_dataset_cleaning.md`** - This documentation file

## 🚀 Quick Start

### Step 1: Clean Your Datasets

```bash
# Run the cleaning script
python clean_dataset_csv.py
```

This will:
- ✅ Create a backup of your original CSV files
- ✅ Check all file paths and image processing
- ✅ Remove rows with issues
- ✅ Create cleaned versions with `_cleaned` suffix
- ✅ Provide detailed reports

### Step 2: Replace Original Files (Optional)

```bash
# Replace original files with cleaned versions
python replace_with_cleaned.py
```

## 📊 What Gets Cleaned

The cleaning process removes rows that have any of these issues:

### File Path Issues
- ❌ Files that don't exist
- ❌ Empty or invalid paths
- ❌ Paths pointing to directories instead of files
- ❌ Permission errors

### Image Processing Issues
- ❌ Files that can't be read with rasterio
- ❌ Images containing NaN values
- ❌ Images containing Inf values
- ❌ Images with all zero values (corrupted)
- ❌ Images with identical values (no variation)

## 🔄 Workflow Options

### Option 1: Keep Both Original and Cleaned Files
```bash
# Just clean, keep originals
python clean_dataset_csv.py
```
- Original files remain unchanged
- Cleaned files saved as `*_cleaned.csv`
- You can manually review and decide what to use

### Option 2: Replace Originals with Cleaned Versions
```bash
# Clean and replace originals
python clean_dataset_csv.py
python replace_with_cleaned.py
```
- Original files are replaced with cleaned versions
- Backup is automatically created
- Ready for immediate use

### Option 3: Restore from Backup (if needed)
```bash
# Restore original files from backup
python replace_with_cleaned.py --restore backup_20231201_143022
```

## 📋 Sample Output

```
🧹 DATASET CSV CLEANER
================================================================================
Started at: 2023-12-01 14:30:22

📦 Creating backup in backup_20231201_143022/
  ✓ Backed up dataset_delhi/paired_dataset.csv
  ✓ Backed up dataset_delhi/unpaired_pairs.csv
  ✓ Backed up dataset_delhi/val_dataset.csv
  ✓ Backed up dataset_delhi/combined_dataset.csv

================================================================================
CLEANING DATASETS
================================================================================

🧹 Cleaning paired_dataset.csv...
✓ Loaded CSV with 100 rows
🔍 Checking each row...

📊 CLEANING SUMMARY:
  Original rows: 100
  Valid rows: 95
  Removed rows: 5
  Success rate: 95.0%
  Cleaned file saved to: dataset_delhi/paired_dataset_cleaned.csv

❌ REMOVED ROWS:
  Row 15: Mask: File does not exist
  Row 23: Past Image: Contains NaN values
  Row 45: Present Image: Error reading file: Permission denied
  Row 67: Mask: All values are zero (possibly corrupted)
  Row 89: Past Image: All values are identical (0.00)

--------------------------------------------------------------------------------

🧹 Cleaning unpaired_pairs.csv...
✓ Loaded CSV with 432 rows
🔍 Checking each row...

📊 CLEANING SUMMARY:
  Original rows: 432
  Valid rows: 428
  Removed rows: 4
  Success rate: 99.1%
  Cleaned file saved to: dataset_delhi/unpaired_pairs_cleaned.csv

--------------------------------------------------------------------------------

🧹 Cleaning val_dataset.csv...
✓ Loaded CSV with 20 rows
🔍 Checking each row...

📊 CLEANING SUMMARY:
  Original rows: 20
  Valid rows: 20
  Removed rows: 0
  Success rate: 100.0%
  Cleaned file saved to: dataset_delhi/val_dataset_cleaned.csv

--------------------------------------------------------------------------------

🧹 Cleaning combined_dataset.csv...
✓ Loaded CSV with 530 rows
🔍 Checking each row...

📊 CLEANING SUMMARY:
  Original rows: 530
  Valid rows: 525
  Removed rows: 5
  Success rate: 99.1%
  Labeled samples: 300
  Unlabeled samples: 225
  Cleaned file saved to: dataset_delhi/combined_dataset_cleaned.csv

================================================================================
📊 CLEANING SUMMARY REPORT
================================================================================

PAIRED DATASET:
  Original: 100 rows
  Valid: 95 rows
  Removed: 5 rows
  Success rate: 95.0%

UNPAIRED DATASET:
  Original: 432 rows
  Valid: 428 rows
  Removed: 4 rows
  Success rate: 99.1%

VAL DATASET:
  Original: 20 rows
  Valid: 20 rows
  Removed: 0 rows
  Success rate: 100.0%

COMBINED DATASET:
  Original: 530 rows
  Valid: 525 rows
  Removed: 5 rows
  Success rate: 99.1%

OVERALL SUMMARY:
  Total original rows: 1082
  Total removed rows: 14
  Overall success rate: 98.7%
  Backup location: backup_20231201_143022/

🎉 CLEANING COMPLETED!
  Cleaned files have '_cleaned' suffix
  Original files backed up in backup_20231201_143022/
  You can now use the cleaned datasets for training
```

## 🛡️ Safety Features

### Automatic Backup
- Original files are automatically backed up before any changes
- Backup directory includes timestamp: `backup_YYYYMMDD_HHMMSS/`
- You can always restore from backup if needed

### Detailed Reporting
- Shows exactly which rows were removed and why
- Provides success rates for each dataset
- Lists specific issues found

### Non-Destructive by Default
- Original files are not modified unless you explicitly run the replacement script
- Cleaned files are saved with `_cleaned` suffix
- You can review results before making changes

## 📁 File Structure After Cleaning

```
dataset_delhi/
├── paired_dataset.csv              # Original (or cleaned if replaced)
├── paired_dataset_cleaned.csv      # Cleaned version
├── unpaired_pairs.csv              # Original (or cleaned if replaced)
├── unpaired_pairs_cleaned.csv      # Cleaned version
├── val_dataset.csv                 # Original (or cleaned if replaced)
├── val_dataset_cleaned.csv         # Cleaned version
├── combined_dataset.csv            # Original (or cleaned if replaced)
└── combined_dataset_cleaned.csv    # Cleaned version

backup_20231201_143022/             # Backup directory
├── paired_dataset.csv              # Original backup
├── unpaired_pairs.csv              # Original backup
├── val_dataset.csv                 # Original backup
└── combined_dataset.csv            # Original backup
```

## 🔧 Advanced Usage

### Clean Individual Datasets
You can modify the `clean_dataset_csv.py` script to clean only specific datasets by commenting out the unwanted ones.

### Custom Output Paths
You can specify custom output paths:
```python
# In clean_dataset_csv.py, modify the function calls:
clean_paired_dataset("dataset_delhi/paired_dataset.csv", "custom_path/cleaned.csv")
```

### Batch Processing
For multiple dataset directories:
```bash
# Create a simple script to process multiple directories
for dir in dataset_*; do
    echo "Processing $dir"
    python clean_dataset_csv.py --input-dir $dir
done
```

## ⚠️ Important Notes

1. **Backup First**: Always ensure you have backups before running cleaning scripts
2. **Review Results**: Check the cleaning reports to understand what was removed
3. **Test After Cleaning**: Run your training scripts to ensure everything works
4. **Large Datasets**: The cleaning process can take time for large datasets
5. **Memory Usage**: Large TIFF files may require sufficient RAM

## 🆘 Troubleshooting

### Common Issues

**"No cleaned files found"**
- Run `clean_dataset_csv.py` first
- Check that the script completed successfully

**"Permission denied"**
- Ensure you have read/write permissions for the dataset directory
- Check file permissions on individual CSV files

**"Backup directory not found"**
- The backup directory name includes timestamp
- Look for directories starting with `backup_`

**"Memory error"**
- Large datasets may require more RAM
- Consider processing datasets in smaller batches

### Getting Help

1. Check the detailed error messages in the output
2. Review the backup directory for original files
3. Run the testing scripts first to identify issues
4. Ensure all dependencies are installed correctly

## 📞 Support

If you encounter issues:
1. Check the error messages carefully
2. Verify file paths and permissions
3. Ensure all dependencies are installed
4. Review the backup files if needed

The cleaning suite is designed to be safe and informative, providing detailed feedback about what was cleaned and why.
