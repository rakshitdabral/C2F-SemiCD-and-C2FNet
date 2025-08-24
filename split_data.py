import os
import shutil
import random
from pathlib import Path

def split_data(source_dir, train_size=50, val_size=12, random_seed=42):
    """
    Split the labelled data into train and validation sets
    
    Args:
        source_dir: Path to the labelled directory
        train_size: Number of samples for training
        val_size: Number of samples for validation
        random_seed: Random seed for reproducibility
    """
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all file names from folder A (they should match across all folders)
    folder_a = os.path.join(source_dir, 'A')
    all_files = [f for f in os.listdir(folder_a) if f.endswith('.tif')]
    all_files = sorted(all_files)  # Sort for consistent ordering
    
    print(f"Total files found: {len(all_files)}")
    print(f"Requested train size: {train_size}")
    print(f"Requested validation size: {val_size}")
    
    if len(all_files) < train_size + val_size:
        print(f"Warning: Not enough files. Available: {len(all_files)}, Requested: {train_size + val_size}")
        return
    
    # Randomly select files for train and validation
    selected_files = random.sample(all_files, train_size + val_size)
    train_files = selected_files[:train_size]
    val_files = selected_files[train_size:]
    
    print(f"Selected {len(train_files)} files for training")
    print(f"Selected {len(val_files)} files for validation")
    
    # Create train and validation directories
    train_dir = os.path.join(source_dir, 'train')
    val_dir = os.path.join(source_dir, 'val')
    
    # Create directory structure
    for split_dir in [train_dir, val_dir]:
        for subdir in ['A', 'B', 'mask']:
            os.makedirs(os.path.join(split_dir, subdir), exist_ok=True)
    
    # Copy files to train directory
    print("\nCopying files to train directory...")
    for filename in train_files:
        # Extract base name (remove _T1, _T2, etc.)
        base_name = filename.replace('_T1.tif', '').replace('_T2.tif', '')
        
        # Extract the patch number from the base name
        # Convert patch_01111 to 01111 for mask files
        patch_number = base_name.replace('patch_', '')
        
        # Copy A file
        src_a = os.path.join(source_dir, 'A', f"{base_name}_T1.tif")
        dst_a = os.path.join(train_dir, 'A', f"{base_name}_T1.tif")
        shutil.copy2(src_a, dst_a)
        
        # Copy B file
        src_b = os.path.join(source_dir, 'B', f"{base_name}_T2.tif")
        dst_b = os.path.join(train_dir, 'B', f"{base_name}_T2.tif")
        shutil.copy2(src_b, dst_b)
        
        # Copy mask file - use patch number instead of full base name
        src_mask = os.path.join(source_dir, 'mask', f"mask_{patch_number}.tif")
        dst_mask = os.path.join(train_dir, 'mask', f"mask_{patch_number}.tif")
        shutil.copy2(src_mask, dst_mask)
    
    # Copy files to validation directory
    print("Copying files to validation directory...")
    for filename in val_files:
        # Extract base name (remove _T1, _T2, etc.)
        base_name = filename.replace('_T1.tif', '').replace('_T2.tif', '')
        
        # Extract the patch number from the base name
        # Convert patch_01111 to 01111 for mask files
        patch_number = base_name.replace('patch_', '')
        
        # Copy A file
        src_a = os.path.join(source_dir, 'A', f"{base_name}_T1.tif")
        dst_a = os.path.join(val_dir, 'A', f"{base_name}_T1.tif")
        shutil.copy2(src_a, dst_a)
        
        # Copy B file
        src_b = os.path.join(source_dir, 'B', f"{base_name}_T2.tif")
        dst_b = os.path.join(val_dir, 'B', f"{base_name}_T2.tif")
        shutil.copy2(src_b, dst_b)
        
        # Copy mask file - use patch number instead of full base name
        src_mask = os.path.join(source_dir, 'mask', f"mask_{patch_number}.tif")
        dst_mask = os.path.join(val_dir, 'mask', f"mask_{patch_number}.tif")
        shutil.copy2(src_mask, dst_mask)
    
    print("\nData splitting completed!")
    print(f"Train data: {train_dir}")
    print(f"Validation data: {val_dir}")
    
    # Print some statistics
    print(f"\nTrain files: {len(os.listdir(os.path.join(train_dir, 'A')))}")
    print(f"Validation files: {len(os.listdir(os.path.join(val_dir, 'A')))}")
    
    return train_dir, val_dir

if __name__ == "__main__":
    # Split the data
    source_directory = "labelled"
    train_dir, val_dir = split_data(source_directory, train_size=50, val_size=12)
