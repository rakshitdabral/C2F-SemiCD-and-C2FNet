#!/usr/bin/env python3
import os
import re
import csv
from pathlib import Path
import random

def main():
    # Root folder containing your subfolders
    root_dir = Path(__file__).parent / "dataset_delhi"
    
    # Input paired dataset CSV
    paired_csv = root_dir / "paired_dataset.csv"
    
    # Output validation dataset CSV
    val_csv = root_dir / "val_dataset.csv"
    
    # Validation split ratio (e.g., 20% of paired data for validation)
    val_ratio = 0.2
    
    # Read all paired data
    all_pairs = []
    with paired_csv.open() as f:
        reader = csv.reader(f)
        header = next(reader)  # Get header row
        for row in reader:
            if len(row) == 3:  # Ensure row has mask, past, present paths
                all_pairs.append(row)
    
    # Randomly select validation samples
    random.seed(42)  # For reproducibility
    num_val_samples = int(len(all_pairs) * val_ratio)
    val_indices = random.sample(range(len(all_pairs)), num_val_samples)
    
    # Split into train and validation sets
    val_pairs = [all_pairs[i] for i in val_indices]
    
    # Write validation CSV
    with val_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mask_path", "past_image_path", "present_image_path"])  # Header
        writer.writerows(val_pairs)
    
    print(f"✔ Created validation dataset with {len(val_pairs)} samples at {val_csv}")
    
    # Update paired dataset to remove validation samples (optional)
    # Uncomment if you want to remove validation samples from the original paired dataset
    """
    train_pairs = [pair for i, pair in enumerate(all_pairs) if i not in val_indices]
    with paired_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mask", "past", "present"])  # Header
        writer.writerows(train_pairs)
    print(f"✔ Updated paired dataset with {len(train_pairs)} samples (removed validation samples)")
    """

if __name__ == "__main__":
    main()