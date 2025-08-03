#!/usr/bin/env python3
import os
import re
import csv
from pathlib import Path

def main():
    # Root folder containing your subfolders
    root_dir = Path(__file__).parent / "dataset_delhi"

    # Define subdirectories
    mask_dir = root_dir / "mask"
    data_dirs = {
        "2018": root_dir / "2018_data",
        "2019": root_dir / "2019_data",
        "2020": root_dir / "2020_data",
        "2023": root_dir / "2023_data",
    }

    # Regex patterns to extract years and ID from filenames
    mask_pattern = re.compile(
        r"mask_(\d{4})_\d{1,2}_\d{1,2}_(\d{4})_\d{1,2}_\d{1,2}_[A-Za-z]_(\d{2}_\d{2})\.tif$"
    )
    image_pattern = re.compile(
        r"S2_Mosaic_(\d{4})-\d{1,2}-\d{1,2}_to_(\d{4})-\d{1,2}-\d{1,2}_(\d{2}_\d{2})\.tif$"
    )

    # Build lookup: (year, id) -> filepath
    image_lookup = {}
    for year, data_dir in data_dirs.items():
        if not data_dir.is_dir():
            print(f"Warning: {data_dir} does not exist or is not a directory")
            continue
        for file in data_dir.iterdir():
            if not file.is_file():
                continue
            m = image_pattern.match(file.name)
            if not m:
                continue
            start_year, end_year, img_id = m.groups()
            # Map both start and end year to this file under the same ID
            image_lookup[(start_year, img_id)] = file
            image_lookup[(end_year,   img_id)] = file

    # Prepare CSV rows
    rows = [("mask_path", "past_image_path", "present_image_path")]
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {mask_dir}")

    for mask_file in mask_dir.iterdir():
        if not mask_file.is_file():
            continue
        m = mask_pattern.match(mask_file.name)
        if not m:
            print(f"Skipping non-mask file: {mask_file.name}")
            continue
        start_year, end_year, mask_id = m.groups()
        past_img = image_lookup.get((start_year, mask_id))
        pres_img = image_lookup.get((end_year,   mask_id))

        if past_img and pres_img:
            rows.append((str(mask_file), str(past_img), str(pres_img)))
        else:
            print(
                f"No match for mask {mask_file.name}: "
                f"past_img={'FOUND' if past_img else 'MISSING'}, "
                f"present_img={'FOUND' if pres_img else 'MISSING'}"
            )

    # Write output CSV
    output_csv = root_dir / "paired_dataset.csv"
    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n✔ CSV file written to: {output_csv}")
    print(f"Total valid pairs: {len(rows)-1}")

if __name__ == "__main__":
    main()
