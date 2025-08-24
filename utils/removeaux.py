import os

def fix_mask_filenames(folder):
    if not os.path.exists(folder):
        print(f"Error: Directory '{folder}' does not exist!")
        return
    
    print(f"Scanning directory: {folder}")
    
    renamed_count = 0
    already_correct_count = 0
    total_files = 0
    
    try:
        for fname in os.listdir(folder):
            if fname.endswith(".tif"):
                total_files += 1
                # Check if filename already starts with "mask_"
                if not fname.startswith("mask_"):
                    new_name = "mask_" + fname
                    old_path = os.path.join(folder, fname)
                    new_path = os.path.join(folder, new_name)
                    
                    # Check if the new filename already exists
                    if os.path.exists(new_path):
                        print(f"Warning: Cannot rename '{fname}' to '{new_name}' - file already exists!")
                        continue
                    
                    try:
                        os.rename(old_path, new_path)
                        print(f"Renamed: {fname} -> {new_name}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"Error renaming {fname}: {e}")
                else:
                    print(f"Already correct: {fname}")
                    already_correct_count += 1
        
        print(f"\nSummary:")
        print(f"Total .tif files found: {total_files}")
        print(f"Files renamed: {renamed_count}")
        print(f"Files already correct: {already_correct_count}")
        
    except Exception as e:
        print(f"Error accessing directory: {e}")

# Usage
mask_folder = r"cartosat_data/mask"
fix_mask_filenames(mask_folder)