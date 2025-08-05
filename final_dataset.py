#!/usr/bin/env python3
import os
import pandas as pd

def make_dataset():
    # ─── Your own file locations ───────────────────────────────────
    labelled_csv = r"dataset_delhi\paired_dataset.csv"
    unpaired_csv = r"dataset_delhi\unpaired_pairs.csv"
    output_csv   = r"dataset_delhi\combined_dataset.csv"
    # ────────────────────────────────────────────────────────────────

    # Ensure output directory exists
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Load labelled data
    df_lab = pd.read_csv(labelled_csv)
    df_lab = df_lab.rename(columns={
        "past_image_path": "past_image_path",
        "present_image_path": "present_image_path",
        "mask_path": "mask"
    })
    df_lab["with_label"] = True

    # Load unpaired data
    df_unp = pd.read_csv(unpaired_csv)
    df_unp = df_unp.rename(columns={
        "past_image_path": "past_image_path",
        "present_image_path": "present_image_path"
    })
    df_unp["mask"] = "N/A"           # <— use "0" here if you prefer zero
    df_unp["with_label"] = False

    # Combine & sort by basename of A
    df_all = pd.concat([df_lab, df_unp], ignore_index=True)
    df_all["__base"] = df_all["past_image_path"].str.replace(r".*[\\/]", "", regex=True)
    df_all = df_all.sort_values("__base").drop(columns="__base")

    # Write out
    df_all.to_csv(output_csv, index=False)
    print(f"✅ Combined dataset written to:\n   {output_csv}")

if __name__ == "__main__":
    make_dataset()
