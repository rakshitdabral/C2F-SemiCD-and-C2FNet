#!/usr/bin/env python3
import csv
from pathlib import Path
import re

def main():
    root = Path(__file__).parent / "dataset_delhi"
    paired_csv = root / "paired_dataset.csv"
    out_csv    = root / "unpaired_pairs.csv"

    # 1) Read mask-paired CSV and collect used IDs per year
    #    We infer each row: mask covers (Y1,Y2) and past_image_path ends in _XX_YY.tif
    used_ids = {("2018","2020"): set(), ("2019","2023"): set()}
    with paired_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            # extract years from paths
            past = Path(row["past_image_path"]).name
            pres = Path(row["present_image_path"]).name

            # extract year and id via regex
            mp = re.match(r".*_(\d{4})-\d{1,2}-\d{1,2}.*_(\d{2}_\d{2})\.tif$", past)
            sp = re.match(r".*_(\d{4})-\d{1,2}-\d{1,2}.*_(\d{2}_\d{2})\.tif$", pres)
            if not (mp and sp):
                continue
            y1, id1 = mp.groups()
            y2, id2 = sp.groups()
            # only if it matches our intended pairs
            key = (y1, y2)
            if key in used_ids and id1 == id2:
                used_ids[key].add(id1)

    # 2) Build lookup of ALL images by (year, id)
    image_pattern = re.compile(r".*_(\d{4})-\d{1,2}-\d{1,2}.*_(\d{2}_\d{2})\.tif$")
    data_dirs = {
        "2018": root / "2018_data",
        "2019": root / "2019_data",
        "2020": root / "2020_data",
        "2023": root / "2023_data",
    }
    lookup = {("2018","2018"):{}, ("2019","2019"):{}, ("2020","2020"):{}, ("2023","2023"):{}}
    for year, d in data_dirs.items():
        if not d.is_dir():
            print(f"⚠️  Missing folder: {d}")
            continue
        for fp in d.glob("*.tif"):
            m = image_pattern.match(fp.name)
            if not m:
                continue
            yy, img_id = m.groups()
            if yy == year:
                lookup[(year, year)][img_id] = fp

    # 3) For each year-pair, find IDs present in both but not in used_ids
    pairs_to_emit = []
    for (y1, y2), used in used_ids.items():
        ids1 = set(lookup[(y1,y1)].keys())
        ids2 = set(lookup[(y2,y2)].keys())
        candidates = (ids1 & ids2) - used
        for img_id in sorted(candidates):
            past_fp = lookup[(y1,y1)][img_id]
            pres_fp = lookup[(y2,y2)][img_id]
            pairs_to_emit.append((str(past_fp), str(pres_fp)))

    # 4) Write out CSV
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["past_image_path","present_image_path"])
        writer.writerows(pairs_to_emit)

    print(f"✔ Wrote {len(pairs_to_emit)} unlabelled pairs to {out_csv}")

if __name__ == "__main__":
    main()
