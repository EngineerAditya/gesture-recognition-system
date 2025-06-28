# prepare_dataset.py
# Author: Aditya Sinha
# Combines individual gesture CSVs into a single labeled dataset
# Notes: Assumes filenames follow "<gesture_name>_<timestamp>.csv" format

import pandas as pd
import os
import glob
import re

input_folder = "data/processed_csv"
output_folder = "data/combined_csv"
output_file = os.path.join(output_folder, "combined_dataset.csv")

os.makedirs(output_folder, exist_ok=True)

all_files = glob.glob(os.path.join(input_folder, "*.csv"))
combined_data = []

print(f"[INFO] Found {len(all_files)} CSV files to combine.")

for file in all_files:
    filename = os.path.basename(file)
    base_name = os.path.splitext(filename)[0]  # Remove .csv

    # Extract gesture name before timestamp
    match = re.search(r'\d', base_name)
    if not match:
        print(f"[WARNING] Skipping unexpected filename: {filename}")
        continue

    gesture_name_raw = base_name[:match.start()]
    gesture_name = gesture_name_raw.rstrip("_")

    df = pd.read_csv(file)
    df["label"] = gesture_name

    combined_data.append(df)

# Combine all into one DataFrame
final_df = pd.concat(combined_data, ignore_index=True)
final_df.to_csv(output_file, index=False)

print(f"[INFO] Combined dataset saved to: {output_file}")
print(f"[INFO] Total samples: {len(final_df)}")
print(f"[INFO] Columns: {list(final_df.columns)}")
