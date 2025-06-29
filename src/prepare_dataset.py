# I have used ChatGPT to structure and make this code readable 😄
# Purpose: Combine all gesture CSV files into one labeled dataset for ML
# Notes: Assumes filenames follow "<gesture_name>_<timestamp>.csv" format

import pandas as pd
import os
import glob
import re

# ----------------- Path Setup -----------------
input_folder = "data/processed_csv"
output_folder = "data/combined_csv"
output_file = os.path.join(output_folder, "combined_dataset.csv")

os.makedirs(output_folder, exist_ok=True)

# ----------------- Find All CSV Files -----------------
all_files = glob.glob(os.path.join(input_folder, "*.csv"))
combined_data = []

print(f"[INFO] Found {len(all_files)} CSV files to combine.")

# ----------------- Process Each File -----------------
for file in all_files:
    filename = os.path.basename(file)
    base_name = os.path.splitext(filename)[0]  # Remove .csv extension

    # Extract gesture name before the first number (timestamp)
    match = re.search(r'\d', base_name)
    if not match:
        print(f"[WARNING] Skipping unexpected filename: {filename}")
        continue

    gesture_name_raw = base_name[:match.start()]
    gesture_name = gesture_name_raw.rstrip("_")

    try:
        df = pd.read_csv(file)
        df["label"] = gesture_name
        combined_data.append(df)
    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")

# ----------------- Combine and Save -----------------
if combined_data:
    final_df = pd.concat(combined_data, ignore_index=True)
    final_df.to_csv(output_file, index=False)

    print(f"[INFO] Combined dataset saved to: {output_file}")
    print(f"[INFO] Total samples: {len(final_df)}")
    print(f"[INFO] Columns: {list(final_df.columns)}")
else:
    print("[WARNING] No valid data found to combine.")
