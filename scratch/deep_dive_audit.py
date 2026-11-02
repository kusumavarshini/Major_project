import os
import sys
import csv

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Check Ashitta files on disk
clean_dir = "FingerSure_Dataset/clean"
distorted_dir = "FingerSure_Dataset/distorted"

print("--- Clean files containing 'ashitta' or 'Ashitta' ---")
for f in os.listdir(clean_dir):
    if "ashitta" in f.lower():
        print(f"Clean: {f}")

print("\n--- Distorted files containing 'ashitta' or 'Ashitta' ---")
for f in os.listdir(distorted_dir):
    if "ashitta" in f.lower():
        print(f"Distorted: {f}")

print("\n--- Rows in metadata.csv for Ashitta ---")
with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    for idx, row in enumerate(csv.DictReader(f), start=2):
        if "ashitta" in row['participant_id'].lower() or "ashitta" in row['filename'].lower():
            print(f"Line {idx}: {row}")

print("\n--- Checking Bharthi files on disk and CSV ---")
for f in os.listdir(distorted_dir):
    if "bharthi" in f.lower():
        print(f"Distorted on disk: {f}")

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    for idx, row in enumerate(csv.DictReader(f), start=2):
        if "bharthi" in row['participant_id'].lower() or "bharthi" in row['filename'].lower():
            print(f"Line {idx}: {row}")

