import os
import sys
import csv
from collections import defaultdict
import cv2

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from database import get_db_connection

# 1. Read metadata.csv
with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

print(f"Total rows in metadata.csv: {len(csv_rows)}")

# 2. Read MySQL
conn = get_db_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM physical_dataset_collection")
db_rows = cursor.fetchall()
cursor.close()
conn.close()

print(f"Total rows in MySQL: {len(db_rows)}")

# 3. Read disk files
clean_dir = os.path.abspath("FingerSure_Dataset/clean")
distorted_dir = os.path.abspath("FingerSure_Dataset/distorted")

clean_disk_files = sorted(os.listdir(clean_dir))
distorted_disk_files = sorted(os.listdir(distorted_dir))

print(f"Clean files on disk: {len(clean_disk_files)}")
print(f"Distorted files on disk: {len(distorted_disk_files)}")

# Check duplicate keys in metadata.csv
# Key definition: (participant_id_normalized, hand, finger, condition)
# Note: how should participant_id normalization be handled? Let's check how participant_id is formatted
participants_raw = set(r['participant_id'] for r in csv_rows)
print("\nUnique raw participant IDs in CSV:")
for p in sorted(participants_raw):
    print(f"  '{p}'")

# Check if there are participant ID case duplicates (e.g. 'BHUSHAN' vs 'Bhushan', 'Bharthi' vs 'BHarthi')
# Let's inspect rows for each participant
p_lower_map = defaultdict(set)
for p in participants_raw:
    p_lower_map[p.lower()].add(p)

case_variants = {k: v for k, v in p_lower_map.items() if len(v) > 1}
if case_variants:
    print("\nCase variations of participant IDs in CSV:")
    for k, v in case_variants.items():
        print(f"  {k}: {v}")

