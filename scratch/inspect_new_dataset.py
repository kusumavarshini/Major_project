import os
import sys
import csv
import json
from collections import defaultdict
import cv2

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from database import get_db_connection

clean_dir = os.path.abspath("FingerSure_Dataset/clean")
distorted_dir = os.path.abspath("FingerSure_Dataset/distorted")

clean_files = [f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.tif', '.tiff'))]
distorted_files = [f for f in os.listdir(distorted_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.tif', '.tiff'))]

print(f"Clean files on disk: {len(clean_files)}")
print(f"Distorted files on disk: {len(distorted_files)}")
print(f"Total files on disk: {len(clean_files) + len(distorted_files)}")

# Test image decoding
corrupted_files = []
for folder, flist in [("clean", clean_files), ("distorted", distorted_files)]:
    for fn in flist:
        p = os.path.join("FingerSure_Dataset", folder, fn)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            corrupted_files.append((folder, fn, "imread_failed"))

print(f"Corrupted image files on disk: {len(corrupted_files)}")

# Inspect metadata.csv
with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

print(f"Total rows in metadata.csv: {len(csv_rows)}")

# Inspect MySQL
conn = get_db_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT COUNT(*) AS total FROM physical_dataset_collection")
db_total = cursor.fetchone()['total']
cursor.execute("SELECT participant_id, hand, finger, `condition`, filename, image_path, capture_timestamp FROM physical_dataset_collection")
db_rows = cursor.fetchall()
cursor.close()
conn.close()

print(f"Total rows in MySQL physical_dataset_collection: {db_total}")

# Look at unique participants across CSV, MySQL, and Disk
participants_csv = set(r['participant_id'].strip() for r in csv_rows)
participants_db = set(r['participant_id'].strip() for r in db_rows)
print(f"Unique participants in CSV: {len(participants_csv)}")
print(f"Unique participants in MySQL: {len(participants_db)}")

# Let's inspect the files in clean and distorted to understand their naming patterns
print("\nSample clean files (first 10):", clean_files[:10])
print("Sample distorted files (first 10):", distorted_files[:10])

