"""
Investigate the duplicate records and count discrepancies across:
1. FingerSure_Dataset/metadata.csv
2. FingerSure_Dataset/metadata_backup_pre_10finger.csv
3. MySQL physical_dataset_collection table
4. Physical disk files in FingerSure_Dataset/clean/ and FingerSure_Dataset/distorted/
"""
import os
import sys
import csv
import json
import hashlib
from collections import defaultdict

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import mysql.connector
from database import get_db_connection

def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

print("="*70)
print("1. DISK FILES INSPECTION")
print("="*70)
clean_dir = os.path.abspath("FingerSure_Dataset/clean")
distorted_dir = os.path.abspath("FingerSure_Dataset/distorted")

clean_files = {f: os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.tif', '.tiff'))}
distorted_files = {f: os.path.join(distorted_dir, f) for f in os.listdir(distorted_dir) if f.lower().endswith(('.png', '.bmp', '.jpg', '.tif', '.tiff'))}

print(f"Clean files on disk: {len(clean_files)}")
print(f"Distorted files on disk: {len(distorted_files)}")
total_disk_files = len(clean_files) + len(distorted_files)
print(f"Total image files on disk: {total_disk_files}")

print("\n" + "="*70)
print("2. METADATA.CSV INSPECTION")
print("="*70)
with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

print(f"Total rows in metadata.csv: {len(csv_rows)}")

# Check duplicate keys in CSV
csv_key_counts = defaultdict(list)
for idx, row in enumerate(csv_rows, start=2): # line number in file
    key = (row.get('participant_id', '').strip().upper(),
           row.get('hand', '').strip().upper(),
           row.get('finger', '').strip().upper(),
           row.get('condition', '').strip())
    csv_key_counts[key].append((idx, row))

print(f"Unique (participant, hand, finger, condition) keys in metadata.csv: {len(csv_key_counts)}")
duplicates_in_csv = {k: v for k, v in csv_key_counts.items() if len(v) > 1}
print(f"Keys with duplicate rows in CSV: {len(duplicates_in_csv)}")

for k, entries in duplicates_in_csv.items():
    print(f"\nDuplicate Key: {k}")
    for line_num, entry in entries:
        print(f"  Line {line_num}: participant_id={entry.get('participant_id')}, hand={entry.get('hand')}, finger={entry.get('finger')}, condition={entry.get('condition')}, filename={entry.get('filename')}, image_path={entry.get('image_path')}, timestamp={entry.get('capture_timestamp')}")

print("\n" + "="*70)
print("3. BACKUP METADATA INSPECTION (PRE-10FINGER)")
print("="*70)
with open("FingerSure_Dataset/metadata_backup_pre_10finger.csv", "r", encoding="utf-8") as f:
    backup_rows = list(csv.DictReader(f))
print(f"Total rows in metadata_backup_pre_10finger.csv: {len(backup_rows)}")

backup_key_counts = defaultdict(list)
for idx, row in enumerate(backup_rows, start=2):
    p_id = (row.get('participant_id') or row.get('participant_name', '')).strip().upper()
    finger_val = row.get('finger', '').strip().upper()
    cond_val = row.get('condition', '').strip()
    backup_key_counts[(p_id, finger_val, cond_val)].append((idx, row))

for k, entries in backup_key_counts.items():
    if len(entries) > 1:
        print(f"Backup duplicate key: {k}")
        for line_num, entry in entries:
            print(f"  Line {line_num}: participant={entry.get('participant_name') or entry.get('participant_id')}, finger={entry.get('finger')}, condition={entry.get('condition')}, filename={entry.get('filename')}, timestamp={entry.get('capture_timestamp')}")

print("\n" + "="*70)
print("4. MYSQL DATABASE INSPECTION")
print("="*70)
conn = get_db_connection()
cursor = conn.cursor(dictionary=True)

cursor.execute("SELECT COUNT(*) AS total FROM physical_dataset_collection")
db_total = cursor.fetchone()['total']
print(f"Total records in MySQL physical_dataset_collection: {db_total}")

cursor.execute("""
    SELECT participant_id, hand, finger, `condition`, filename, image_path, capture_timestamp, COUNT(*) as cnt
    FROM physical_dataset_collection
    GROUP BY participant_id, hand, finger, `condition`
    HAVING cnt > 1
""")
db_duplicates = cursor.fetchall()
print(f"Duplicate (participant, hand, finger, condition) in MySQL: {len(db_duplicates)}")
for dup in db_duplicates:
    print(f"  MySQL duplicate: {dup}")

cursor.execute("SELECT participant_id, hand, finger, `condition`, filename, image_path, capture_timestamp FROM physical_dataset_collection")
db_rows = cursor.fetchall()

print("\n" + "="*70)
print("5. MAPPING CSV ROWS TO DISK FILES")
print("="*70)
csv_missing_on_disk = []
csv_found_on_disk = []
file_seen_in_csv = defaultdict(list)

for idx, row in enumerate(csv_rows, start=2):
    img_path = row.get('image_path', '')
    full_path = os.path.join("FingerSure_Dataset", img_path)
    filename = row.get('filename', '')
    file_seen_in_csv[filename].append((idx, row))
    if os.path.exists(full_path):
        csv_found_on_disk.append((idx, row, full_path))
    else:
        csv_missing_on_disk.append((idx, row, full_path))

print(f"CSV rows whose image_path exists on disk: {len(csv_found_on_disk)}")
print(f"CSV rows whose image_path MISSING on disk: {len(csv_missing_on_disk)}")
if csv_missing_on_disk:
    for idx, row, p in csv_missing_on_disk:
        print(f"  Line {idx}: {row.get('filename')} -> {p} (NOT FOUND)")

print(f"Unique filenames mentioned in CSV: {len(file_seen_in_csv)}")
duplicate_filenames_in_csv = {f: rows for f, rows in file_seen_in_csv.items() if len(rows) > 1}
print(f"Filenames appearing more than once in CSV: {len(duplicate_filenames_in_csv)}")
for fn, rows in duplicate_filenames_in_csv.items():
    print(f"  Filename '{fn}' appears {len(rows)} times:")
    for lnum, r in rows:
        print(f"    Line {lnum}: participant={r.get('participant_id')}, timestamp={r.get('capture_timestamp')}")

print("\n" + "="*70)
print("6. MAPPING DISK FILES TO CSV")
print("="*70)
all_disk_files = {}
all_disk_files.update({f: ("clean", p) for f, p in clean_files.items()})
all_disk_files.update({f: ("distorted", p) for f, p in distorted_files.items()})

disk_files_not_in_csv = []
for fn, (folder, fpath) in all_disk_files.items():
    if fn not in file_seen_in_csv:
        disk_files_not_in_csv.append((fn, folder, fpath))

print(f"Disk files NOT referenced in metadata.csv: {len(disk_files_not_in_csv)}")
for fn, folder, fpath in disk_files_not_in_csv:
    print(f"  Disk file without CSV row: {folder}/{fn}")

print("\n" + "="*70)
print("7. DISCREPANCY RECONCILIATION: 88 vs 86 vs 84")
print("="*70)
print(f"Disk unique image files: {total_disk_files}")
print(f"Unique filenames in CSV: {len(file_seen_in_csv)}")
print(f"Total rows in CSV: {len(csv_rows)}")
print(f"Total rows in MySQL: {db_total}")

cursor.close()
conn.close()
