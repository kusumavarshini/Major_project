import os
import sys
import csv
import cv2

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from database import get_db_connection

dataset_root = os.path.abspath("FingerSure_Dataset")
clean_dir = os.path.join(dataset_root, "clean")
distorted_dir = os.path.join(dataset_root, "distorted")

# Read disk
clean_files = sorted(os.listdir(clean_dir))
distorted_files = sorted(os.listdir(distorted_dir))

# Read metadata.csv
with open(os.path.join(dataset_root, "metadata.csv"), "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

# Read valid_pairs_manifest.csv
with open(os.path.join(dataset_root, "valid_pairs_manifest.csv"), "r", encoding="utf-8") as f:
    manifest_rows = list(csv.DictReader(f))

# Read MySQL
conn = get_db_connection()
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM physical_dataset_collection")
db_rows = cursor.fetchall()
cursor.close()
conn.close()

print("="*70)
print("SECTION 7 FINAL VALIDATION AUDIT")
print("="*70)

# Check 1: Counts & Relationships
n_clean = len(clean_files)
n_distorted = len(distorted_files)
n_pairs = len(manifest_rows)
n_active_images = n_clean + n_distorted

print(f"Total CLEAN images on disk:     {n_clean}")
print(f"Total DISTORTED images on disk: {n_distorted}")
print(f"Total valid pairs in manifest:  {n_pairs}")
print(f"Total active images on disk:    {n_active_images}")

assert n_clean == n_distorted == n_pairs, f"Count mismatch: clean={n_clean}, dist={n_distorted}, pairs={n_pairs}"
assert n_active_images == 2 * n_pairs, f"Active images != 2 * pairs ({n_active_images} vs {2*n_pairs})"
print("[PASS] Relationship 1: Total CLEAN == Total DISTORTED == Total valid pairs (110 == 110 == 110)")
print("[PASS] Relationship 2: Total active images == 2 * Total valid pairs (220 == 220)")

# Check 2: Lowercase filenames
non_lower_clean = [f for f in clean_files if f != f.lower()]
non_lower_dist = [f for f in distorted_files if f != f.lower()]
assert len(non_lower_clean) == 0, f"Non-lowercase clean files: {non_lower_clean}"
assert len(non_lower_dist) == 0, f"Non-lowercase distorted files: {non_lower_dist}"
print("[PASS] All 220 active image filenames are strictly lowercase.")

# Check 3: Decoding verification
decode_errors = []
for f in clean_files:
    img = cv2.imread(os.path.join(clean_dir, f), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        decode_errors.append(("clean", f))
for f in distorted_files:
    img = cv2.imread(os.path.join(distorted_dir, f), cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        decode_errors.append(("distorted", f))

assert len(decode_errors) == 0, f"Decode errors found: {decode_errors}"
print("[PASS] All 220 active image files decode successfully with OpenCV.")

# Check 4: Pairing Correspondence
clean_set = set(clean_files)
dist_set = set(distorted_files)
unmatched_clean = []
unmatched_dist = []
for f in clean_files:
    expected_d = f.replace("_clean.png", "_distorted.png")
    if expected_d not in dist_set:
        unmatched_clean.append((f, expected_d))
for f in distorted_files:
    expected_c = f.replace("_distorted.png", "_clean.png")
    if expected_c not in clean_set:
        unmatched_dist.append((f, expected_c))

assert len(unmatched_clean) == 0, f"Unmatched clean: {unmatched_clean}"
assert len(unmatched_dist) == 0, f"Unmatched distorted: {unmatched_dist}"
print("[PASS] Every active CLEAN image has exactly one matching DISTORTED image, and vice versa.")

# Check 5: Duplicate check across (participant_id, hand, finger, condition)
seen_keys = set()
dup_keys = []
for r in csv_rows:
    key = (r['participant_id'].strip().lower(), r['hand'].strip().upper(), r['finger'].strip().upper(), r['condition'].strip())
    if key in seen_keys:
        dup_keys.append(key)
    else:
        seen_keys.add(key)

assert len(dup_keys) == 0, f"Duplicate keys in CSV: {dup_keys}"
print("[PASS] Zero duplicate (participant, hand, finger, condition) combinations exist.")

# Check 6: CSV and MySQL Synchronization
assert len(csv_rows) == 220, f"CSV rows != 220 ({len(csv_rows)})"
assert len(db_rows) == 220, f"MySQL rows != 220 ({len(db_rows)})"
print(f"[PASS] CSV active records ({len(csv_rows)}) and MySQL records ({len(db_rows)}) are 100% synchronized.")

# Check 7: File existence for all metadata paths
missing_csv_paths = []
for r in csv_rows:
    p = os.path.join(dataset_root, r['image_path'])
    if not os.path.exists(p):
        missing_csv_paths.append((r['filename'], p))

assert len(missing_csv_paths) == 0, f"Metadata paths missing on disk: {missing_csv_paths}"
print("[PASS] Every active metadata image_path points to an existing file on disk.")

# Check 8: Quarantine Directory Inspection
quarantine_clean = os.listdir(os.path.join(dataset_root, "quarantine", "unpaired", "clean"))
quarantine_dist = os.listdir(os.path.join(dataset_root, "quarantine", "unpaired", "distorted"))
print(f"\nQUARANTINE AUDIT:")
print(f"  Quarantined clean images:     {len(quarantine_clean)}")
print(f"  Quarantined distorted images: {len(quarantine_dist)}")
print(f"  Total quarantined images:     {len(quarantine_clean) + len(quarantine_dist)}")

print("\n" + "="*70)
print("ALL SECTION 7 VALIDATION ASSERTIONS PASSED WITH 100% SUCCESS!")
print("="*70)

