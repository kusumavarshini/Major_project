import os
import sys
import csv
from collections import defaultdict
import cv2

clean_dir = os.path.abspath("FingerSure_Dataset/clean")
distorted_dir = os.path.abspath("FingerSure_Dataset/distorted")

# Read metadata.csv
with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

# Map disk files (case sensitive and case-insensitive)
clean_disk = {f: os.path.join(clean_dir, f) for f in os.listdir(clean_dir)}
distorted_disk = {f: os.path.join(distorted_dir, f) for f in os.listdir(distorted_dir)}

clean_disk_lower = {f.lower(): (f, os.path.join(clean_dir, f)) for f in clean_disk}
distorted_disk_lower = {f.lower(): (f, os.path.join(distorted_dir, f)) for f in distorted_disk}

print(f"Total rows in metadata: {len(csv_rows)}")
print(f"Clean files on disk: {len(clean_disk)}")
print(f"Distorted files on disk: {len(distorted_disk)}")

# Group metadata entries by (participant_id, hand, finger)
# And also by (participant_id.upper(), hand, finger)
by_exact_triple = defaultdict(lambda: {"clean": [], "distorted": []})
by_upper_triple = defaultdict(lambda: {"clean": [], "distorted": []})

for idx, r in enumerate(csv_rows, start=2):
    p_id = r['participant_id'].strip()
    hand = r['hand'].strip().upper()
    finger = r['finger'].strip().upper()
    cond = r['condition'].strip()
    
    triple_exact = (p_id, hand, finger)
    triple_upper = (p_id.upper(), hand, finger)
    
    side = "clean" if cond == "0" else "distorted"
    by_exact_triple[triple_exact][side].append((idx, r))
    by_upper_triple[triple_upper][side].append((idx, r))

print(f"\nUnique (participant_id, hand, finger) exact triples: {len(by_exact_triple)}")
print(f"Unique (participant_id.upper(), hand, finger) upper triples: {len(by_upper_triple)}")

# Check if any upper triple has different exact participant_ids
for t_up, data in by_upper_triple.items():
    all_pids = set(r['participant_id'] for s in ["clean", "distorted"] for idx, r in data[s])
    if len(all_pids) > 1:
        print(f"Upper triple {t_up} has multiple participant IDs: {all_pids}")

print("\n--- Let's inspect unpaired entries in by_exact_triple ---")
unpaired_exact = []
complete_exact = []
for triple, data in by_exact_triple.items():
    c_count = len(data['clean'])
    d_count = len(data['distorted'])
    if c_count == 0 or d_count == 0:
        unpaired_exact.append((triple, c_count, d_count, data))
    else:
        complete_exact.append((triple, c_count, d_count, data))

print(f"Complete pairs (c >= 1 and d >= 1): {len(complete_exact)}")
print(f"Unpaired triples (c == 0 or d == 0): {len(unpaired_exact)}")

for triple, c_count, d_count, data in unpaired_exact:
    p_id, hand, finger = triple
    print(f"\nUnpaired Triple: {p_id} {hand} {finger} -> Clean: {c_count}, Distorted: {d_count}")
    for side in ["clean", "distorted"]:
        for idx, r in data[side]:
            fn = r['filename']
            # Check if file exists on disk
            folder = "clean" if side == "clean" else "distorted"
            full_p = os.path.join("FingerSure_Dataset", folder, fn)
            exists = os.path.exists(full_p)
            print(f"  {side.upper()}: line {idx}, file='{fn}', exists_on_disk={exists}")
