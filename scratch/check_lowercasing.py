import os
import sys
import csv
from collections import defaultdict
import cv2

clean_dir = os.path.abspath("FingerSure_Dataset/clean")
distorted_dir = os.path.abspath("FingerSure_Dataset/distorted")

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

# Group by (participant_id_lower, hand_upper, finger_upper)
triples = defaultdict(lambda: {"clean": [], "distorted": []})

for idx, r in enumerate(csv_rows, start=2):
    p_id = r['participant_id'].strip()
    p_lower = p_id.lower()
    hand = r['hand'].strip().upper()
    finger = r['finger'].strip().upper()
    cond = r['condition'].strip()
    
    folder = clean_dir if cond == '0' else distorted_dir
    fn = r['filename'].strip()
    full_path = os.path.join(folder, fn)
    
    exists = os.path.exists(full_path)
    decodes = False
    if exists:
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            decodes = True
            
    record_info = {
        "csv_line": idx,
        "row": r,
        "filename": fn,
        "full_path": full_path,
        "exists": exists,
        "decodes": decodes
    }
    
    side = "clean" if cond == '0' else "distorted"
    triples[(p_lower, hand, finger)][side].append(record_info)

valid_pairs = []
for (p_lower, hand, finger), data in sorted(triples.items()):
    clean_list = [c for c in data['clean'] if c['exists'] and c['decodes']]
    dist_list = [d for d in data['distorted'] if d['exists'] and d['decodes']]
    
    if len(clean_list) >= 1 and len(dist_list) >= 1:
        # Keep latest of each if duplicate
        clean_list.sort(key=lambda x: x['row']['capture_timestamp'])
        dist_list.sort(key=lambda x: x['row']['capture_timestamp'])
        valid_pairs.append({
            "participant_id": clean_list[-1]['row']['participant_id'],
            "participant_id_lower": p_lower,
            "hand": hand,
            "finger": finger,
            "clean": clean_list[-1],
            "distorted": dist_list[-1]
        })

print(f"Total valid pairs: {len(valid_pairs)}")

clean_lower = defaultdict(list)
dist_lower = defaultdict(list)

for pair in valid_pairs:
    c_fn = pair['clean']['filename']
    d_fn = pair['distorted']['filename']
    
    clean_lower[c_fn.lower()].append(c_fn)
    dist_lower[d_fn.lower()].append(d_fn)

print(f"Unique lower clean filenames: {len(clean_lower)}")
print(f"Unique lower distorted filenames: {len(dist_lower)}")

c_collisions = {k: v for k, v in clean_lower.items() if len(v) > 1}
d_collisions = {k: v for k, v in dist_lower.items() if len(v) > 1}

print(f"Clean collisions: {len(c_collisions)}")
print(f"Distorted collisions: {len(d_collisions)}")

# Correspondence check
mismatches = []
for pair in valid_pairs:
    c_lower = pair['clean']['filename'].lower()
    d_lower = pair['distorted']['filename'].lower()
    expected_d = c_lower.replace("_clean.png", "_distorted.png")
    if d_lower != expected_d:
        mismatches.append((pair['participant_id'], pair['hand'], pair['finger'], c_lower, d_lower, expected_d))

print(f"Filename correspondence mismatches (clean vs distorted): {len(mismatches)}")
for m in mismatches:
    print(f"  Mismatch: pid={m[0]}, hand={m[1]}, finger={m[2]}")
    print(f"    Clean:     {m[3]}")
    print(f"    Distorted: {m[4]}")
    print(f"    Expected:  {m[5]}")
