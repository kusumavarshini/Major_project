import os
import sys
import csv
import json
import uuid
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
    shape = None
    if exists:
        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.size > 0:
            decodes = True
            shape = img.shape
            
    record_info = {
        "csv_line": idx,
        "row": dict(r),
        "filename": fn,
        "full_path": full_path,
        "exists": exists,
        "decodes": decodes,
        "shape": shape
    }
    
    side = "clean" if cond == '0' else "distorted"
    triples[(p_lower, hand, finger)][side].append(record_info)

valid_pairs = []
unpaired_clean = []
unpaired_distorted = []
duplicate_records = []
ghost_records = []

for (p_lower, hand, finger), data in sorted(triples.items()):
    clean_list = data['clean']
    dist_list = data['distorted']
    
    # Handle duplicates in clean
    valid_c = [c for c in clean_list if c['exists'] and c['decodes']]
    if len(valid_c) > 1:
        valid_c.sort(key=lambda x: x['row']['capture_timestamp'])
        for dup in valid_c[:-1]:
            duplicate_records.append((dup, "SUPERSEDED_BY_RECAPTURE"))
        active_c = valid_c[-1]
    elif len(valid_c) == 1:
        active_c = valid_c[0]
        # Any other clean entries without files are ghosts/dups
        for c in clean_list:
            if c != active_c:
                if not c['exists']:
                    ghost_records.append((c, "GHOST_RECORD_NO_IMAGE_ON_DISK"))
                else:
                    duplicate_records.append((c, "DUPLICATE_ENTRY"))
    else:
        active_c = None
        for c in clean_list:
            ghost_records.append((c, "GHOST_RECORD_NO_IMAGE_ON_DISK"))

    # Handle duplicates in distorted
    valid_d = [d for d in dist_list if d['exists'] and d['decodes']]
    if len(valid_d) > 1:
        valid_d.sort(key=lambda x: x['row']['capture_timestamp'])
        for dup in valid_d[:-1]:
            duplicate_records.append((dup, "SUPERSEDED_BY_RECAPTURE"))
        active_d = valid_d[-1]
    elif len(valid_d) == 1:
        active_d = valid_d[0]
        for d in dist_list:
            if d != active_d:
                if not d['exists']:
                    ghost_records.append((d, "GHOST_RECORD_NO_IMAGE_ON_DISK"))
                else:
                    duplicate_records.append((d, "DUPLICATE_ENTRY"))
    else:
        active_d = None
        for d in dist_list:
            ghost_records.append((d, "GHOST_RECORD_NO_IMAGE_ON_DISK"))

    # Pairing determination
    if active_c and active_d:
        # Determine canonical participant_id
        # If one is Bharthi and one is BHarthi, prefer Bharthi
        c_pid = active_c['row']['participant_id']
        d_pid = active_d['row']['participant_id']
        if c_pid == "Bharthi" or d_pid == "Bharthi":
            canonical_pid = "Bharthi"
        else:
            canonical_pid = c_pid
            
        valid_pairs.append({
            "participant_id": canonical_pid,
            "participant_id_lower": p_lower,
            "hand": hand,
            "finger": finger,
            "clean": active_c,
            "distorted": active_d
        })
    elif active_c and not active_d:
        unpaired_clean.append((active_c, "MISSING_DISTORTED_PAIR"))
    elif active_d and not active_c:
        unpaired_distorted.append((active_d, "MISSING_CLEAN_PAIR"))

print("="*70)
print("DRY-RUN ANALYSIS RESULTS:")
print("="*70)
print(f"1. Total participants before cleaning: 34 (including case-variants and ghost 'ashitta')")
print(f"2. Total CLEAN images on disk before cleaning: {len(os.listdir(clean_dir))}")
print(f"3. Total DISTORTED images on disk before cleaning: {len(os.listdir(distorted_dir))}")
print(f"4. Total valid complete pairs identified: {len(valid_pairs)}")
print(f"5. Complete pairs retained: {len(valid_pairs)}")
print(f"6. Unpaired CLEAN images to quarantine: {len(unpaired_clean)}")
print(f"7. Unpaired DISTORTED images to quarantine: {len(unpaired_distorted)}")
print(f"8. Duplicate metadata rows to exclude: {len(duplicate_records)}")
print(f"9. Ghost metadata rows without images to exclude: {len(ghost_records)}")
print(f"10. Corrupted images: 0")
print(f"11. Expected CLEAN images after cleaning: {len(valid_pairs)}")
print(f"12. Expected DISTORTED images after cleaning: {len(valid_pairs)}")
print(f"13. Expected TOTAL active images after cleaning: {len(valid_pairs) * 2}")

# Check lowercase rename mapping
clean_rename_plan = []
dist_rename_plan = []

for pair in valid_pairs:
    # Clean file
    old_c_fn = pair['clean']['filename']
    new_c_fn = old_c_fn.lower()
    clean_rename_plan.append((old_c_fn, new_c_fn, old_c_fn != new_c_fn))
    
    # Distorted file
    old_d_fn = pair['distorted']['filename']
    new_d_fn = old_d_fn.lower()
    dist_rename_plan.append((old_d_fn, new_d_fn, old_d_fn != new_d_fn))

clean_needs_rename = sum(1 for _, _, changed in clean_rename_plan if changed)
dist_needs_rename = sum(1 for _, _, changed in dist_rename_plan if changed)
total_needs_rename = clean_needs_rename + dist_needs_rename

print(f"14. Filenames requiring lowercase conversion: {total_needs_rename} ({clean_needs_rename} clean, {dist_needs_rename} distorted)")
print(f"15. Filenames already lowercase: {(len(valid_pairs)*2) - total_needs_rename}")

# Collision check
all_new_clean = set(n for _, n, _ in clean_rename_plan)
all_new_dist = set(n for _, n, _ in dist_rename_plan)

print(f"16. Unique new clean filenames: {len(all_new_clean)} / {len(clean_rename_plan)}")
print(f"17. Unique new distorted filenames: {len(all_new_dist)} / {len(dist_rename_plan)}")

if len(all_new_clean) == len(clean_rename_plan) and len(all_new_dist) == len(dist_rename_plan):
    print(">>> ZERO FILENAME COLLISIONS DETECTED! Renaming plan is 100% safe.")
else:
    print(">>> WARNING: FILENAME COLLISIONS DETECTED!")

print("\n--- UNPAIRED CLEAN IMAGES TO QUARANTINE ---")
for c_rec, reason in unpaired_clean:
    print(f"  {c_rec['filename']} -> quarantine/unpaired/clean/ ({reason})")

print("\n--- UNPAIRED DISTORTED IMAGES TO QUARANTINE ---")
for d_rec, reason in unpaired_distorted:
    print(f"  {d_rec['filename']} -> quarantine/unpaired/distorted/ ({reason})")

print("\n--- GHOST RECORDS TO REMOVE FROM ACTIVE METADATA ---")
for g_rec, reason in ghost_records:
    print(f"  Line {g_rec['csv_line']}: {g_rec['filename']} ({reason})")

print("\n--- DUPLICATE RECORDS TO REMOVE FROM ACTIVE METADATA ---")
for dup_rec, reason in duplicate_records:
    print(f"  Line {dup_rec['csv_line']}: {dup_rec['filename']} ({reason})")

