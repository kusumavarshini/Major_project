import os
import sys
import csv
from collections import defaultdict
import cv2

clean_dir = os.path.abspath("FingerSure_Dataset/clean")
distorted_dir = os.path.abspath("FingerSure_Dataset/distorted")

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

# Let's inspect all rows and check against disk
# Group by (participant_id_lower, hand_upper, finger_upper)
triples = defaultdict(lambda: {"clean": [], "distorted": []})

for idx, r in enumerate(csv_rows, start=2):
    p_id = r['participant_id'].strip()
    p_lower = p_id.lower()
    hand = r['hand'].strip().upper()
    finger = r['finger'].strip().upper()
    cond = r['condition'].strip()
    
    # Check physical file on disk
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
        "row": r,
        "filename": fn,
        "full_path": full_path,
        "exists": exists,
        "decodes": decodes,
        "shape": shape
    }
    
    side = "clean" if cond == '0' else "distorted"
    triples[(p_lower, hand, finger)][side].append(record_info)

# Now evaluate each triple
valid_pairs = []
unpaired_records = []
duplicate_records = []
corrupted_records = []
ghost_records = []

for (p_lower, hand, finger), data in sorted(triples.items()):
    clean_list = data['clean']
    dist_list = data['distorted']
    
    # Check duplicates within clean
    # If more than 1 clean record, check if they are duplicates of the same file
    if len(clean_list) > 1:
        # e.g. Kusuma lines 2 and 8
        print(f"Multiple clean records for ({p_lower}, {hand}, {finger}): {len(clean_list)} records")
        # Keep the latest one that exists and decodes
        valid_clean_entries = [c for c in clean_list if c['exists'] and c['decodes']]
        if len(valid_clean_entries) > 1:
            # Sort by timestamp or line, keep the latest
            valid_clean_entries.sort(key=lambda x: x['row']['capture_timestamp'])
            for dup in valid_clean_entries[:-1]:
                duplicate_records.append((dup, "DUPLICATE_SUPERSEDED_BY_RECAPTURE"))
            clean_list = [valid_clean_entries[-1]]
        elif len(valid_clean_entries) == 1:
            for c in clean_list:
                if c != valid_clean_entries[0]:
                    duplicate_records.append((c, "DUPLICATE_SUPERSEDED"))
            clean_list = valid_clean_entries
            
    # Check duplicates within distorted
    if len(dist_list) > 1:
        print(f"Multiple distorted records for ({p_lower}, {hand}, {finger}): {len(dist_list)} records")
        valid_dist_entries = [d for d in dist_list if d['exists'] and d['decodes']]
        if len(valid_dist_entries) > 1:
            valid_dist_entries.sort(key=lambda x: x['row']['capture_timestamp'])
            for dup in valid_dist_entries[:-1]:
                duplicate_records.append((dup, "DUPLICATE_SUPERSEDED"))
            dist_list = [valid_dist_entries[-1]]
        elif len(valid_dist_entries) == 1:
            for d in dist_list:
                if d != valid_dist_entries[0]:
                    duplicate_records.append((d, "DUPLICATE_SUPERSEDED"))
            dist_list = valid_dist_entries

    # Now we have 0 or 1 clean, and 0 or 1 dist
    has_clean = len(clean_list) == 1
    has_dist = len(dist_list) == 1
    
    c_rec = clean_list[0] if has_clean else None
    d_rec = dist_list[0] if has_dist else None
    
    # Check if disk files exist
    c_ok = c_rec and c_rec['exists'] and c_rec['decodes']
    d_ok = d_rec and d_rec['exists'] and d_rec['decodes']
    
    if c_rec and not c_rec['exists']:
        ghost_records.append((c_rec, "FILE_NOT_FOUND_ON_DISK"))
    if d_rec and not d_rec['exists']:
        ghost_records.append((d_rec, "FILE_NOT_FOUND_ON_DISK"))
        
    if c_ok and d_ok:
        valid_pairs.append({
            "participant_id": c_rec['row']['participant_id'], # or canonical
            "participant_id_lower": p_lower,
            "hand": hand,
            "finger": finger,
            "clean": c_rec,
            "distorted": d_rec
        })
    else:
        # Unpaired!
        if c_ok and not d_ok:
            reason = "MISSING_DISTORTED_PAIR" if not d_rec else ("DISTORTED_FILE_NOT_FOUND" if not d_rec['exists'] else "DISTORTED_CORRUPTED")
            unpaired_records.append((c_rec, reason))
        elif d_ok and not c_ok:
            reason = "MISSING_CLEAN_PAIR" if not c_rec else ("CLEAN_FILE_NOT_FOUND" if not c_rec['exists'] else "CLEAN_CORRUPTED")
            unpaired_records.append((d_rec, reason))

print("\n" + "="*60)
print(f"SUMMARY OF ANALYSIS:")
print(f"Valid Complete Pairs: {len(valid_pairs)}")
print(f"Unpaired Image Records: {len(unpaired_records)}")
print(f"Duplicate Records in CSV: {len(duplicate_records)}")
print(f"Ghost Records (no file on disk): {len(ghost_records)}")
print(f"Corrupted Images: {len(corrupted_records)}")
print("="*60)

print("\n--- UNPAIRED RECORDS DETAILS ---")
for rec, reason in unpaired_records:
    r = rec['row']
    cond_name = "CLEAN" if r['condition'] == '0' else "DISTORTED"
    print(f"  {cond_name:<9} | Line {rec['csv_line']:3d} | {r['participant_id']} {r['hand']} {r['finger']} | File: {rec['filename']} | Reason: {reason}")

print("\n--- DUPLICATE RECORDS DETAILS ---")
for rec, reason in duplicate_records:
    r = rec['row']
    print(f"  Line {rec['csv_line']:3d} | {r['participant_id']} {r['hand']} {r['finger']} | File: {rec['filename']} | Reason: {reason}")

print("\n--- GHOST RECORDS DETAILS ---")
for rec, reason in ghost_records:
    r = rec['row']
    print(f"  Line {rec['csv_line']:3d} | {r['participant_id']} {r['hand']} {r['finger']} | File: {rec['filename']} | Reason: {reason}")

