import os
import sys
import csv

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

target_participants = ["JYOTHSNIKA", "MEGHANA", "HRUDHYA", "MANGALAMMA", "Bharthi", "BHarthi"]

for tp in target_participants:
    print(f"\n==================== Participant: {tp} ====================")
    for idx, r in enumerate(csv_rows, start=2):
        if r['participant_id'].upper() == tp.upper():
            print(f"Line {idx:3d}: hand={r['hand']:<5} finger={r['finger']:<7} cond={r['condition']} file={r['filename']} time={r['capture_timestamp']}")

