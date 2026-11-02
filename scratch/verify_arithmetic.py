import os
import sys
import csv
from collections import defaultdict

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

print(f"Total rows in metadata.csv: {len(rows)}")

participant_counts = defaultdict(lambda: {"clean": 0, "distorted": 0, "rows": []})
for idx, r in enumerate(rows, start=2):
    p = r['participant_id']
    cond = "clean" if r['condition'] == '0' else "distorted"
    participant_counts[p][cond] += 1
    participant_counts[p]["rows"].append((idx, r))

print(f"\nTotal participants in metadata.csv: {len(participant_counts)}")
print("-" * 65)
print(f"{'Participant':<15} | {'Clean':<6} | {'Distorted':<10} | {'Total Rows':<10}")
print("-" * 65)
for p in sorted(participant_counts.keys(), key=lambda s: s.upper()):
    c = participant_counts[p]['clean']
    d = participant_counts[p]['distorted']
    tot = c + d
    flag = ""
    if p.upper() == "KUSUMA":
        flag = " <- 2 duplicate clean rows!"
    elif p.upper() == "ASHITTA":
        flag = " <- 0 distorted, files missing on disk!"
    print(f"{p:<15} | {c:<6} | {d:<10} | {tot:<10}{flag}")

print("-" * 65)

