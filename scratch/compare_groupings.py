import os
import sys
import csv
from collections import defaultdict
import cv2

clean_dir = os.path.abspath("FingerSure_Dataset/clean")
distorted_dir = os.path.abspath("FingerSure_Dataset/distorted")

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

# Let's check each participant, hand, finger
# What if we group by (participant_id.lower(), hand.upper(), finger.upper())?
pairs_by_lower = defaultdict(lambda: {"clean": [], "distorted": []})
pairs_by_exact = defaultdict(lambda: {"clean": [], "distorted": []})

for idx, r in enumerate(csv_rows, start=2):
    p_exact = r['participant_id'].strip()
    p_lower = p_exact.lower()
    hand = r['hand'].strip().upper()
    finger = r['finger'].strip().upper()
    cond = r['condition'].strip()
    
    side = "clean" if cond == "0" else "distorted"
    pairs_by_lower[(p_lower, hand, finger)][side].append((idx, r))
    pairs_by_exact[(p_exact, hand, finger)][side].append((idx, r))

print(f"Total lower keys: {len(pairs_by_lower)}")
print(f"Total exact keys: {len(pairs_by_exact)}")

# Let's see the differences between lower keys and exact keys
diff_keys = []
for (p_exact, hand, finger), data in pairs_by_exact.items():
    p_lower = p_exact.lower()
    lower_data = pairs_by_lower[(p_lower, hand, finger)]
    if len(lower_data['clean']) != len(data['clean']) or len(lower_data['distorted']) != len(data['distorted']):
        diff_keys.append((p_exact, p_lower, hand, finger, data, lower_data))

print(f"Keys where exact differs from lower: {len(diff_keys)}")
for p_exact, p_lower, hand, finger, data, lower_data in diff_keys:
    print(f"  Exact key: ('{p_exact}', '{hand}', '{finger}') -> clean={len(data['clean'])}, dist={len(data['distorted'])}")
    print(f"  Lower key: ('{p_lower}', '{hand}', '{finger}') -> clean={len(lower_data['clean'])}, dist={len(lower_data['distorted'])}")


