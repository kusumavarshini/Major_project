import os
import sys
import csv
from collections import defaultdict

with open("FingerSure_Dataset/valid_pairs_manifest.csv", "r", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))

by_person = defaultdict(list)
for p in pairs:
    person_key = p['participant_id'].strip().lower()
    by_person[person_key].append(p)

sorted_persons = sorted(by_person.keys())
import random
rng = random.Random(42)
shuffled_persons = list(sorted_persons)
rng.shuffle(shuffled_persons)

splits = {
    "Train": shuffled_persons[:18],
    "Val": shuffled_persons[18:22],
    "Test": shuffled_persons[22:]
}

for split_name, person_list in splits.items():
    finger_counts = defaultdict(int)
    hand_counts = defaultdict(int)
    total_pairs = 0
    for p in person_list:
        for pair in by_person[p]:
            finger_counts[f"{pair['hand']}_{pair['finger']}"] += 1
            hand_counts[pair['hand']] += 1
            total_pairs += 1
    print(f"\n--- {split_name} Split ({len(person_list)} persons, {total_pairs} pairs, {total_pairs*2} images) ---")
    print(f"  Hands: Right={hand_counts['RIGHT']}, Left={hand_counts['LEFT']}")
    print("  Fingers:")
    for finger in ['RIGHT_THUMB', 'RIGHT_INDEX', 'RIGHT_MIDDLE', 'RIGHT_RING', 'RIGHT_LITTLE',
                   'LEFT_THUMB', 'LEFT_INDEX', 'LEFT_MIDDLE', 'LEFT_RING', 'LEFT_LITTLE']:
        cnt = finger_counts[finger]
        print(f"    {finger:<15}: {cnt:2d} pairs ({cnt*2:2d} images)")

