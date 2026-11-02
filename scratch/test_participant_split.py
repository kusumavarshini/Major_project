import os
import sys
import csv
import random
from collections import defaultdict

# Read valid_pairs_manifest.csv
with open("FingerSure_Dataset/valid_pairs_manifest.csv", "r", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))

print(f"Total pairs: {len(pairs)}")

# Group pairs by lower participant ID
by_person = defaultdict(list)
for p in pairs:
    person_key = p['participant_id'].strip().lower()
    by_person[person_key].append(p)

sorted_persons = sorted(by_person.keys())
print(f"Total unique persons: {len(sorted_persons)}")

# Print pairs per person
print("\nPairs per person:")
for person in sorted_persons:
    pair_list = by_person[person]
    raw_pids = set(p['participant_id'] for p in pair_list)
    print(f"  {person:<15}: {len(pair_list):2d} pairs ({len(pair_list)*2:2d} images) [raw pids: {raw_pids}]")

# Let's see what random shuffle with seed 42 does
rng = random.Random(42)
shuffled_persons = list(sorted_persons)
rng.shuffle(shuffled_persons)

# Let's test a 70:15:15 split on 26 persons:
# e.g. 18 train, 4 val, 4 test
train_persons = shuffled_persons[:18]
val_persons = shuffled_persons[18:22]
test_persons = shuffled_persons[22:]

print("\n--- Split: 18 / 4 / 4 ---")
for name, p_list in [("Train", train_persons), ("Val", val_persons), ("Test", test_persons)]:
    n_pairs = sum(len(by_person[p]) for p in p_list)
    n_images = n_pairs * 2
    print(f"  {name:<5}: {len(p_list):2d} participants, {n_pairs:3d} pairs, {n_images:3d} images ({n_images/220*100:.1f}%) -> {sorted(p_list)}")

