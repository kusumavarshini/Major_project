import os
import sys
import csv
from collections import defaultdict
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

# Read train and val manifests from real_v1
train_path = "experiments/classifier_real_v1/train_manifest.csv"
val_path = "experiments/classifier_real_v1/val_manifest.csv"

with open(train_path, "r", encoding="utf-8") as f:
    train_rows = list(csv.DictReader(f))
with open(val_path, "r", encoding="utf-8") as f:
    val_rows = list(csv.DictReader(f))

dev_rows = train_rows + val_rows
print(f"Total dev samples: {len(dev_rows)} ({len(dev_rows)//2} pairs)")

# Check participants
dev_persons = sorted(list(set(r['person_id'] for r in dev_rows)))
print(f"Total dev persons: {len(dev_persons)}: {dev_persons}")

# We want 5 folds grouped by person_id, keeping clean-distorted pairs together
# Group rows by pair_id
pairs_dict = defaultdict(list)
for r in dev_rows:
    pairs_dict[r['pair_id']].append(r)

pair_keys = sorted(pairs_dict.keys())
print(f"Unique dev pairs: {len(pair_keys)}")

pair_groups = [pairs_dict[k][0]['person_id'] for k in pair_keys]
pair_labels = [0 for k in pair_keys] # all pairs have balanced classes (1 clean, 1 distorted)

# GroupKFold with 5 splits
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(pair_keys, pair_labels, groups=pair_groups), start=1):
    val_pairs = [pair_keys[i] for i in val_idx]
    val_persons = sorted(list(set(pairs_dict[k][0]['person_id'] for k in val_pairs)))
    val_images = sum(len(pairs_dict[k]) for k in val_pairs)
    train_pairs = [pair_keys[i] for i in train_idx]
    train_persons = sorted(list(set(pairs_dict[k][0]['person_id'] for k in train_pairs)))
    train_images = sum(len(pairs_dict[k]) for k in train_pairs)
    print(f"Fold {fold}:")
    print(f"  Train: {len(train_persons):2d} persons, {len(train_pairs):2d} pairs ({train_images:3d} images)")
    print(f"  Val:   {len(val_persons):2d} persons, {len(val_pairs):2d} pairs ({val_images:3d} images) -> {val_persons}")

