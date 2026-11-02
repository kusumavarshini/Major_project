"""
Participant-Grouped Cross-Validation Split Module for Real V2
============================================================
Creates 5-fold GroupKFold splits strictly on the 22 development participants.
Zero inclusion of test participants. Preserves matched Clean-Distorted pairs.
"""

import os
import sys
import csv
from collections import defaultdict
from typing import Dict, List, Any
from sklearn.model_selection import GroupKFold

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFESTS_DIR = os.path.join(EXPERIMENT_DIR, "manifests")
os.makedirs(MANIFESTS_DIR, exist_ok=True)

V1_DIR = os.path.join(REPO_ROOT, "experiments", "classifier_real_v1")
V1_TRAIN = os.path.join(V1_DIR, "train_manifest.csv")
V1_VAL = os.path.join(V1_DIR, "val_manifest.csv")
V1_TEST = os.path.join(V1_DIR, "test_manifest.csv")


def generate_cv_manifests() -> Dict[str, Any]:
    with open(V1_TRAIN, "r", encoding="utf-8") as f:
        train_rows = list(csv.DictReader(f))
    with open(V1_VAL, "r", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))
    with open(V1_TEST, "r", encoding="utf-8") as f:
        test_rows = list(csv.DictReader(f))

    # Dev set = Train + Val (22 participants, 190 images = 95 pairs)
    dev_rows = train_rows + val_rows
    fieldnames = list(dev_rows[0].keys())

    # Save dev_manifest.csv
    dev_csv_path = os.path.join(MANIFESTS_DIR, "dev_manifest.csv")
    with open(dev_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(dev_rows)

    # Save test_manifest.csv (untouched copy of V1 test set)
    test_csv_path = os.path.join(MANIFESTS_DIR, "test_manifest.csv")
    with open(test_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)

    # Group dev set by pair_id to keep pairs intact
    pairs_dict = defaultdict(list)
    for r in dev_rows:
        pairs_dict[r['pair_id']].append(r)

    pair_keys = sorted(pairs_dict.keys())
    pair_groups = [pairs_dict[k][0]['person_id'] for k in pair_keys]
    dummy_labels = [0 for _ in pair_keys]

    gkf = GroupKFold(n_splits=5)
    fold_info = []

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(pair_keys, dummy_labels, groups=pair_groups), start=1):
        tr_pairs = [pair_keys[i] for i in tr_idx]
        va_pairs = [pair_keys[i] for i in val_idx]

        tr_records = [r for k in tr_pairs for r in pairs_dict[k]]
        va_records = [r for k in va_pairs for r in pairs_dict[k]]

        tr_persons = sorted(list(set(r['person_id'] for r in tr_records)))
        va_persons = sorted(list(set(r['person_id'] for r in va_records)))

        # Assert zero overlap
        overlap = set(tr_persons) & set(va_persons)
        assert len(overlap) == 0, f"Leakage in Fold {fold_idx}: {overlap}"

        # Write fold CSVs
        tr_path = os.path.join(MANIFESTS_DIR, f"cv_fold_{fold_idx}_train.csv")
        va_path = os.path.join(MANIFESTS_DIR, f"cv_fold_{fold_idx}_val.csv")

        with open(tr_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(tr_records)

        with open(va_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(va_records)

        fold_info.append({
            "fold": fold_idx,
            "train_persons": tr_persons,
            "train_pairs": len(tr_pairs),
            "train_images": len(tr_records),
            "val_persons": va_persons,
            "val_pairs": len(va_pairs),
            "val_images": len(va_records),
            "train_manifest": tr_path,
            "val_manifest": va_path
        })
        print(f"Generated Fold {fold_idx}: Train {len(tr_persons)} persons ({len(tr_records)} imgs) | Val {len(va_persons)} persons ({len(va_records)} imgs)")

    return {
        "dev_samples": len(dev_rows),
        "test_samples": len(test_rows),
        "dev_persons": sorted(list(set(r['person_id'] for r in dev_rows))),
        "test_persons": sorted(list(set(r['person_id'] for r in test_rows))),
        "folds": fold_info
    }


if __name__ == "__main__":
    generate_cv_manifests()

