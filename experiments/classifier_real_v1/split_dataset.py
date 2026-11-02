"""
Dataset Splitting Module for FingerSure Real Dataset CNN Experiment (classifier_real_v1)
========================================================================================
- Splits the cleaned physical dataset (FingerSure_Dataset/) using participant-level isolation.
- Ratio: ~70:15:15 based on participant IDs.
- Random Seed: 42 (strictly fixed).
- Preserves paired Clean-Distorted relationship (zero data leakage across splits).
- Generates train_manifest.csv, val_manifest.csv, test_manifest.csv.
"""

import os
import sys
import csv
import random
from collections import defaultdict
from typing import Dict, List, Any

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(REPO_ROOT, "FingerSure_Dataset")
VALID_PAIRS_CSV = os.path.join(DATASET_DIR, "valid_pairs_manifest.csv")
METADATA_CSV = os.path.join(DATASET_DIR, "metadata.csv")


def load_dataset() -> List[Dict[str, Any]]:
    if not os.path.exists(VALID_PAIRS_CSV):
        raise FileNotFoundError(f"Valid pairs manifest not found at: {VALID_PAIRS_CSV}")
    
    with open(VALID_PAIRS_CSV, "r", encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    return pairs


def create_participant_splits(seed: int = 42) -> Dict[str, Any]:
    pairs = load_dataset()
    print(f"[Dataset Loader] Loaded {len(pairs)} complete fingerprint pairs.")
    
    # Group pairs by canonical person (case-insensitive participant_id)
    by_person = defaultdict(list)
    for idx, p in enumerate(pairs):
        p['pair_id'] = f"PAIR_{idx+1:03d}"
        person_key = p['participant_id'].strip().lower()
        by_person[person_key].append(p)
        
    unique_persons = sorted(by_person.keys())
    print(f"[Dataset Loader] Found {len(unique_persons)} unique participants.")
    
    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled_persons = list(unique_persons)
    rng.shuffle(shuffled_persons)
    
    # 70:15:15 participant split:
    # 26 unique persons: 18 Train (69.2%), 4 Val (15.4%), 4 Test (15.4%)
    train_persons = shuffled_persons[:18]
    val_persons = shuffled_persons[18:22]
    test_persons = shuffled_persons[22:]
    
    split_assignment = {
        "train": train_persons,
        "val": val_persons,
        "test": test_persons
    }
    
    # Build per-image manifest records
    split_records = {"train": [], "val": [], "test": []}
    
    for split_name, person_list in split_assignment.items():
        for person in person_list:
            for pair in by_person[person]:
                # Clean sample (label = 0)
                clean_full = os.path.join(DATASET_DIR, pair['clean_filepath'])
                if not os.path.exists(clean_full):
                    raise FileNotFoundError(f"Clean file missing on disk: {clean_full}")
                clean_record = {
                    "pair_id": pair['pair_id'],
                    "participant_id": pair['participant_id'],
                    "person_id": person,
                    "hand": pair['hand'],
                    "finger": pair['finger'],
                    "condition": "0",
                    "label": "0",
                    "class_name": "clean",
                    "filename": pair['clean_filename'],
                    "image_path": pair['clean_filepath'],
                    "full_filepath": clean_full,
                    "split": split_name
                }
                split_records[split_name].append(clean_record)
                
                # Distorted sample (label = 1)
                dist_full = os.path.join(DATASET_DIR, pair['distorted_filepath'])
                if not os.path.exists(dist_full):
                    raise FileNotFoundError(f"Distorted file missing on disk: {dist_full}")
                dist_record = {
                    "pair_id": pair['pair_id'],
                    "participant_id": pair['participant_id'],
                    "person_id": person,
                    "hand": pair['hand'],
                    "finger": pair['finger'],
                    "condition": "1",
                    "label": "1",
                    "class_name": "distorted",
                    "filename": pair['distorted_filename'],
                    "image_path": pair['distorted_filepath'],
                    "full_filepath": dist_full,
                    "split": split_name
                }
                split_records[split_name].append(dist_record)

    # Save manifests
    manifest_paths = {}
    fieldnames = [
        "pair_id", "participant_id", "person_id", "hand", "finger",
        "condition", "label", "class_name", "filename", "image_path",
        "full_filepath", "split"
    ]
    
    for split_name in ["train", "val", "test"]:
        out_path = os.path.join(EXPERIMENT_DIR, f"{split_name}_manifest.csv")
        records = split_records[split_name]
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)
        manifest_paths[split_name] = out_path
        print(f"[Manifest Writer] Saved {len(records)} records -> {out_path}")

    # Summary statistics
    summary = {
        "seed": seed,
        "total_pairs": len(pairs),
        "total_images": len(pairs) * 2,
        "unique_persons_count": len(unique_persons),
        "splits": {}
    }
    
    for split_name, person_list in split_assignment.items():
        records = split_records[split_name]
        c_count = sum(1 for r in records if r['label'] == "0")
        d_count = sum(1 for r in records if r['label'] == "1")
        pairs_cnt = len(records) // 2
        raw_pids = sorted(list(set(r['participant_id'] for r in records)))
        
        finger_dist = defaultdict(int)
        for r in records:
            finger_dist[f"{r['hand']}_{r['finger']}"] += 1
            
        summary["splits"][split_name] = {
            "participants_count": len(person_list),
            "participants": sorted(person_list),
            "raw_participant_ids_count": len(raw_pids),
            "raw_participant_ids": raw_pids,
            "pairs_count": pairs_cnt,
            "total_images": len(records),
            "clean_images": c_count,
            "distorted_images": d_count,
            "image_pct": (len(records) / (len(pairs) * 2)) * 100,
            "finger_distribution": dict(finger_dist)
        }
        
    return summary


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING PARTICIPANT-LEVEL DATASET SPLIT (SEED=42)")
    print("=" * 70)
    summary = create_participant_splits(seed=42)
    
    for s_name, s_data in summary["splits"].items():
        print(f"\n[{s_name.upper()} SPLIT]")
        print(f"  Participants:    {s_data['participants_count']} unique persons ({s_data['raw_participant_ids_count']} raw IDs)")
        print(f"  Persons:         {s_data['participants']}")
        print(f"  Pairs:           {s_data['pairs_count']} pairs")
        print(f"  Total Images:    {s_data['total_images']} images ({s_data['image_pct']:.1f}% of total)")
        print(f"  Class Balance:   Clean={s_data['clean_images']} (50.0%), Distorted={s_data['distorted_images']} (50.0%)")
        print(f"  Fingers covered: {len(s_data['finger_distribution'])} distinct finger positions")

