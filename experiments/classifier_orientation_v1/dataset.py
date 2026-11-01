"""
Fingerprint Variation Dataset & Identity-Level Split Module
===========================================================
Provides PyTorch Dataset and DataLoader abstractions for binary fingerprint distortion classification.
Implements strict identity-level 70:15:15 partitioning to eliminate data leakage.
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# PATH RESOLUTION HELPER
# ==============================================================================

def get_project_root() -> Path:
    """Returns the root directory of the Fingerprint project."""
    return Path(__file__).resolve().parent.parent.parent

def resolve_project_path(p: Union[str, Path]) -> Path:
    """Resolves a path relative to the project root unless it is already an absolute path."""
    path_obj = Path(p)
    if path_obj.is_absolute():
        return path_obj
    proj_root = get_project_root()
    return (proj_root / path_obj).resolve()

# ==============================================================================
# SPLIT MANAGEMENT (DATA LEAKAGE PREVENTION)
# ==============================================================================

def create_or_load_splits(
    dataset_root: Union[str, Path],
    split_dir: Union[str, Path],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    force_rebuild: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Partitions the 1,500 unique fingerprint identities into 70:15:15 train/val/test splits.
    Guarantees that all 4 variations of an identity (1 clean + 3 distorted) reside in the same split.
    
    Returns:
        (train_df, val_df, test_df)
    """
    dataset_root = resolve_project_path(dataset_root)
    split_dir = resolve_project_path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train_manifest.csv"
    val_csv = split_dir / "val_manifest.csv"
    test_csv = split_dir / "test_manifest.csv"

    if not force_rebuild and train_csv.exists() and val_csv.exists() and test_csv.exists():
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        test_df = pd.read_csv(test_csv)
        _verify_split_leakage_and_counts(train_df, val_df, test_df)
        return train_df, val_df, test_df

    clean_dir = dataset_root / "clean"
    orient_dir = dataset_root / "orientation_variation"
    partial_dir = dataset_root / "partial_capture"
    uneven_dir = dataset_root / "uneven_contact"

    for d in [clean_dir, orient_dir, partial_dir, uneven_dir]:
        if not d.exists():
            raise FileNotFoundError(f"Required dataset directory not found: {d}")

    # Discover original clean images to establish identity stems
    clean_images = sorted(list(clean_dir.glob("*_clean.png")))
    if not clean_images:
        raise ValueError(f"No clean images found in {clean_dir}")

    # Extract unique identity stems (e.g. 'FM001_C1_clean.png' -> 'FM001_C1')
    stems = sorted([f.name.replace("_clean.png", "") for f in clean_images])
    total_identities = len(stems)

    if total_identities != 1500:
        print(f"[WARNING] Discovered {total_identities} identities (expected 1,500).")

    # Deterministic identity shuffle
    rng = random.Random(seed)
    shuffled_stems = list(stems)
    rng.shuffle(shuffled_stems)

    n_train = int(round(total_identities * train_ratio))
    n_val = int(round(total_identities * val_ratio))
    n_test = total_identities - (n_train + n_val)

    train_stems = set(shuffled_stems[:n_train])
    val_stems = set(shuffled_stems[n_train:n_train + n_val])
    test_stems = set(shuffled_stems[n_train + n_val:])

    # Strict leakage assertion
    assert len(train_stems & val_stems) == 0, "DATA LEAKAGE: Overlap between train and val stems!"
    assert len(train_stems & test_stems) == 0, "DATA LEAKAGE: Overlap between train and test stems!"
    assert len(val_stems & test_stems) == 0, "DATA LEAKAGE: Overlap between val and test stems!"
    assert (len(train_stems) + len(val_stems) + len(test_stems)) == total_identities

    def build_records(stem_set: set) -> List[Dict]:
        records = []
        for s in sorted(list(stem_set)):
            # 1. Clean impression (Class 0: CLEAN)
            p_clean = clean_dir / f"{s}_clean.png"
            if not p_clean.exists():
                raise FileNotFoundError(f"Missing clean file: {p_clean}")
            records.append({
                "identity_stem": s,
                "filepath": str(p_clean),
                "filename": p_clean.name,
                "class_name": "clean",
                "label": 0,
                "variation_type": "clean"
            })

            # 2. Orientation variation (Class 1: DISTORTED)
            p_orient = orient_dir / f"{s}_orientation.png"
            if not p_orient.exists():
                raise FileNotFoundError(f"Missing orientation file: {p_orient}")
            records.append({
                "identity_stem": s,
                "filepath": str(p_orient),
                "filename": p_orient.name,
                "class_name": "distorted",
                "label": 1,
                "variation_type": "orientation_variation"
            })

            # 3. Partial capture (Class 1: DISTORTED)
            p_part = partial_dir / f"{s}_partial.png"
            if not p_part.exists():
                raise FileNotFoundError(f"Missing partial file: {p_part}")
            records.append({
                "identity_stem": s,
                "filepath": str(p_part),
                "filename": p_part.name,
                "class_name": "distorted",
                "label": 1,
                "variation_type": "partial_capture"
            })

            # 4. Uneven contact (Class 1: DISTORTED)
            p_unev = uneven_dir / f"{s}_uneven.png"
            if not p_unev.exists():
                raise FileNotFoundError(f"Missing uneven contact file: {p_unev}")
            records.append({
                "identity_stem": s,
                "filepath": str(p_unev),
                "filename": p_unev.name,
                "class_name": "distorted",
                "label": 1,
                "variation_type": "uneven_contact"
            })
        return records

    train_df = pd.DataFrame(build_records(train_stems))
    val_df = pd.DataFrame(build_records(val_stems))
    test_df = pd.DataFrame(build_records(test_stems))

    # Save CSV manifests
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    _verify_split_leakage_and_counts(train_df, val_df, test_df)
    return train_df, val_df, test_df


def _verify_split_leakage_and_counts(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    """Rigorous audit function verifying zero identity leakage and expected class ratios."""
    train_stems = set(train_df["identity_stem"])
    val_stems = set(val_df["identity_stem"])
    test_stems = set(test_df["identity_stem"])

    assert len(train_stems & val_stems) == 0, "LEAKAGE: Stem overlap train/val!"
    assert len(train_stems & test_stems) == 0, "LEAKAGE: Stem overlap train/test!"
    assert len(val_stems & test_stems) == 0, "LEAKAGE: Stem overlap val/test!"

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        clean_count = (df["label"] == 0).sum()
        dist_count = (df["label"] == 1).sum()
        total = len(df)
        identities = len(df["identity_stem"].unique())
        assert total == identities * 4, f"{name}: Each identity must have exactly 4 images (found {total} vs {identities * 4})"
        assert clean_count == identities, f"{name}: Clean count must equal identity count"
        assert dist_count == identities * 3, f"{name}: Distorted count must equal 3x identity count"


# ==============================================================================
# PYTORCH DATASET
# ==============================================================================

class FingerprintVariationDataset(Dataset):
    """
    PyTorch Dataset for single-channel 224x224 fingerprint distortion classification.
    Adheres strictly to the existing FingerprintCNN preprocessing convention:
    - Grayscale uint8 input
    - Resized to (224, 224) via INTER_AREA if necessary (already 224x224 in dataset)
    - Normalized to [0.0, 1.0] float32
    - Output tensor shape: (1, 224, 224)
    """

    def __init__(self, manifest: Union[pd.DataFrame, str, Path], target_size: Tuple[int, int] = (224, 224)):
        if isinstance(manifest, (str, Path)):
            manifest_path = resolve_project_path(manifest)
            self.df = pd.read_csv(manifest_path)
        else:
            self.df = manifest.copy().reset_index(drop=True)
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        img_path = resolve_project_path(row["filepath"])

        # 1. Read grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {img_path}")

        # 2. Resize if necessary (no-op if already 224x224)
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        # 3. Normalize to [0.0, 1.0] float32 (identical to production inference)
        img_norm = img.astype(np.float32) / 255.0

        # 4. Convert to tensor: shape (1, H, W)
        tensor = torch.from_numpy(img_norm).unsqueeze(0)

        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return {
            "image": tensor,
            "label": label,
            "identity_stem": row["identity_stem"],
            "variation_type": row["variation_type"],
            "filename": row["filename"]
        }


# ==============================================================================
# DATALOADER FACTORY
# ==============================================================================

def get_dataloaders(
    config: Union[Dict, str, Path],
    batch_size: Optional[int] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, pd.DataFrame]]:
    """
    Creates train, validation, and test PyTorch DataLoaders configured from config.json.
    """
    if isinstance(config, (str, Path)):
        cfg_path = resolve_project_path(config)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = config

    ds_cfg = cfg["dataset"]
    train_cfg = cfg.get("training", {})
    bs = batch_size or train_cfg.get("batch_size", 32)
    workers = num_workers if num_workers is not None else train_cfg.get("num_workers", 0)

    dataset_root = resolve_project_path(ds_cfg["dataset_root"])
    split_dir = resolve_project_path(ds_cfg.get("split_manifest_dir", "experiments/classifier_orientation_v1/splits"))
    splits = ds_cfg.get("splits", {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15})
    seed = ds_cfg.get("seed", 42)

    train_df, val_df, test_df = create_or_load_splits(
        dataset_root=dataset_root,
        split_dir=split_dir,
        train_ratio=splits.get("train_ratio", 0.70),
        val_ratio=splits.get("val_ratio", 0.15),
        test_ratio=splits.get("test_ratio", 0.15),
        seed=seed
    )

    train_ds = FingerprintVariationDataset(train_df)
    val_ds = FingerprintVariationDataset(val_df)
    test_ds = FingerprintVariationDataset(test_df)

    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available()
    )

    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": test_df
    }

    return train_loader, val_loader, test_loader, split_dfs
