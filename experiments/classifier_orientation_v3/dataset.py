"""
FingerprintCNN V3 Dataset & DataLoader Module
=============================================
Provides PyTorch Dataset and DataLoader classes with domain-adaptive augmentation
for the synthetic 6,000-image dataset (4,200 Train / 900 Val / 900 Test).
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from experiments.classifier_orientation_v3.augmentations import DomainAdaptiveAugmenter

def resolve_path(p: Union[str, Path]) -> Path:
    p_obj = Path(p)
    return p_obj if p_obj.is_absolute() else (PROJ_ROOT / p_obj).resolve()


class FingerprintV3Dataset(Dataset):
    """
    PyTorch Dataset for single-channel 224x224 fingerprint distortion classification.
    Applies online domain-adaptive augmentation during training, and clean deterministic
    preprocessing during validation and testing.
    """
    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        is_train: bool = False,
        augmenter: Optional[DomainAdaptiveAugmenter] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        if isinstance(manifest, (str, Path)):
            manifest_path = resolve_path(manifest)
            self.df = pd.read_csv(manifest_path)
        else:
            self.df = manifest.copy().reset_index(drop=True)

        self.is_train = is_train
        self.augmenter = augmenter if (is_train and augmenter is not None) else None
        self.target_size = target_size

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        img_p = resolve_path(row["filepath"])
        label = int(row["label"])

        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at {img_p}")

        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        # Apply domain-adaptive augmentation if in training mode
        if self.is_train and self.augmenter is not None:
            img, _ = self.augmenter(img, label)

        # Normalization to [0.0, 1.0] float32
        img_norm = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_norm).unsqueeze(0)  # Shape: (1, H, W)
        target = torch.tensor(float(label), dtype=torch.float32)

        return {
            "image": tensor,
            "label": target,
            "identity_stem": row["identity_stem"],
            "variation_type": row.get("variation_type", "unknown"),
            "filename": row["filename"]
        }


def get_v3_dataloaders(
    config: Union[Dict, str, Path],
    batch_size: Optional[int] = None,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, pd.DataFrame]]:
    """
    Creates train, validation, and test PyTorch DataLoaders configured from config.json.
    """
    if isinstance(config, (str, Path)):
        cfg_path = resolve_path(config)
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = config

    ds_cfg = cfg["dataset"]
    train_cfg = cfg.get("training", {})
    bs = batch_size or train_cfg.get("batch_size", 32)
    workers = num_workers if num_workers is not None else train_cfg.get("num_workers", 0)

    split_dir = resolve_path(ds_cfg.get("split_manifest_dir", "experiments/classifier_orientation_v3/splits"))

    train_df = pd.read_csv(split_dir / "train_manifest.csv")
    val_df = pd.read_csv(split_dir / "val_manifest.csv")
    test_df = pd.read_csv(split_dir / "test_manifest.csv")

    seed = ds_cfg.get("seed", 42)
    augmenter = DomainAdaptiveAugmenter(p_photometric=0.60, p_geometric=0.40, p_structural=0.45, seed=seed)

    train_ds = FingerprintV3Dataset(train_df, is_train=True, augmenter=augmenter)
    val_ds = FingerprintV3Dataset(val_df, is_train=False, augmenter=None)
    test_ds = FingerprintV3Dataset(test_df, is_train=False, augmenter=None)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader, {"train": train_df, "val": val_df, "test": test_df}

