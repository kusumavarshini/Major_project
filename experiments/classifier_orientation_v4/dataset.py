"""
Fingerprint Variation Dataset & DataLoader Module (V4)
======================================================
Loads the balanced FingerSure V4 dataset (6,000 images, 1,500 identities).
- CLEAN (0): Normal clean + global rotation/tilt variations
- DISTORTED (1): Independent elastic structural deformations

Features:
- Single-channel 224x224 grayscale uint8 images normalized to [0.0, 1.0] float32
- Tensor output shape: (1, 224, 224)
- High-performance in-memory caching to eliminate disk I/O bottlenecks during CPU training
- Clean interface returning batch dictionaries compatible with FingerprintCNN
"""

import sys
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Resolve project root
PROJ_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))


class FingerprintV4Dataset(Dataset):
    """
    PyTorch Dataset for single-channel 224x224 fingerprint distortion classification.
    Uses in-memory image caching for accelerated epoch throughput.
    """

    def __init__(self, manifest: Union[pd.DataFrame, str, Path], cache_images: bool = True):
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.is_absolute():
                manifest_path = PROJ_ROOT / manifest_path
            self.df = pd.read_csv(manifest_path)
        else:
            self.df = manifest.copy().reset_index(drop=True)

        self.cache_images = cache_images
        self.image_cache: Dict[int, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Retrieve cached image array if available
        if self.cache_images and idx in self.image_cache:
            img_arr = self.image_cache[idx]
        else:
            filepath = Path(row["filepath"])
            if not filepath.is_absolute():
                filepath = PROJ_ROOT / filepath

            if not filepath.is_file():
                raise FileNotFoundError(f"Image file does not exist: {filepath}")

            img = cv2.imread(str(filepath), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to decode image: {filepath}")

            if img.shape != (224, 224):
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

            img_arr = img
            if self.cache_images:
                self.image_cache[idx] = img_arr

        # Normalize to [0.0, 1.0] float32 tensor of shape (1, 224, 224)
        tensor_img = torch.from_numpy(img_arr).float().div(255.0).unsqueeze(0)
        label_val = float(row["label"])

        return {
            "image": tensor_img,
            "label": torch.tensor(label_val, dtype=torch.float32),
            "identity_stem": str(row["identity_stem"]),
            "variation_type": str(row["variation_type"]),
            "filepath": str(row["filepath"]),
            "filename": str(row["filename"])
        }

    def preload_all(self):
        """Optionally preload all images into memory upfront."""
        for i in range(len(self)):
            _ = self.__getitem__(i)


def get_v4_dataloaders(
    config_or_path: Union[str, Path, Dict]
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, pd.DataFrame]]:
    """
    Constructs train, validation, and test DataLoaders for V4 experiment.
    """
    if isinstance(config_or_path, (str, Path)):
        cfg_path = Path(config_or_path)
        if not cfg_path.is_absolute():
            cfg_path = PROJ_ROOT / cfg_path
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = config_or_path

    ds_cfg = cfg["dataset"]
    batch_size = ds_cfg.get("batch_size", 32)
    num_workers = ds_cfg.get("num_workers", 0)

    train_path = PROJ_ROOT / ds_cfg["train_manifest"]
    val_path = PROJ_ROOT / ds_cfg["val_manifest"]
    test_path = PROJ_ROOT / ds_cfg["test_manifest"]

    train_ds = FingerprintV4Dataset(train_path, cache_images=True)
    val_ds = FingerprintV4Dataset(val_path, cache_images=True)
    test_ds = FingerprintV4Dataset(test_path, cache_images=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    split_dfs = {
        "train": train_ds.df,
        "val": val_ds.df,
        "test": test_ds.df
    }

    return train_loader, val_loader, test_loader, split_dfs
