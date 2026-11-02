"""
Dataset Module for Real V2 Experiments
======================================
Supports configurable data augmentation strategies:
- 'none': purely deterministic preprocessing
- 'conservative': mild rotation, translation, brightness/contrast, sensor noise
- 'extended': slightly wider range while strictly preserving ridge topology
"""

import os
import sys
import csv
import random
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class RealV2Dataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        target_size: Tuple[int, int] = (224, 224),
        aug_mode: str = "none",  # 'none', 'conservative', 'extended'
        seed: Optional[int] = None
    ):
        self.manifest_path = manifest_path
        self.target_size = target_size
        self.aug_mode = aug_mode
        self.seed = seed

        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.samples = list(csv.DictReader(f))

        if len(self.samples) == 0:
            raise ValueError(f"Manifest is empty: {manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def _apply_augmentation(self, img: np.ndarray) -> np.ndarray:
        if self.aug_mode == "none":
            return img

        h, w = img.shape[:2]

        if self.aug_mode == "conservative":
            rot_range = (-8.0, 8.0)
            trans_range = (-4.0, 4.0)
            contrast_range = (0.92, 1.08)
            brightness_range = (-0.06, 0.06)
            noise_sigma = (0.005, 0.015)
        elif self.aug_mode == "extended":
            rot_range = (-10.0, 10.0)
            trans_range = (-6.0, 6.0)
            contrast_range = (0.90, 1.10)
            brightness_range = (-0.08, 0.08)
            noise_sigma = (0.005, 0.020)
        else:
            return img

        # 1. Rotation and translation
        if random.random() < 0.6:
            angle = random.uniform(*rot_range)
            tx = random.uniform(*trans_range)
            ty = random.uniform(*trans_range)
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # 2. Brightness & contrast
        if random.random() < 0.5:
            contrast = random.uniform(*contrast_range)
            brightness = random.uniform(*brightness_range)
            img = img * contrast + brightness
            img = np.clip(img, 0.0, 1.0)

        # 3. Sensor noise
        if random.random() < 0.3:
            std = random.uniform(*noise_sigma)
            noise = np.random.normal(0.0, std, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        return img.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        item = self.samples[idx]
        filepath = item['full_filepath']

        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read image at: {filepath}")

        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

        img_norm = img.astype(np.float32) / 255.0
        img_norm = self._apply_augmentation(img_norm)

        tensor = torch.from_numpy(img_norm).unsqueeze(0)
        label = float(item['label'])
        label_tensor = torch.tensor(label, dtype=torch.float32)

        meta = {
            "pair_id": item['pair_id'],
            "participant_id": item['participant_id'],
            "person_id": item['person_id'],
            "hand": item['hand'],
            "finger": item['finger'],
            "class_name": item['class_name'],
            "filename": item['filename'],
            "image_path": item['image_path']
        }

        return tensor, label_tensor, meta


def get_real_v2_dataloaders(
    train_manifest: str,
    val_manifest: str,
    batch_size: int = 16,
    aug_mode: str = "conservative"
) -> Tuple[DataLoader, DataLoader]:
    train_ds = RealV2Dataset(train_manifest, aug_mode=aug_mode)
    val_ds = RealV2Dataset(val_manifest, aug_mode="none")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader

