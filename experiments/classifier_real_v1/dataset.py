"""
Dataset & DataLoader Module for FingerSure Real Dataset CNN Experiment (classifier_real_v1)
==========================================================================================
- PyTorch Dataset loading from manifest CSVs.
- Preprocessing: Grayscale -> Resize (224x224) -> float32 -> Normalize to [0, 1].
- Conservative Training Augmentation (rotation, translation, brightness/contrast, noise).
- Deterministic Evaluation (val/test): strictly unaugmented preprocessing.
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


class PhysicalFingerprintDataset(Dataset):
    """
    Dataset for physical fingerprint distortion classification.
    """
    def __init__(
        self,
        manifest_path: str,
        target_size: Tuple[int, int] = (224, 224),
        augment: bool = False,
        seed: Optional[int] = None
    ):
        self.manifest_path = manifest_path
        self.target_size = target_size
        self.augment = augment
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
        """
        Conservative augmentations on training samples only:
        - Random small rotation: [-8, 8] deg
        - Random small translation: [-4, 4] pixels
        - Random mild brightness: [-0.08, +0.08]
        - Random mild contrast: [0.92, 1.08]
        - Mild additive Gaussian sensor noise: sigma <= 0.015
        """
        h, w = img.shape[:2]
        
        # 1. Random small rotation & translation
        if random.random() < 0.6:
            angle = random.uniform(-8.0, 8.0)
            tx = random.uniform(-4.0, 4.0)
            ty = random.uniform(-4.0, 4.0)
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            # Border reflect or replicate to avoid black artificial margins
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # 2. Random mild brightness & contrast
        if random.random() < 0.5:
            contrast = random.uniform(0.92, 1.08)
            brightness = random.uniform(-0.06, 0.06)
            img = img * contrast + brightness
            img = np.clip(img, 0.0, 1.0)

        # 3. Mild sensor Gaussian noise
        if random.random() < 0.3:
            noise_std = random.uniform(0.005, 0.015)
            noise = np.random.normal(0.0, noise_std, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        return img.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, str]]:
        item = self.samples[idx]
        filepath = item['full_filepath']
        
        # Read grayscale
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to read image at: {filepath}")
            
        # Resize to 224x224
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
        # Float32 normalized to [0, 1]
        img_norm = img.astype(np.float32) / 255.0
        
        # Apply augmentation if enabled
        if self.augment:
            img_norm = self._apply_augmentation(img_norm)
            
        # Channel dimension: [1, H, W]
        tensor = torch.from_numpy(img_norm).unsqueeze(0)
        
        # Label as float tensor for BCEWithLogitsLoss
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


def get_dataloaders(
    train_manifest: str,
    val_manifest: str,
    test_manifest: str,
    batch_size: int = 16,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = PhysicalFingerprintDataset(train_manifest, augment=True)
    val_ds = PhysicalFingerprintDataset(val_manifest, augment=False)
    test_ds = PhysicalFingerprintDataset(test_manifest, augment=False)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader

