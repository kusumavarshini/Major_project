"""
Domain-Adaptive Augmentation Module for FingerprintCNN (Experiment V3)
======================================================================
Implements:
1. Symmetric Photometric Augmentations (Optical TIR Darkening, Unsharp Masking, Sensor Noise, Gamma, Local Contrast).
2. Geometric Augmentations (Aspect-Ratio Vertical Squeeze, Rotation/Orientation) applied to BOTH classes.
3. Structural Distortion Augmentations (Local Elastic Shear, Uneven Contact, Partial Contact) applied ONLY to DISTORTED.
"""

import math
import random
from typing import Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch

# ==============================================================================
# 1. SYMMETRIC PHOTOMETRIC AUGMENTATIONS (APPLIED TO BOTH CLASSES)
# ==============================================================================

def apply_tir_darkening(img: np.ndarray, gamma: Optional[float] = None, target_min: Optional[float] = None) -> np.ndarray:
    """
    Simulates optical Total Internal Reflection (TIR) on SecuGen glass.
    Non-linearly darkens ridge cores (intensities < 220) down towards 10-30 while preserving white background.
    """
    if gamma is None:
        gamma = random.uniform(1.3, 2.0)
    if target_min is None:
        target_min = random.uniform(10.0, 30.0)

    out = img.astype(np.float32)
    bg_thresh = 220.0
    ridge_mask = out < bg_thresh

    if np.any(ridge_mask):
        ridge_vals = out[ridge_mask]
        r_min = float(ridge_vals.min())
        r_max = float(ridge_vals.max())
        if r_max > r_min:
            # Normalize ridges to [0, 1]
            norm_r = (ridge_vals - r_min) / (r_max - r_min)
            # Power law stretching
            darkened_norm = np.power(norm_r, gamma)
            # Scale to [target_min, bg_thresh]
            new_r = target_min + darkened_norm * (bg_thresh - target_min)
            out[ridge_mask] = new_r

    return np.clip(out, 0, 255).astype(np.uint8)


def apply_unsharp_mask(img: np.ndarray, amount: Optional[float] = None, sigma: Optional[float] = None) -> np.ndarray:
    """
    Modulates Laplacian sharpness variance to bridge the 28x sharpness gap.
    Uses unsharp masking to enhance high-frequency ridge boundaries.
    """
    if amount is None:
        amount = random.uniform(0.6, 1.6)
    if sigma is None:
        sigma = random.uniform(0.8, 1.4)

    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_sensor_noise(img: np.ndarray, noise_std: Optional[float] = None) -> np.ndarray:
    """
    Injects mild optical CMOS sensor shot noise in low-luminance ridge troughs.
    """
    if noise_std is None:
        noise_std = random.uniform(1.0, 3.5)

    noise = np.random.normal(0, noise_std, img.shape).astype(np.float32)
    # Mask noise predominantly to non-background areas
    fg_weight = (img < 230).astype(np.float32)
    out = img.astype(np.float32) + noise * fg_weight
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_gamma_brightness(img: np.ndarray, gamma: Optional[float] = None, bias: Optional[float] = None) -> np.ndarray:
    """
    Applies bounded global illumination and gamma adjustments.
    """
    if gamma is None:
        gamma = random.uniform(0.85, 1.20)
    if bias is None:
        bias = random.uniform(-10.0, 10.0)

    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 + bias for i in range(256)])
    table = np.clip(table, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


def apply_local_contrast(img: np.ndarray, clip_limit: Optional[float] = None) -> np.ndarray:
    """
    Enhances local block contrast variance to simulate physical platen optical contrast.
    """
    if clip_limit is None:
        clip_limit = random.uniform(1.2, 2.5)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(img)


# ==============================================================================
# 2. GEOMETRIC AUGMENTATIONS (APPLIED TO BOTH CLASSES SYMMETRICALLY)
# ==============================================================================

def apply_anisotropic_squeeze(img: np.ndarray, sy: Optional[float] = None) -> np.ndarray:
    """
    Simulates the physical vertical aspect-ratio squeeze created when downsampling 300x400 to 224x224.
    """
    if sy is None:
        sy = random.uniform(0.75, 1.05)

    h, w = img.shape[:2]
    new_h = max(32, int(round(h * sy)))
    resized = cv2.resize(img, (w, new_h), interpolation=cv2.INTER_AREA)

    if new_h < h:
        # Pad top/bottom with white background (240)
        pad_top = (h - new_h) // 2
        pad_bot = h - new_h - pad_top
        out = cv2.copyMakeBorder(resized, pad_top, pad_bot, 0, 0, cv2.BORDER_CONSTANT, value=240)
    else:
        # Center crop back to h
        start_y = (new_h - h) // 2
        out = resized[start_y:start_y + h, :]

    return cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)


def apply_rotation_orientation(img: np.ndarray, angle_deg: Optional[float] = None) -> np.ndarray:
    """
    Applies in-plane rotation to simulate finger orientation.
    Applied symmetrically to both classes across realistic capture ranges.
    """
    if angle_deg is None:
        angle_deg = random.uniform(-15.0, 15.0)

    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=240)
    return out


# ==============================================================================
# 3. STRUCTURAL DISTORTION AUGMENTATIONS (APPLIED ONLY TO DISTORTED SAMPLES)
# ==============================================================================

def apply_local_elastic_displacement(img: np.ndarray, alpha: Optional[float] = None, sigma: Optional[float] = None) -> np.ndarray:
    """
    Simulates localized skin shear elasticity using smooth Gaussian displacement fields.
    Strictly bounded (alpha <= 8, sigma >= 12) to ensure ridge continuity is preserved without self-intersections.
    """
    if alpha is None:
        alpha = random.uniform(3.0, 7.5)
    if sigma is None:
        sigma = random.uniform(12.0, 18.0)

    h, w = img.shape[:2]
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)

    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=240)


def apply_uneven_contact(img: np.ndarray) -> np.ndarray:
    """
    Simulates non-uniform finger contact pressure where one sector of the fingerprint fades out towards the platen edge.
    """
    h, w = img.shape[:2]
    # Random center off-center
    cx = w * random.uniform(0.3, 0.7)
    cy = h * random.uniform(0.3, 0.7)
    sig_x = w * random.uniform(0.35, 0.55)
    sig_y = h * random.uniform(0.35, 0.55)

    y, x = np.ogrid[:h, :w]
    mask = np.exp(-(((x - cx) ** 2) / (2 * sig_x ** 2) + ((y - cy) ** 2) / (2 * sig_y ** 2)))
    mask = np.clip(mask, 0.0, 1.0)

    out = img.astype(np.float32) * mask + 240.0 * (1.0 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_partial_contact(img: np.ndarray) -> np.ndarray:
    """
    Simulates partial fingerprint contact (where only a portion of the finger touches the platen).
    Uses a smooth platen edge boundary without rectangular black boxes.
    """
    h, w = img.shape[:2]
    # Linear cut gradient along random angle
    theta = random.uniform(0, 2 * math.pi)
    nx, ny = math.cos(theta), math.sin(theta)
    offset = random.uniform(0.15, 0.35) * min(h, w)

    y, x = np.ogrid[:h, :w]
    dist = (x - w/2) * nx + (y - h/2) * ny
    # Smooth sigmoid transition over 15 pixels
    mask = 1.0 / (1.0 + np.exp((dist - offset) / 6.0))

    out = img.astype(np.float32) * mask + 240.0 * (1.0 - mask)
    return np.clip(out, 0, 255).astype(np.uint8)


# ==============================================================================
# 4. COMPOSITE DOMAIN-ADAPTIVE AUGMENTER
# ==============================================================================

class DomainAdaptiveAugmenter:
    """
    Composite augmentation pipeline for FingerprintCNN V3.
    Applies symmetric photometric and geometric transformations to both classes,
    and applies realistic structural deformation transformations exclusively to DISTORTED samples.
    """
    def __init__(
        self,
        p_photometric: float = 0.60,
        p_geometric: float = 0.40,
        p_structural: float = 0.45,
        seed: Optional[int] = 42
    ):
        self.p_photometric = p_photometric
        self.p_geometric = p_geometric
        self.p_structural = p_structural
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def __call__(self, img: np.ndarray, label: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Args:
            img: 224x224 uint8 single-channel grayscale image.
            label: 0 for CLEAN, 1 for DISTORTED.
        Returns:
            (augmented_img, applied_transforms_dict)
        """
        out = img.copy()
        log = {"applied": []}

        # -------------------------------------------------------------
        # 1. Symmetric Photometric Domain Adaptation (Clean AND Distorted)
        # -------------------------------------------------------------
        if random.random() < self.p_photometric:
            # Step A: TIR Ridge Darkening
            if random.random() < 0.70:
                gamma = random.uniform(1.3, 1.9)
                target_darkness = random.uniform(12.0, 28.0)
                out = apply_tir_darkening(out, gamma=gamma, target_min=target_darkness)
                log["applied"].append(f"TIR_Darkening(g={gamma:.2f},d={target_darkness:.1f})")

            # Step B: Sharpness / Edge Gradient Boosting
            if random.random() < 0.60:
                amt = random.uniform(0.6, 1.4)
                out = apply_unsharp_mask(out, amount=amt)
                log["applied"].append(f"Sharpness(amt={amt:.2f})")

            # Step C: Sensor Noise
            if random.random() < 0.35:
                noise_s = random.uniform(1.0, 3.0)
                out = apply_sensor_noise(out, noise_std=noise_s)
                log["applied"].append(f"SensorNoise(s={noise_s:.1f})")

            # Step D: Gamma / Illumination
            if random.random() < 0.35:
                g = random.uniform(0.9, 1.15)
                out = apply_gamma_brightness(out, gamma=g)
                log["applied"].append(f"Gamma({g:.2f})")

        # -------------------------------------------------------------
        # 2. Geometric Non-Uniform Squeeze & Orientation (BOTH classes)
        # -------------------------------------------------------------
        if random.random() < self.p_geometric:
            # Aspect-Ratio Squeeze (simulates 300x400 -> 224x224 INTER_AREA scaling)
            if random.random() < 0.50:
                sy = random.uniform(0.78, 0.98)
                out = apply_anisotropic_squeeze(out, sy=sy)
                log["applied"].append(f"AspectSqueeze(sy={sy:.2f})")

            # Natural Orientation / Tilt (symmetrically applied to both classes)
            if random.random() < 0.50:
                ang = random.uniform(-12.0, 12.0)
                out = apply_rotation_orientation(out, angle_deg=ang)
                log["applied"].append(f"OrientationTilt(deg={ang:.1f})")

        # -------------------------------------------------------------
        # 3. Structural Deformation (DISTORTED SAMPLES ONLY)
        # -------------------------------------------------------------
        if label == 1 and random.random() < self.p_structural:
            dist_choice = random.choice(["elastic", "uneven", "partial"])
            if dist_choice == "elastic":
                alpha = random.uniform(3.5, 6.5)
                out = apply_local_elastic_displacement(out, alpha=alpha, sigma=15.0)
                log["applied"].append(f"ElasticDisplacement(a={alpha:.1f})")
            elif dist_choice == "uneven":
                out = apply_uneven_contact(out)
                log["applied"].append("UnevenContactFalloff")
            elif dist_choice == "partial":
                out = apply_partial_contact(out)
                log["applied"].append("PartialContactBoundary")

        return out, log

