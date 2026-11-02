import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from matcher import MATCHER

c_path = 'data/test_distorted/clean/FAMILY-13/CHILD/FM013_C2.png'
d_path = 'data/test_distorted/distorted/FAMILY-13/CHILD/FM013_C2_distorted.png'
r_path = 'data/rectified_ddrnet_test/FAMILY-13__CHILD__FM013_C2__bending.png'

c = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)
d = cv2.imread(d_path, cv2.IMREAD_GRAYSCALE)
r = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)

print(f"Clean shape: {c.shape}, min={c.min()}, max={c.max()}, mean={c.mean():.2f}, std={c.std():.2f}")
print(f"Distorted shape: {d.shape}, min={d.min()}, max={d.max()}, mean={d.mean():.2f}, std={d.std():.2f}")
print(f"Rectified shape: {r.shape}, min={r.min()}, max={r.max()}, mean={r.mean():.2f}, std={r.std():.2f}")

if r.shape != c.shape:
    r_resized = cv2.resize(r, (c.shape[1], c.shape[0]))
else:
    r_resized = r

ssim_dist = ssim(c, d, data_range=255)
psnr_dist = psnr(c, d, data_range=255)
ssim_rect = ssim(c, r_resized, data_range=255)
psnr_rect = psnr(c, r_resized, data_range=255)

print(f"SSIM (Clean vs Distorted): {ssim_dist:.4f}, PSNR: {psnr_dist:.2f} dB")
print(f"SSIM (Clean vs Rectified): {ssim_rect:.4f}, PSNR: {psnr_rect:.2f} dB")

tpl_c = MATCHER.extract_template(c)
tpl_d = MATCHER.extract_template(d)
tpl_r = MATCHER.extract_template(r)
print(f"Minutiae - Clean: {len(tpl_c.get('minutiae', []))}, Distorted: {len(tpl_d.get('minutiae', []))}, Rectified: {len(tpl_r.get('minutiae', []))}")
