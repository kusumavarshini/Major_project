import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

# Reconfigure stdout for utf-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 80)
print("FINGERSURE V3: SYNTHETIC-TO-PHYSICAL DOMAIN ADAPTATION AUDIT")
print("Target Directory: experiments/classifier_orientation_v3/")
print("=" * 80)

v3_dir = PROJ_ROOT / "experiments" / "classifier_orientation_v3"
v3_dir.mkdir(parents=True, exist_ok=True)

# Dataset paths
synth_root = PROJ_ROOT / "data" / "orientation_capture_variations_v1"
phys_root = PROJ_ROOT / "FingerSure_Dataset"

synth_clean_dir = synth_root / "clean"
synth_orient_dir = synth_root / "orientation_variation"
synth_part_dir = synth_root / "partial_capture"
synth_uneven_dir = synth_root / "uneven_contact"

phys_clean_dir = phys_root / "clean"
phys_dist_dir = phys_root / "distorted"

# 1. Discover files
synth_clean_files = sorted(list(synth_clean_dir.glob("*.png")))
synth_orient_files = sorted(list(synth_orient_dir.glob("*.png")))
synth_part_files = sorted(list(synth_part_dir.glob("*.png")))
synth_uneven_files = sorted(list(synth_uneven_dir.glob("*.png")))

phys_clean_files = sorted(list(phys_clean_dir.glob("*.png")))
phys_dist_files = sorted(list(phys_dist_dir.glob("*.png")))

print(f"Synthetic Counts: Clean={len(synth_clean_files)}, Orient={len(synth_orient_files)}, Part={len(synth_part_files)}, Uneven={len(synth_uneven_files)}")
print(f"Physical Counts:  Clean={len(phys_clean_files)}, Distorted={len(phys_dist_files)}")

# Build analysis queue
# For synthetic: evaluate all 1,500 clean and sample 500 of each distorted component (or evaluate all if fast)
# Evaluating 1,500 clean + 500*3 = 3,000 synthetic + 84 physical is very fast in OpenCV.
analysis_items = []

for p in synth_clean_files:
    analysis_items.append({"path": p, "domain": "Synthetic", "category": "Synthetic Clean", "class_label": 0, "variation_type": "clean"})

# Sample 500 per synthetic distorted variation to keep statistics balanced and fast
for p in synth_orient_files[:500]:
    analysis_items.append({"path": p, "domain": "Synthetic", "category": "Synthetic Orientation", "class_label": 1, "variation_type": "orientation_variation"})
for p in synth_part_files[:500]:
    analysis_items.append({"path": p, "domain": "Synthetic", "category": "Synthetic Partial", "class_label": 1, "variation_type": "partial_capture"})
for p in synth_uneven_files[:500]:
    analysis_items.append({"path": p, "domain": "Synthetic", "category": "Synthetic Uneven", "class_label": 1, "variation_type": "uneven_contact"})

for p in phys_clean_files:
    analysis_items.append({"path": p, "domain": "Physical (SecuGen)", "category": "Physical Clean", "class_label": 0, "variation_type": "Normal Relaxed"})

# Categorize physical distorted prints based on known physical attributes
for p in phys_dist_files:
    # We will compute the optical properties and classify
    analysis_items.append({"path": p, "domain": "Physical (SecuGen)", "category": "Physical Distorted", "class_label": 1, "variation_type": "Physical Distorted"})

print(f"\n[ANALYSIS] Extracting quantitative metrics on {len(analysis_items)} images...")

metrics_list = []
t0 = time.time()

for item in analysis_items:
    p = item["path"]
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    
    h, w = img.shape[:2]
    aspect_ratio = float(w / h)
    
    # Pixel intensity distribution
    flat = img.ravel()
    mean_val = float(np.mean(flat))
    std_val = float(np.std(flat))
    min_val = int(np.min(flat))
    max_val = int(np.max(flat))
    p5 = float(np.percentile(flat, 5))
    p25 = float(np.percentile(flat, 25))
    p50 = float(np.percentile(flat, 50))
    p75 = float(np.percentile(flat, 75))
    p95 = float(np.percentile(flat, 95))
    
    # Sharpness via Laplacian variance
    lap_var = float(cv2.Laplacian(img, cv2.CV_64F).var())
    
    # Foreground vs Background separation (Background is bright optical glass/white ~ > 220)
    bg_mask = (img >= 220)
    fg_mask = (img < 220)
    
    fg_coverage_pct = float(np.sum(fg_mask) / (h * w) * 100.0)
    
    fg_pixels = img[fg_mask]
    bg_pixels = img[bg_mask]
    
    if len(fg_pixels) > 0:
        ridge_mean = float(np.mean(fg_pixels))
        ridge_p5 = float(np.percentile(fg_pixels, 5))
        ridge_std = float(np.std(fg_pixels))
    else:
        ridge_mean = mean_val
        ridge_p5 = p5
        ridge_std = std_val
        
    bg_mean = float(np.mean(bg_pixels)) if len(bg_pixels) > 0 else 255.0
    
    # Contrast metrics
    michelson_contrast = float((bg_mean - ridge_mean) / (bg_mean + ridge_mean + 1e-6))
    weber_contrast = float((bg_mean - ridge_mean) / (bg_mean + 1e-6))
    dynamic_range = max_val - min_val
    
    # Edge density via Canny
    edges = cv2.Canny(img, 50, 150)
    edge_density = float(np.sum(edges > 0) / (h * w) * 100.0)
    
    # Local contrast: standard deviation of 16x16 blocks
    blk_size = 16
    h_blocks = h // blk_size
    w_blocks = w // blk_size
    local_stds = []
    for bi in range(h_blocks):
        for bj in range(w_blocks):
            block = img[bi*blk_size:(bi+1)*blk_size, bj*blk_size:(bj+1)*blk_size]
            # only compute on blocks containing foreground
            if np.any(block < 220):
                local_stds.append(np.std(block))
    local_contrast = float(np.mean(local_stds)) if len(local_stds) > 0 else 0.0

    # Specific subcategory for physical distorted
    var_sub = item["variation_type"]
    if item["category"] == "Physical Distorted":
        if fg_coverage_pct < 28.0:
            var_sub = "Physical Light / Partial"
        elif mean_val > 190.0 or std_val < 55.0:
            var_sub = "Physical Uneven Contact"
        elif ridge_p5 < 15.0 and fg_coverage_pct > 40.0:
            var_sub = "Physical Rotated / Heavy Shear"
        else:
            var_sub = "Physical Tilted Contact"

    metrics_list.append({
        "filename": p.name,
        "domain": item["domain"],
        "category": item["category"],
        "variation_type": var_sub,
        "class_label": item["class_label"],
        "width": w,
        "height": h,
        "aspect_ratio": round(aspect_ratio, 3),
        "mean_intensity": round(mean_val, 2),
        "std_intensity": round(std_val, 2),
        "min_intensity": min_val,
        "max_intensity": max_val,
        "p5_intensity": round(p5, 2),
        "p25_intensity": round(p25, 2),
        "median_intensity": round(p50, 2),
        "p75_intensity": round(p75, 2),
        "p95_intensity": round(p95, 2),
        "ridge_mean_intensity": round(ridge_mean, 2),
        "ridge_p5_darkness": round(ridge_p5, 2),
        "ridge_std": round(ridge_std, 2),
        "bg_mean_intensity": round(bg_mean, 2),
        "michelson_contrast": round(michelson_contrast, 4),
        "weber_contrast": round(weber_contrast, 4),
        "dynamic_range": dynamic_range,
        "sharpness_laplacian": round(lap_var, 2),
        "edge_density_pct": round(edge_density, 2),
        "local_contrast": round(local_contrast, 2),
        "fg_coverage_pct": round(fg_coverage_pct, 2)
    })

df_metrics = pd.DataFrame(metrics_list)
csv_out = v3_dir / "image_statistics.csv"
df_metrics.to_csv(csv_out, index=False)
print(f"[METRICS] Computed metrics on {len(df_metrics)} images in {time.time()-t0:.2f}s.")
print(f"[OUTPUT] Saved image_statistics.csv to {csv_out}")

# ==============================================================================
# 2. GROUP-WISE SUMMARY STATISTICS
# ==============================================================================
summary_cols = [
    "mean_intensity", "std_intensity", "p5_intensity", "median_intensity", "p95_intensity",
    "ridge_mean_intensity", "ridge_p5_darkness", "bg_mean_intensity", "michelson_contrast",
    "dynamic_range", "sharpness_laplacian", "edge_density_pct", "local_contrast", "fg_coverage_pct"
]

cat_summary = df_metrics.groupby("category")[summary_cols].mean().round(2)
print("\n" + "=" * 80)
print("CATEGORY-LEVEL MEAN OPTICAL & STRUCTURAL PROPERTIES:")
print("=" * 80)
print(cat_summary.T.to_string())

# Specific clean vs distorted comparison by domain
domain_class_summary = df_metrics.groupby(["domain", "class_label"])[summary_cols].mean().round(2)
print("\n" + "=" * 80)
print("DOMAIN & CLASS SUMMARY (0=CLEAN, 1=DISTORTED):")
print("=" * 80)
print(domain_class_summary.T.to_string())

# ==============================================================================
# 3. VISUAL CONTACT SHEETS GENERATION
# ==============================================================================
print("\n[CONTACT SHEETS] Generating visual domain comparison contact sheets...")

# Select representative images:
# 1. Synthetic Clean (5 images)
# 2. Synthetic Distorted (2 orient, 2 partial, 1 uneven)
# 3. Physical Clean (5 images)
# 4. Physical Distorted - Rotated / Sideways (5 images)
# 5. Physical Distorted - Light / Uneven Contact (5 images)

sample_synth_clean = df_metrics[df_metrics["category"] == "Synthetic Clean"].head(5)["filename"].tolist()
sample_synth_dist = (
    df_metrics[df_metrics["category"] == "Synthetic Orientation"].head(2)["filename"].tolist() +
    df_metrics[df_metrics["category"] == "Synthetic Partial"].head(2)["filename"].tolist() +
    df_metrics[df_metrics["category"] == "Synthetic Uneven"].head(1)["filename"].tolist()
)

sample_phys_clean = [
    "A_LEFT_THUMB_clean.png",
    "Kusuma_LEFT_THUMB_clean.png",
    "Neha_RIGHT_THUMB_clean.png",
    "Sanjana_LEFT_THUMB_clean.png",
    "shuban_RIGHT_THUMB_clean.png"
]

sample_phys_rotated = [
    "Kusuma_LEFT_THUMB_distorted.png",
    "sushmitha_RIGHT_THUMB_distorted.png",
    "Dhrithi_RIGHT_THUMB_distorted.png",
    "Sanjana_RIGHT_THUMB_distorted.png",
    "harshitha_LEFT_THUMB_distorted.png"
]

sample_phys_light_uneven = [
    "manal_RIGHT_THUMB_distorted.png",
    "Mangalamma_RIGHT_THUMB_distorted.png",
    "manya_RIGHT_THUMB_distorted.png",
    "Lakshmi_LEFT_THUMB_distorted.png",
    "Bhushan_LEFT_THUMB_distorted.png"
]

rows = [
    ("Synthetic CLEAN (In-Distribution Reference)", synth_clean_dir, sample_synth_clean),
    ("Synthetic DISTORTED (Orientation, Partial, Uneven)", synth_root, sample_synth_dist),
    ("Physical CLEAN (SecuGen Hamster Pro 20)", phys_clean_dir, sample_phys_clean),
    ("Physical DISTORTED - Rotated / Sideways", phys_dist_dir, sample_phys_rotated),
    ("Physical DISTORTED - Light / Uneven Contact", phys_dist_dir, sample_phys_light_uneven)
]

fig, axes = plt.subplots(5, 5, figsize=(15, 16))

for r_idx, (row_label, base_dir, filenames) in enumerate(rows):
    for c_idx, fn in enumerate(filenames):
        ax = axes[r_idx, c_idx]
        
        # Resolve path
        if "Synthetic DISTORTED" in row_label:
            if "_orientation.png" in fn:
                p = synth_orient_dir / fn
            elif "_partial.png" in fn:
                p = synth_part_dir / fn
            else:
                p = synth_uneven_dir / fn
        else:
            p = base_dir / fn
            
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            # Resize for uniform contact sheet display (224x224)
            img_disp = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
            ax.imshow(img_disp, cmap="gray", vmin=0, vmax=255)
            
            # Subtitle with key metrics
            mean_i = img.mean()
            lap_i = cv2.Laplacian(img, cv2.CV_64F).var()
            ax.set_title(f"{fn[:16]}...\nMean={mean_i:.0f} | Lap={lap_i:.0f}", fontsize=8)
        
        ax.axis("off")
        
    # Label the row on the first column
    axes[r_idx, 0].text(-0.25, 0.5, row_label, transform=axes[r_idx, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=11, fontweight="bold")

plt.suptitle("FingerSure Domain Audit: Synthetic vs. Real Physical SecuGen Captures\n(Visualizing Contrast, Dynamic Range, Sharpness & Contact Morphologies)", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
contact_sheet_path = v3_dir / "domain_comparison_contact_sheet.png"
plt.savefig(contact_sheet_path, dpi=300)
plt.close()
print(f"[PLOT] Saved domain_comparison_contact_sheet.png to {contact_sheet_path}")

# ==============================================================================
# 4. DISTRIBUTION GRAPHS (HISTOGRAMS, BOXPLOTS, SCATTERS)
# ==============================================================================
print("[PLOTS] Generating quantitative feature distribution plots...")

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

synth_mask = (df_metrics["domain"] == "Synthetic")
phys_mask = (df_metrics["domain"] == "Physical (SecuGen)")

# Subplot 1: Overall Mean Intensity Histogram
axes[0, 0].hist(df_metrics[synth_mask]["mean_intensity"], bins=30, alpha=0.6, color="#1f77b4", label="Synthetic (N=3,000)", density=True)
axes[0, 0].hist(df_metrics[phys_mask]["mean_intensity"], bins=20, alpha=0.7, color="#d62728", label="Physical SecuGen (N=84)", density=True)
axes[0, 0].set_title("Mean Image Intensity Distribution", fontweight="bold")
axes[0, 0].set_xlabel("Mean Grayscale Value (0-255)")
axes[0, 0].legend()
axes[0, 0].grid(True, linestyle=":", alpha=0.6)

# Subplot 2: Ridge Darkness (5th percentile)
axes[0, 1].hist(df_metrics[synth_mask]["ridge_p5_darkness"], bins=30, alpha=0.6, color="#1f77b4", label="Synthetic (Ridge p5)", density=True)
axes[0, 1].hist(df_metrics[phys_mask]["ridge_p5_darkness"], bins=20, alpha=0.7, color="#d62728", label="Physical (Ridge p5)", density=True)
axes[0, 1].set_title("Ridge Darkness (5th Percentile Intensity)", fontweight="bold")
axes[0, 1].set_xlabel("Ridge Darkness (Lower = Darker Ridges)")
axes[0, 1].legend()
axes[0, 1].grid(True, linestyle=":", alpha=0.6)

# Subplot 3: Laplacian Sharpness (Log scale)
axes[0, 2].boxplot([
    df_metrics[synth_mask]["sharpness_laplacian"],
    df_metrics[phys_mask]["sharpness_laplacian"]
], tick_labels=["Synthetic", "Physical SecuGen"], patch_artist=True)
axes[0, 2].set_title("Image Sharpness (Laplacian Variance)", fontweight="bold")
axes[0, 2].set_yscale("log")
axes[0, 2].set_ylabel("Laplacian Variance (Log Scale)")
axes[0, 2].grid(True, linestyle=":", alpha=0.6)

# Subplot 4: Michelson Contrast
axes[1, 0].hist(df_metrics[synth_mask]["michelson_contrast"], bins=30, alpha=0.6, color="#1f77b4", label="Synthetic", density=True)
axes[1, 0].hist(df_metrics[phys_mask]["michelson_contrast"], bins=20, alpha=0.7, color="#d62728", label="Physical SecuGen", density=True)
axes[1, 0].set_title("Foreground/Background Michelson Contrast", fontweight="bold")
axes[1, 0].set_xlabel("Michelson Contrast Ratio")
axes[1, 0].legend()
axes[1, 0].grid(True, linestyle=":", alpha=0.6)

# Subplot 5: Foreground Coverage Area
axes[1, 1].boxplot([
    df_metrics[synth_mask]["fg_coverage_pct"],
    df_metrics[phys_mask]["fg_coverage_pct"]
], tick_labels=["Synthetic", "Physical SecuGen"], patch_artist=True)
axes[1, 1].set_title("Foreground Fingerprint Area Coverage (%)", fontweight="bold")
axes[1, 1].set_ylabel("Coverage (%)")
axes[1, 1].grid(True, linestyle=":", alpha=0.6)

# Subplot 6: Sharpness vs Ridge Darkness Scatter
axes[1, 2].scatter(df_metrics[synth_mask]["ridge_p5_darkness"], df_metrics[synth_mask]["sharpness_laplacian"],
                   color="#1f77b4", alpha=0.3, s=15, label="Synthetic")
axes[1, 2].scatter(df_metrics[phys_mask]["ridge_p5_darkness"], df_metrics[phys_mask]["sharpness_laplacian"],
                   color="#d62728", alpha=0.8, s=35, edgecolors="black", label="Physical SecuGen")
axes[1, 2].set_title("Sharpness vs. Ridge Darkness Domain Shift", fontweight="bold")
axes[1, 2].set_xlabel("Ridge Darkness (5th Percentile)")
axes[1, 2].set_ylabel("Laplacian Variance (Log Scale)")
axes[1, 2].set_yscale("log")
axes[1, 2].legend()
axes[1, 2].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
dist_plot_path = v3_dir / "intensity_and_sharpness_distributions.png"
plt.savefig(dist_plot_path, dpi=300)
plt.close()
print(f"[PLOT] Saved intensity_and_sharpness_distributions.png to {dist_plot_path}")

print("\n" + "=" * 80)
print("DOMAIN AUDIT DATA EXTRACTION COMPLETED SUCCESSFULLY!")
print("=" * 80)

