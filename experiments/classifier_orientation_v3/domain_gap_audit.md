# FingerSure – Synthetic-to-Physical Domain Adaptation Audit (Classifier V3 Preparation)

**Audit Date:** 2026-11-01  
**Target Experiment:** `experiments/classifier_orientation_v3/`  
**Dataset Under Audit:**
- **Synthetic Training Data (`data/orientation_capture_variations_v1/`):** 6,000 total images (1,500 identities $\times$ 4 impressions: clean, orientation, partial, uneven).
- **Physical SecuGen Data ([`FingerSure_Dataset/`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/)):** 84 total images (21 participants $\times$ 4 impressions: 42 clean, 42 natural distortion).
**Evaluation Artifacts Directory:** [`experiments/classifier_orientation_v3/`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v3/)

---

## 1. Executive Summary & Audit Motivation

In the previous physical evaluation of `classifier_orientation_v2`, the model achieved:
- **Synthetic Test Accuracy:** **99.89%** (ROC-AUC **1.0000**)
- **Physical Out-of-Fold Accuracy:** **65.48%** (ROC-AUC **0.7517**)
- **Fixed Threshold 0.50 Distortion Recall:** **0.00%** (all 42 physical distorted prints fell below $0.50$, peaking at $0.4576$)

While V2 corrected the inverted-feature failure of V3-Colab (which had an AUC of 0.3515), it exhibited a **domain logit calibration offset**: physical fingerprint images systematically produced lower classification probabilities than synthetic images.

To prepare the next independent CNN experiment, `classifier_orientation_v3`, this audit conducted quantitative, optical, and structural measurements across **3,084 fingerprint images** (3,000 synthetic + 84 real physical SecuGen captures) to identify the exact physical differences and formulate a domain-adapted augmentation and training strategy.

---

## 2. Quantitative Image Domain Analysis

Every image was analyzed across 14 optical and structural dimensions. Measurements were recorded in [`image_statistics.csv`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v3/image_statistics.csv).

### Comprehensive Domain Comparison Table

| Metric | Synthetic Clean (N=1,500) | Synthetic Distorted (N=1,500) | Physical Clean (N=42) | Physical Distorted (N=42) | Domain Ratio (Phys / Synth) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Image Resolution** | $224 \times 224$ | $224 \times 224$ | $300 \times 400$ | $300 \times 400$ | Native Aspect Ratio $0.75$ vs $1.00$ |
| **Mean Grayscale Intensity** | $208.69 \pm 10.3$ | $211.22 \pm 11.8$ | $147.92 \pm 23.4$ | $162.58 \pm 28.1$ | **$1.35\times$ darker** overall |
| **Standard Deviation Intensity** | $43.23 \pm 6.8$ | $44.72 \pm 7.9$ | $90.42 \pm 7.5$ | $88.98 \pm 11.2$ | **$2.05\times$ higher spread** |
| **5th Percentile Intensity ($P_5$)** | $127.49 \pm 15.6$ | $117.65 \pm 18.2$ | $16.88 \pm 8.4$ | $19.48 \pm 9.6$ | **$6.9\times$ darker low-end** |
| **Median Intensity ($P_{50}$)** | $231.57 \pm 12.1$ | $236.06 \pm 10.5$ | $165.50 \pm 41.2$ | $204.98 \pm 38.6$ | Bimodal optical distribution |
| **95th Percentile Intensity ($P_{95}$)**| $243.92 \pm 3.1$ | $242.18 \pm 3.8$ | $246.93 \pm 1.8$ | $247.12 \pm 1.6$ | Identical optical glass saturation |
| **Ridge Darkness ($P_5$ of Ridges)** | **$115.83 \pm 14.2$** | **$102.79 \pm 16.5$** | **$12.14 \pm 6.2$** | **$13.90 \pm 7.1$** | **$8.3\times$ darker physical ridges** |
| **Mean Ridge Intensity** | $152.60 \pm 11.8$ | $146.78 \pm 13.9$ | $84.43 \pm 14.1$ | $83.41 \pm 15.8$ | **$1.8\times$ darker mean ridges** |
| **Mean Background Intensity** | $238.35 \pm 3.2$ | $238.79 \pm 3.5$ | $238.81 \pm 2.1$ | $238.32 \pm 2.4$ | **100% Identical background** |
| **Michelson Contrast** | $0.23 \pm 0.04$ | $0.25 \pm 0.05$ | $0.49 \pm 0.07$ | $0.50 \pm 0.08$ | **$2.05\times$ higher contrast** |
| **Full Dynamic Range** | $163.73 \pm 20.4$ | $165.27 \pm 22.8$ | $252.74 \pm 4.2$ | $251.07 \pm 6.1$ | Full 8-bit dynamic range $[0, 255]$ |
| **Laplacian Sharpness Variance** | **$398.13 \pm 124$** | **$378.56 \pm 141$** | **$11,457.05 \pm 3,820$** | **$10,478.60 \pm 3,910$** | **$28.2\times$ higher sharpness!** |
| **Edge Density (Canny %)** | $8.07 \pm 2.1\%$ | $8.51 \pm 2.4\%$ | $23.54 \pm 3.1\%$ | $20.06 \pm 4.2\%$ | **$2.6\times$ higher edge density** |
| **Local Block Contrast (16x16)** | $18.46 \pm 3.5$ | $21.42 \pm 4.2$ | $72.02 \pm 6.8$ | $69.20 \pm 7.9$ | **$3.5\times$ higher local contrast** |
| **Foreground Contact Area (%)** | $30.96 \pm 7.2\%$ | $29.43 \pm 8.6\%$ | $57.03 \pm 9.4\%$ | $47.69 \pm 12.8\%$ | **$1.8\times$ larger physical area** |

---

## 3. Detailed Component Breakdown (Synthetic vs. Physical Subgroups)

To prevent averaging artifacts, metrics were dissected across all individual variation subcategories:

| Category | Mean Intensity | Ridge $P_5$ Darkness | Sharpness (Laplacian) | Michelson Contrast | Local Contrast | FG Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Synthetic Clean** | 208.69 | 115.83 | 398.13 | 0.23 | 18.46 | 30.96% |
| **Synthetic Orientation** | 205.85 | 94.36 | 497.36 | 0.28 | 23.53 | 32.15% |
| **Synthetic Partial** | 216.51 | 97.79 | 316.60 | 0.26 | 22.80 | 23.18% |
| **Synthetic Uneven** | 211.29 | 116.23 | 321.71 | 0.21 | 17.95 | 32.96% |
| **Physical Clean (SecuGen)** | 147.92 | 12.14 | 11,457.05 | 0.49 | 72.02 | 57.03% |
| **Physical Distorted - Rotated / Heavy** | 158.42 | 11.85 | 11,215.40 | 0.51 | 71.84 | 51.20% |
| **Physical Distorted - Tilted / Shear** | 163.20 | 14.12 | 10,120.30 | 0.49 | 68.45 | 46.10% |
| **Physical Distorted - Light / Uneven** | 192.80 | 22.40 | 7,420.15 | 0.41 | 58.10 | 25.40% |

---

## 4. Visual Inspection & Physical Morphology Audit

Visual inspections were synthesized in [`domain_comparison_contact_sheet.png`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v3/domain_comparison_contact_sheet.png) and feature distributions in [`intensity_and_sharpness_distributions.png`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v3/intensity_and_sharpness_distributions.png).

### Visible Structural Differences Identified:

1. **Optical Total Internal Reflection (TIR) Mechanics:**
   - **Physical Images:** Captured via an illuminated glass prism where skin ridges create solid optical contact, completely frustrating total internal reflection. This results in pitch-black ridges ($I \approx 0 - 25$) surrounded by pristine white background glass ($I \approx 238 - 255$). The transition between ridge and valley is extraordinarily steep, producing Laplacian variances exceeding $10,000$.
   - **Synthetic Images:** Rendered from scanned FVC impressions where ridges are gray ($I \approx 100 - 150$) and the background has paper/sensor grain noise. Ridge-to-valley gradients are soft, yielding Laplacian variances around $350 - 500$.
2. **Contact Geometry and Sensor Size:**
   - **Physical Thumb Impressions:** Human thumbs pressed against the SecuGen platen occupy $50\% - 65\%$ of the $300 \times 400$ sensor area. When resized to $224 \times 224$ via `INTER_AREA`, the fingerprint is vertically compressed by $0.75\times$, causing ridges to pack more tightly.
   - **Synthetic Impressions:** Crop frames were centered with $30\%$ coverage, leaving wide white borders.
3. **Morphology of Physical Distortion vs Appearance Differences:**
   - **Rotated / Sideways Prints:** Display clear diagonal ridge flow ($> 20^\circ$) and asymmetric boundary oval cutoffs where the thumb rolled off the optical platen edges.
   - **Light / Partial Contact Prints:** In prints from participants with dry skin or light touch (e.g. `manal`, `Mangalamma`, `manya`, `Lakshmi`), only a small circle of central core touches the glass ($< 25\%$ coverage). Within that small circle, the ridges are pristine, sharp, and concentric.
   - **Distinction:** Light contact reduces foreground coverage but does NOT necessarily warp ridge flow. The CNN must learn to evaluate both geometric deformation and anomalous contact boundaries without mistaking contrast variations for distortion.

---

## 5. Domain Gap Impact on CNN Activations

Why did `classifier_orientation_v2` predict all physical distorted prints with probabilities below $0.46$?

1. **Feature Suppression in Early Convolutions:**  
   The 4-layer `FingerprintCNN` uses $3 \times 3$ convolutions in `Conv1` (32 filters) and `Conv2` (64 filters). These filters respond strongly to high-frequency edge gradients. Because physical SecuGen images possess $28\times$ higher sharpness and pitch-black ridges, the convolutional activations across normal ridges are enormous. The network associates these strong, clean, regular ridge signals with pristine, authentic fingerprints ("CLEAN"), pulling the scalar classification logit $z$ downward into negative values.
2. **Probability Compression:**  
   Because positive logits require spatial irregularity to overcome this "high ridge quality" bias, physical distorted prints produce logits between $-3.5$ and $-0.17$ ($P \in [0.03, 0.46]$). The ranking is preserved ($AUC = 0.7517$), but the threshold boundary is displaced from $0.50$ down to $0.035 - 0.040$.

---

## 6. Audit Conclusions & Core Directives for V3

1. **Retain the Binary Classification Definition:**  
   $\text{CLEAN} = 0$ (properly placed, centered, full-contact prints), $\text{DISTORTED} = 1$ (rotated, tilted, sideways, partial contact, uneven contact, and heavy shear).
2. **Domain Adaptation Must Be Symmetric:**  
   Photometric augmentations (ridge darkening, contrast boosting, sharpness enhancement) must be applied to **both CLEAN and DISTORTED synthetic training images**. This aligns the visual appearance of the synthetic domain with the SecuGen sensor without corrupting the label distinction.
3. **Preserve Structural Geometry:**  
   Augmentations must NOT destroy ridge continuity or alter ridge topology. Blurring, thresholding, or extreme gamma must be strictly bounded.

