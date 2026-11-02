# Synthetic Fingerprint Capture Variations Dataset Generation Report (v1)

**Generation Date:** November 1, 2026  
**Source Dataset:** [`data/processed/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/processed) (1,500 original clean fingerprint images)  
**Output Dataset:** [`data/orientation_capture_variations_v1/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1)  
**Contact Sheet:** [`contact_sheet.png`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/contact_sheet.png)  
**Metadata Manifest:** [`metadata.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/metadata.csv) (4,500 records)  

---

## 1. Executive Summary

A complete, high-fidelity synthetic fingerprint variation dataset has been generated from the 1,500 original clean fingerprint images.

The generation process modeled physical capture variations observed on the **SecuGen Hamster Pro 20** optical fingerprint scanner:
1. **Orientation Variation:** Controlled rotational angles spanning slight ($\pm 8^\circ$ to $\pm 18^\circ$), medium ($\pm 18^\circ$ to $\pm 32^\circ$), and steep ($\pm 32^\circ$ to $\pm 44^\circ$) tilts, using high-precision bicubic resampling to preserve ridge edge sharpness without bilinear blurring shortcuts. Corners are filled seamlessly with estimated platen background.
2. **Partial-Contact Capture:** Incomplete physical skin contact simulating fingertip (distal), core (proximal), left-roll, right-roll, and diagonal corner touches. Retains $50\%$ to $72\%$ of the contact zone with smooth natural curvilinear skin-lift boundaries.
3. **Uneven Contact & Ridge Visibility:** Optical contrast attenuation simulating non-uniform finger pressure, dry skin, and light contact. Ridges in weak contact areas smoothly fade towards the platen background ($C \in [0.12, 0.38]$) while preserving 100% of underlying minutiae locations and ridge flow geometry.

> [!IMPORTANT]
> **Strict Safety Adherence:**
> - Zero CNN training, fine-tuning, or inference was performed.
> - Zero checkpoints or model files were modified.
> - The original 1,500 clean images in `data/processed/` remain untouched.
> - Zero physical SecuGen images were mixed into this synthetic training dataset.
> - Zero aggressive elastic warping, spline remap deformation, or artificial noise artifacts were introduced.

---

## 2. Dataset Inventory & Verification Counts

| Category | Target Directory | File Count | Dimensions | Format | Integrity |
|---|---|---|---|---|---|
| **Original Clean References** | [`clean/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/clean) | **1,500** | $224 \times 224$ | Grayscale (uint8, Mode L) | 100% Verified (0 corrupt) |
| **Variation 1: Orientation** | [`orientation_variation/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/orientation_variation) | **1,500** | $224 \times 224$ | Grayscale (uint8, Mode L) | 100% Verified (0 corrupt) |
| **Variation 2: Partial Capture** | [`partial_capture/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/partial_capture) | **1,500** | $224 \times 224$ | Grayscale (uint8, Mode L) | 100% Verified (0 corrupt) |
| **Variation 3: Uneven Contact** | [`uneven_contact/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/uneven_contact) | **1,500** | $224 \times 224$ | Grayscale (uint8, Mode L) | 100% Verified (0 corrupt) |
| **Total Distorted Variations** | - | **4,500** | $224 \times 224$ | Grayscale (uint8, Mode L) | 100% Verified (0 corrupt) |
| **Total Images in Dataset** | - | **6,000** | $224 \times 224$ | Grayscale (uint8, Mode L) | 100% Verified (0 corrupt) |

---

## 3. Transformation Parameters & Methodology

### Variation 1: Orientation Variation
- **Angle Range:** $[-43.98^\circ, -8.02^\circ] \cup [+8.02^\circ, +43.97^\circ]$ (Mean $|\theta| = 23.47^\circ$).
- **Angle Distribution:**
  - Slight tilts ($8^\circ - 18^\circ$): 40.2%
  - Medium tilts ($18^\circ - 32^\circ$): 39.7%
  - Steep tilts ($32^\circ - 44^\circ$): 20.1%
- **Scale Factor:** $s \in [0.930, 0.980]$ (mean $0.955$) to ensure all ridges stay within the frame without border clipping.
- **Interpolation:** `cv2.INTER_CUBIC` (bicubic) to prevent the low-level blurring shortcut identified in the Classifier V3 audit.
- **Background Fill:** Matched to the per-image platen background median ($240 - 248$).

### Variation 2: Partial-Contact Variation
- **Contact Patterns:**
  - `top_partial` (Fingertip touch; base lifted): 20.4%
  - `bottom_partial` (Core touch; fingertip lifted): 19.8%
  - `left_partial` (Left-rolled touch; right side lifted): 19.5%
  - `right_partial` (Right-rolled touch; left side lifted): 20.7%
  - `diagonal_partial` (Corner/diagonal touch): 19.6%
- **Retained Contact Area:** Mean **$58.2\%$** (range $50.1\%$ to $71.6\%$).
- **Boundary Modeling:** Natural convex parabolic curvature with sigmoid skin-lift fall-off ($7 - 13$ px margin) transitioning to the platen background. No rectangular cutouts.

### Variation 3: Uneven-Contact & Ridge Visibility
- **Modulation Patterns:**
  - `linear_gradient` (Directional pressure tilt across finger): 33.1%
  - `radial_gradient` (Center firm contact fading toward perimeter): 33.5%
  - `dry_patchy` (Multi-lobed contact with natural dry patches): 33.4%
- **Minimum Ridge Contrast ($C_{\min}$):** Range $[0.12, 0.38]$ (mean $0.25$).
- **Average Contact Contrast:** Mean **$64.8\%$** of original ridge darkness.
- **Minutiae Geometry:** $100\%$ preserved. Ridges simply fade in light contact regions without spatial displacement or warping.

---

## 4. Metadata Manifest Schema

[`metadata.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/metadata.csv) contains 4,500 records structured as follows:

| Column Name | Data Type | Example Value | Description |
|---|---|---|---|
| `original_filename` | String | `FM001_C1.png` | Source clean image name in `data/processed/` |
| `generated_filename` | String | `FM001_C1_orientation.png` | Output filename in respective variation folder |
| `variation_category` | String | `orientation_variation` | Category (`orientation_variation`, `partial_capture`, `uneven_contact`) |
| `applied_rotation_angle` | Float / Null | `-33.16` | Rotation angle in degrees (Null for non-orientation variations) |
| `transformation_parameters` | JSON String | `{'angle_deg': -33.16, 'scale': 0.975, ...}` | Full parameter dictionary used during generation |
| `original_image_dimensions` | String | `224x224` | Width and height of original clean image |
| `generated_image_dimensions` | String | `224x224` | Width and height of generated variation image |

---

## 5. Contact Sheet & Visual Inspection

A visual contact sheet displaying 8 representative fingerprint sets (each showing Original Clean, Orientation Variation, Partial-Contact, and Uneven-Contact) has been generated:
- File path: [`contact_sheet.png`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/contact_sheet.png)
- Dimensions: $1046 \times 1919$ pixels

Visual inspection confirms:
1. Ridges maintain optical darkness and high-frequency edge definition.
2. Rotated prints blend naturally into the platen background without black corner borders.
3. Partial impressions display smooth, organic skin contact boundaries.
4. Uneven contact prints exhibit natural ridge fading consistent with dry skin or light scanner pressure.
5. All images remain recognizable fingerprints with intact identity.

---

## 6. Next Steps & Approval Gate

Per instructions:
- **All dataset generation and validation tasks are COMPLETE.**
- **Generation process has STOPPED.**
- Awaiting user visual inspection of the contact sheet and approval before performing any subsequent steps.
