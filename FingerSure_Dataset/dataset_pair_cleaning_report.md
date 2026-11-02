# FingerSure – Physical Dataset Pair Filtering & Filename Normalization Report

**Report Generated:** 2026-11-02T17:47:54.023380  
**Dataset Directory:** [`C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\FingerSure_Dataset`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset)  
**JSON Report:** [`FingerSure_Dataset/dataset_pair_cleaning_report.json`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/dataset_pair_cleaning_report.json)  
**Valid Pairs Manifest:** [`FingerSure_Dataset/valid_pairs_manifest.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/valid_pairs_manifest.csv)  

---

## 1. Executive Summary & Required Metrics

| No. | Cleaning Metric Description | Count / Value | Verification Status |
| :---: | :--- | :---: | :--- |
| 1 | **Total participants before cleaning** | **34** | Raw distinct participant IDs |
| 2 | **Total CLEAN images before cleaning** | **118** | Physical files on disk |
| 3 | **Total DISTORTED images before cleaning** | **111** | Physical files on disk |
| 4 | **Total image pairs before cleaning** | **110** | Complete matched pairs |
| 5 | **Number of complete pairs retained** | **110** | 100% complete & valid |
| 6 | **Number of unpaired CLEAN images quarantined** | **8** | Moved to quarantine |
| 7 | **Number of unpaired DISTORTED images quarantined** | **1** | Moved to quarantine |
| 8 | **Number of corrupted images** | **0** | All images decode with OpenCV |
| 9 | **Number of duplicate records excluded** | **2** | Superseded Kusuma recaptures |
| 10 | **Total CLEAN images after cleaning** | **110** | Exactly matches valid pairs |
| 11 | **Total DISTORTED images after cleaning** | **110** | Exactly matches valid pairs |
| 12 | **Total images after cleaning** | **220** | Exactly $2 \times$ valid pairs |
| 13 | **Number of filenames converted to lowercase** | **220** | Converted via safe 2-step rename |
| 14 | **Filename collisions or unresolved metadata references** | **0** | Zero collisions detected |

---

## 2. Mathematical Relationship Verification

$$\text{Total CLEAN images (110)} = \text{Total DISTORTED images (110)} = \text{Total valid pairs (110)}$$

$$\text{Total active images (220)} = 2 \times \text{Total valid pairs (110)}$$

- **Parity:** Both equations hold with 100% precision.
- **CSV Synchronization:** `FingerSure_Dataset/metadata.csv` contains exactly **220** active rows.
- **MySQL Synchronization:** `physical_dataset_collection` contains exactly **220** active records.

---

## 3. Quarantined Unpaired Images Detail

The following 9 incomplete/unpaired images were safely relocated to the quarantine directory structure:

| File Name | Category | Hand & Finger | Quarantine Target Location | Exclusion Reason |
| :--- | :---: | :--- | :--- | :--- |
| `HRUDHYA_RIGHT_INDEX_CLEAN.png` | CLEAN | RIGHT INDEX | `quarantine/unpaired/clean/` | `MISSING_DISTORTED_PAIR` |
| `HRUDHYA_RIGHT_MIDDLE_CLEAN.png` | CLEAN | RIGHT MIDDLE | `quarantine/unpaired/clean/` | `MISSING_DISTORTED_PAIR` |
| `HRUDHYA_RIGHT_RING_CLEAN.png` | CLEAN | RIGHT RING | `quarantine/unpaired/clean/` | `MISSING_DISTORTED_PAIR` |
| `JYOTHSNIKA_LEFT_INDEX_CLEAN.png` | CLEAN | LEFT INDEX | `quarantine/unpaired/clean/` | `MISSING_DISTORTED_PAIR` |
| `JYOTHSNIKA_LEFT_MIDDLE_CLEAN.png` | CLEAN | LEFT MIDDLE | `quarantine/unpaired/clean/` | `MISSING_DISTORTED_PAIR` |
| `JYOTHSNIKA_RIGHT_MIDDLE_CLEAN.png` | CLEAN | RIGHT MIDDLE | `quarantine/unpaired/clean/` | `MISSING_DISTORTED_PAIR` |
| `MANGALAMMA_LEFT_LITTLE_CLEAN.png` | CLEAN | LEFT LITTLE | `quarantine/unpaired/clean/` | `MISSING_DISTORTED_PAIR` |
| `MEGHANA_LEFT_INDEX_CLEAN.png` | CLEAN | LEFT INDEX | `quarantine/unpaired/clean/` | `MISSING_DISTORTED_PAIR` |
| `JYOTHSNIKA_RIGHT_RING_DISTORTED.png` | DISTORTED | RIGHT RING | `quarantine/unpaired/distorted/` | `MISSING_CLEAN_PAIR` |

> [!NOTE]
> **Preservation Guarantee:** None of these images were deleted. Their original metadata and reason for exclusion are recorded in [`FingerSure_Dataset/quarantine/quarantine_metadata.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/quarantine/quarantine_metadata.csv).

---

## 4. Retained Complete Pairs Breakdown (Per Participant)

| Participant ID | Retained Valid Pairs | Active Clean Images | Active Distorted Images | Total Images | Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `A` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `B` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `Bharthi` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `BHUSHAN` | 8 | 8 | 8 | 16 | Partial (8/10 pairs) |
| `Bhushan` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `Dhrithi` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `Disha` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `Ganesh` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `HARSHITHA` | 8 | 8 | 8 | 16 | Partial (8/10 pairs) |
| `harshitha` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `HRUDHYA` | 1 | 1 | 1 | 2 | Partial (1/10 pairs) |
| `INCHARA` | 10 | 10 | 10 | 20 | Full 10-Finger Complete (20/20) |
| `issa` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `jyothika` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `JYOTHSNIKA` | 4 | 4 | 4 | 8 | Partial (4/10 pairs) |
| `KUSUMA` | 6 | 6 | 6 | 12 | Partial (6/10 pairs) |
| `Kusuma` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `LAKSHMI` | 8 | 8 | 8 | 16 | Partial (8/10 pairs) |
| `Lakshmi` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `manal` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `MANGALAMMA` | 7 | 7 | 7 | 14 | Partial (7/10 pairs) |
| `Mangalamma` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `manya` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `MEGHANA` | 6 | 6 | 6 | 12 | Partial (6/10 pairs) |
| `NEHA` | 8 | 8 | 8 | 16 | Partial (8/10 pairs) |
| `Neha` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `Priyanka` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `RIGZIN` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `Sanjana` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `shuban` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `sushmitha` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| `thryaee` | 2 | 2 | 2 | 4 | Partial (2/10 pairs) |
| **TOTAL** | **110** | **110** | **110** | **220** | **100% Paired** |

---

## 5. Filename Normalization Audit & Windows Compatibility

- **Case Normalization:** All 220 active image files in `clean/` and `distorted/` now have strictly lowercase filenames.
- **Cross-Platform Safety:** On Windows NTFS (case-preserving, case-insensitive), a safe two-step rename through UUID-based temporary filenames was utilized to guarantee true casing changes on disk without risking file deletion or collisions.
- **Naming Schema Verification:** Every active file strictly follows `{participant_id_lower}_{hand_lower}_{finger_lower}_{condition_lower}.png`.
- **1-to-1 Correspondence:** Every `clean/<name>_clean.png` has a corresponding `distorted/<name>_distorted.png`.

---

## 6. Stale Production References Audit

- **Historical Gallery Manifest:** [`FingerSure_Dataset/gallery_manifest.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/gallery_manifest.csv) and MySQL `fingerprint_gallery` reference historical clean thumb templates recorded on 2026-10-30 with original uppercase filenames (e.g. `A_LEFT_THUMB_clean.png`).
- **Action Taken:** Per strict safety requirements, the production enrollment gallery and SourceAFIS templates were **NOT modified**.
