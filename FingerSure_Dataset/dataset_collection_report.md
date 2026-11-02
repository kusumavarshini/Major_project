# FingerSure – Physical Fingerprint Dataset Collection Validation Report

**Report Generated:** 2026-11-02T17:48:04  
**Dataset Directory:** [`C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\FingerSure_Dataset`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset)  
**Target Milestone:** 42 Participants $\times$ 20 Captures (10 Clean, 10 Distorted) = **840 Total Images**  

---

## 1. Executive Summary & Progress Toward Target (42 Participants)

| Metric | Target Benchmark | Actual Recorded Count | Progress (%) | Audit Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Participants** | 42 | **26** | 61.9% | ONGOING |
| **Fully Complete Participants (20/20)** | 42 | **5** | 11.9% | PENDING FULL COMPLETION |
| **Incomplete Participants** | 0 | **21** | — | 21 participants require remaining fingers |
| **Total CLEAN Images** | 420 | **110** | 26.2% | Disk verified |
| **Total DISTORTED Images** | 420 | **110** | 26.2% | Disk verified |
| **Total Fingerprint Images** | **840** | **220** | **26.19%** | Physical PNG files |
| **Corrupted / Invalid Images** | 0 | **0** | — | VERIFIED CLEAN (0 errors) |
| **Duplicate Records** | 0 | **0** | — | VERIFIED NO DUPLICATES |
| **Logged Failed Captures** | — | **64** | — | Logged separately in `failed_captures.csv` |

---

## 2. MySQL Database Synchronization Status

- **Database Name:** `fingersure_eval_db`
- **Collection Table:** `physical_dataset_collection`
- **Total Images in MySQL:** **220**
- **Clean Images in MySQL:** **110**
- **Distorted Images in MySQL:** **110**
- **Participants in MySQL:** **26**
- **Failed Captures in MySQL:** **64**

---

## 3. Per-Finger Capture Distribution (All 10 Fingers)

| Hand | Finger | Finger Code | Clean Count | Distorted Count | Total Images | Completeness |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **RIGHT** | THUMB | `RIGHT_THUMB` | 26 | 26 | **52** | Active |
| **RIGHT** | INDEX | `RIGHT_INDEX` | 9 | 9 | **18** | Active |
| **RIGHT** | MIDDLE | `RIGHT_MIDDLE` | 8 | 8 | **16** | Active |
| **RIGHT** | RING | `RIGHT_RING` | 8 | 8 | **16** | Active |
| **RIGHT** | LITTLE | `RIGHT_LITTLE` | 6 | 6 | **12** | Active |
| **LEFT** | THUMB | `LEFT_THUMB` | 25 | 25 | **50** | Active |
| **LEFT** | INDEX | `LEFT_INDEX` | 7 | 7 | **14** | Active |
| **LEFT** | MIDDLE | `LEFT_MIDDLE` | 7 | 7 | **14** | Active |
| **LEFT** | RING | `LEFT_RING` | 9 | 9 | **18** | Active |
| **LEFT** | LITTLE | `LEFT_LITTLE` | 5 | 5 | **10** | Active |

---

## 4. Per-Participant Capture Breakdown

| Participant ID | Clean Captures (/10) | Distorted Captures (/10) | Total Completed (/20) | Remaining (/20) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `A` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `B` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `BHARTHI` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `BHUSHAN` | 10/10 | 10/10 | **20/20** | 0 | **COMPLETE (20/20)** |
| `DHRITHI` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `DISHA` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `GANESH` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `HARSHITHA` | 10/10 | 10/10 | **20/20** | 0 | **COMPLETE (20/20)** |
| `HRUDHYA` | 1/10 | 1/10 | **2/20** | 18 | Pending (18 remaining) |
| `INCHARA` | 10/10 | 10/10 | **20/20** | 0 | **COMPLETE (20/20)** |
| `ISSA` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `JYOTHIKA` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `JYOTHSNIKA` | 4/10 | 4/10 | **8/20** | 12 | Pending (12 remaining) |
| `KUSUMA` | 8/10 | 8/10 | **16/20** | 4 | Pending (4 remaining) |
| `LAKSHMI` | 10/10 | 10/10 | **20/20** | 0 | **COMPLETE (20/20)** |
| `MANAL` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `MANGALAMMA` | 9/10 | 9/10 | **18/20** | 2 | Pending (2 remaining) |
| `MANYA` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `MEGHANA` | 6/10 | 6/10 | **12/20** | 8 | Pending (8 remaining) |
| `NEHA` | 10/10 | 10/10 | **20/20** | 0 | **COMPLETE (20/20)** |
| `PRIYANKA` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `RIGZIN` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `SANJANA` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `SHUBAN` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `SUSHMITHA` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |
| `THRYAEE` | 2/10 | 2/10 | **4/20** | 16 | Pending (16 remaining) |

---

## 5. Data Integrity, Corrupted Images & Duplicate Audit

### A. Corrupted or Unreadable Images
- **Zero corrupted images detected.** All physical PNG files decoded successfully with valid pixel gradients and non-zero dimensions.

### B. Duplicate Record Audit
- **Zero duplicate entries detected.** All captures follow unique constraint `(participant_id, hand, finger, condition)`.

### C. Failed Captures Log
- Total capture failures/rejections recorded: **64**.
- Log file: [`FingerSure_Dataset/failed_captures.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/failed_captures.csv).
