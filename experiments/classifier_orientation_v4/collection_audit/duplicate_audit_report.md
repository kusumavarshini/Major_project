# FingerSure – Physical Dataset Collection Duplicate & Integrity Audit Report

**Audit Type:** AUDIT-ONLY forensic inspection  
**Execution Timestamp:** 2026-11-02T12:10:00+05:30  
**Dataset Directory:** [`FingerSure_Dataset/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset)  
**Database Name:** `fingersure_eval_db`  
**Database Table:** `physical_dataset_collection`  
**Report Artifact:** [`experiments/classifier_orientation_v4/collection_audit/duplicate_audit_report.md`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/collection_audit/duplicate_audit_report.md)  
**Machine-Readable JSON:** [`experiments/classifier_orientation_v4/collection_audit/duplicate_audit_report.json`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/collection_audit/duplicate_audit_report.json)  

---

## 1. Executive Summary

A comprehensive forensic audit of the physical dataset collection records was performed to investigate the duplicate warning flagged by `validate_dataset_collection.py`.

| Audit Metric | Disk (Physical PNGs) | CSV (`metadata.csv`) | MySQL (`physical_dataset_collection`) |
| :--- | :---: | :---: | :---: |
| **Total Image Records** | **84** | **88** | **86** |
| **Clean Fingerprints** | 42 | 45 (44 unique + 1 duplicate) | 44 (all unique) |
| **Distorted Fingerprints** | 42 | 43 (42 unique + 1 case-alias) | 42 (all unique) |
| **Unique Participants** | 21 | 23 (21 real + 1 case-alias + 1 ghost) | 22 (21 real + 1 ghost) |
| **Duplicate Pairs Identified**| **0** (0 binary dups) | **2 duplicate pairs** | **0** (enforced by UNIQUE constraint) |
| **Ghost Records (No Image)**| 0 | **2 records** (`ashitta`) | **2 records** (`ashitta`) |
| **Physical Images Accounted**| 84 / 84 (100%) | 84 / 84 (100% represented) | 84 / 84 (100% represented) |

---

## 2. Identification of the Exact Duplicate Records

The duplicate detection audit identified exactly **two duplicate record pairs** (representing 2 redundant rows) in [`FingerSure_Dataset/metadata.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/metadata.csv):

### Duplicate Pair 1: `Kusuma` — `RIGHT_THUMB` (CLEAN)
- **Participant ID:** `Kusuma`
- **Hand:** `RIGHT`
- **Finger:** `THUMB`
- **Condition:** `0` (`CLEAN`)
- **Filename:** `Kusuma_RIGHT_THUMB_clean.png`
- **Relative Path:** `clean/Kusuma_RIGHT_THUMB_clean.png`

| Occurrence | CSV Line | Capture Timestamp | Distortion Condition | Record Status |
| :--- | :---: | :---: | :---: | :--- |
| **Occurrence A** | Line 2 | `2026-10-30T11:47:50` | `Normal Relaxed` | **Obsolete Historical Entry** (file overwritten) |
| **Occurrence B** | Line 8 | `2026-10-30T12:11:24` | `Normal Relaxed` | **Active Entry** (matches physical file on disk) |

### Duplicate Pair 2: `Kusuma` — `LEFT_THUMB` (CLEAN)
- **Participant ID:** `Kusuma`
- **Hand:** `LEFT`
- **Finger:** `THUMB`
- **Condition:** `0` (`CLEAN`)
- **Filename:** `Kusuma_LEFT_THUMB_clean.png`
- **Relative Path:** `clean/Kusuma_LEFT_THUMB_clean.png`

| Occurrence | CSV Line | Capture Timestamp | Distortion Condition | Record Status |
| :--- | :---: | :---: | :---: | :--- |
| **Occurrence A** | Line 3 | `2026-10-30T11:54:09` | `Normal Relaxed` | **Obsolete Historical Entry** (file overwritten) |
| **Occurrence B** | Line 7 | `2026-10-30T12:10:28` | `Normal Relaxed` | **Active Entry** (matches physical file on disk) |

---

## 3. Storage Location Audit: CSV vs. MySQL

| Storage Medium | Contains Duplicates? | Technical Explanation |
| :--- | :---: | :--- |
| **`FingerSure_Dataset/metadata.csv`** | **YES** (2 duplicates) | The legacy collection script from October 30, 2026 used a simple append mode (`open(..., 'a')`). When the operator initiated a re-capture session 23 minutes later, the new capture entries were appended to the bottom of the CSV without checking for or replacing earlier rows. |
| **MySQL `physical_dataset_collection`** | **NO** (0 duplicates) | The relational database schema strictly defines: <br>`UNIQUE KEY uq_participant_finger_cond (participant_id, hand, finger, condition)`.<br>When the metadata synchronization process was executed using `INSERT ... ON DUPLICATE KEY UPDATE`, the second set of captures (Lines 7 and 8) updated the existing rows for `Kusuma` in-place rather than inserting duplicate records. As a result, MySQL contains exactly 4 records for participant `Kusuma` (Left Clean, Left Distorted, Right Clean, Right Distorted). |

---

## 4. Physical Image File Integrity & Disk Verification

We conducted a cryptographic SHA-256 and filesystem metadata audit of the physical images in [`clean/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/clean) and [`distorted/`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/distorted):

### Are image files duplicated on disk?
> [!IMPORTANT]
> **NO. The image files on disk are NOT duplicated.** Only the metadata rows in `metadata.csv` are duplicated.

- There is **exactly one** physical file on disk for `Kusuma_RIGHT_THUMB_clean.png`.
- There is **exactly one** physical file on disk for `Kusuma_LEFT_THUMB_clean.png`.
- SHA-256 hash analysis across all 42 clean images and 42 distorted images confirmed **zero binary duplicate image files** in the dataset.

### Physical File Characteristics for Kusuma Clean Captures:
1. `clean/Kusuma_RIGHT_THUMB_clean.png`:
   - File Size: `95,953 bytes`
   - Filesystem Last Modified (`mtime`): `2026-10-30T12:11:24.648124`
   - SHA-256: `f591761343ef80d294680879c4a5c54320986eb97eef9f270a6c0a006c9a9d28`
   - **Correlation:** The disk timestamp `12:11:24` matches CSV Line 8 (`2026-10-30T12:11:24`) to the exact second. The earlier capture from Line 2 (`11:47:50`) was overwritten in-place when Line 8 was saved.

2. `clean/Kusuma_LEFT_THUMB_clean.png`:
   - File Size: `92,281 bytes`
   - Filesystem Last Modified (`mtime`): `2026-10-30T12:10:28.374591`
   - SHA-256: `3d7a759d12e5024f9fa31f016d910351ceeb8ba5be5d36e76cf022934149fa81`
   - **Correlation:** The disk timestamp `12:10:28` matches CSV Line 7 (`2026-10-30T12:10:28`) to the exact second. The earlier capture from Line 3 (`11:54:09`) was overwritten in-place when Line 7 was saved.

---

## 5. Representation Audit of the 84 Physical Images

We performed a bi-directional verification between the 84 physical PNG files on disk and the metadata entries:

### A. Disk-to-Metadata Mapping:
- **Total Physical PNGs on Disk:** 84 (42 clean, 42 distorted)
- **Exact Case-Sensitive Matches:** 83 files
- **Case-Insensitive Match (Casing Discrepancy):** 1 file
  - Disk file: `distorted/Bharthi_LEFT_THUMB_distorted.png`
  - Metadata CSV Line 27: `filename = BHarthi_LEFT_THUMB_distorted.png` (`participant_id = BHarthi`)
  - Status: Windows NTFS resolves this case-insensitively, but on case-sensitive filesystems (Linux/Docker) this will present as a mismatch.
- **Physical Images Completely Missing from Metadata:** **0 (0.0%)**
- **Conclusion:** **100% of the 84 physical images on disk are represented in the metadata.**

---

## 6. Full Discrepancy Reconciliation: 88 vs. 86 vs. 84

The core question is why the counts differ: **88 rows in CSV**, **86 records in MySQL**, and **84 physical images on disk**.

The discrepancy is $+4$ records in CSV relative to disk images ($88 - 84 = +4$). The forensic breakdown explains this:

```
[84 Physical Images on Disk]
   ├── 42 Clean PNGs (21 participants × 2 thumbs)
   └── 42 Distorted PNGs (21 participants × 2 thumbs)
         │
         ├── +2 Kusuma Clean Re-capture Duplicates (Lines 2 & 3 vs 7 & 8)
         │     └── Overwrote physical disk files; duplicate rows remained in CSV.
         │
         └── +2 Ashitta Ghost / Orphan Records (Lines 36 & 37)
               └── Recorded in CSV/MySQL, but corresponding PNGs do not exist on disk.
         │
         ▼
[88 Rows in metadata.csv]
   └── In MySQL: The 2 Kusuma duplicates collapsed due to the UNIQUE KEY (88 - 2 = 86).
         │
         ▼
[86 Records in MySQL physical_dataset_collection]
```

### Mathematical Proof of Reconciliation:

$$\text{CSV Rows (88)} = \underbrace{84}_{\text{Real Disk Images}} + \underbrace{2}_{\text{Kusuma Duplicate Rows}} + \underbrace{2}_{\text{Ashitta Ghost Rows}}$$

$$\text{MySQL Records (86)} = \underbrace{84}_{\text{Real Disk Images}} + \underbrace{0}_{\text{Kusuma (deduplicated by UNIQUE key)}} + \underbrace{2}_{\text{Ashitta Ghost Rows}}$$

$$\text{Disk Images (84)} = 21 \text{ Verified Participants} \times 4 \text{ Captures} = 42 \text{ Clean} + 42 \text{ Distorted}$$

### Details on the 2 `ashitta` Ghost Records:
- **Line 36:** `ashitta, RIGHT, THUMB, condition=0, clean/ashitta_RIGHT_THUMB_clean.png, 2026-10-30T12:53:40`
- **Line 37:** `ashitta, LEFT, THUMB, condition=0, clean/ashitta_LEFT_THUMB_clean.png, 2026-10-30T12:55:47`
- Neither `clean/ashitta_RIGHT_THUMB_clean.png` nor `clean/ashitta_LEFT_THUMB_clean.png` exists anywhere on disk.
- Participant `ashitta` has **0 distorted captures** recorded.
- Participant `ashitta` is **completely absent from `gallery_manifest.csv`** (the 42 enrolled gallery templates only belong to the 21 real participants).
- This indicates `ashitta` was an aborted or failed capture session where metadata was appended before the scanner write completed or the files were subsequently removed.

---

## 7. Recommended Corrections (For Future Approved Maintenance)

Per the strict constraints of this task, **no records have been altered, removed, or overwritten**.

When approved, the following non-destructive sequence should be executed:

1. **Clean up Obsolete Kusuma Duplicates in CSV:**
   - Remove lines 2 and 3 from [`FingerSure_Dataset/metadata.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/metadata.csv) (preserving lines 7 and 8, which match the physical disk images and modification timestamps).
2. **Standardize Participant ID Casing for `Bharthi`:**
   - On line 27 of `metadata.csv`, update `participant_id` from `BHarthi` to `Bharthi` and filename from `BHarthi_LEFT_THUMB_distorted.png` to `Bharthi_LEFT_THUMB_distorted.png` to ensure exact case-sensitive cross-platform parity.
3. **Resolve `ashitta` Ghost Entries:**
   - Option A (*Recommended if participant is available*): Capture the actual 10 fingers for participant `ashitta` under the new 10-finger protocol.
   - Option B (*Dataset hygiene*): Move lines 36 and 37 to a quarantined `FingerSure_Dataset/orphan_records.csv` and remove them from `physical_dataset_collection` so that CSV, MySQL, and Disk counts match at 84 records.
4. **Re-sync Verification:**
   - Run `validate_dataset_collection.py` to confirm 0 duplicates, 0 missing files, and 100% parity across disk, CSV, and database.

