# FingerSure — Terminal-Based Fingerprint Dataset Collection System

## Overview
This automated terminal collection application captures paired **CLEAN** and **DISTORTED** physical fingerprints for approximately 50 participants using the connected **SecuGen Hamster Pro 20** optical fingerprint scanner.

Collection is strictly limited to:
- **Right Thumb (`RIGHT_THUMB`)**
- **Left Thumb (`LEFT_THUMB`)**

---

## 1. Quick Launch Instructions

### Option A: Direct Python Command (Recommended)
Open a terminal (PowerShell or CMD) in the project folder and run:
```powershell
cd "c:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition"
python collect_fingerprints.py
```
*(Or activate `.venv` first if running in an isolated environment: `.venv\Scripts\activate`)*

### Option B: One-Click Batch Script
Double-click `start_collector.bat` in either:
- `c:\Users\kusum\OneDrive\Desktop\trial\start_collector.bat`
- `c:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\start_collector.bat`

---

## 2. Dataset Structure

All captures and metadata are stored exclusively inside the dedicated `FingerSure_Dataset` directory:

```text
FingerSure_Dataset/
│
├── clean/
│   ├── Rahul_RIGHT_THUMB_clean.png
│   ├── Rahul_LEFT_THUMB_clean.png
│   ├── Anu_RIGHT_THUMB_clean.png
│   └── ...
│
├── distorted/
│   ├── Rahul_RIGHT_THUMB_distorted.png
│   ├── Rahul_LEFT_THUMB_distorted.png
│   ├── Anu_RIGHT_THUMB_distorted.png
│   └── ...
│
└── metadata.csv
```

### Filename Format
- Clean Thumb: `<safe_id>_<FINGER>_clean.png` (e.g., `Rahul_RIGHT_THUMB_clean.png`)
- Distorted Thumb: `<safe_id>_<FINGER>_distorted.png` (e.g., `Rahul_RIGHT_THUMB_distorted.png`)
- Numbered Version (if re-collected with confirmation): `<safe_id>_<FINGER>_clean_v2.png`

---

## 3. Simplified Streamlined Collection Workflow

The workflow has exactly two capture modes and asks **zero distortion variation questions**:
1. Enter Participant Name / ID
2. Select Finger (Right Thumb or Left Thumb)
3. Select Capture Type (Clean or Distorted)
4. Press ENTER to scan physical finger on SecuGen scanner
5. Accept and save

### Step 1: Collect CLEAN Fingerprint for Participant 1 (e.g., Rahul)
1. Run `python collect_fingerprints.py`.
2. Enter Participant Name / ID: `Rahul`.
3. Select Finger: `1` (Right Thumb).
4. Select Capture Type: `1` (Clean Fingerprint).
5. Follow the clean instructions (clean/dry finger, relaxed natural contact on prism).
6. Press **ENTER**.
7. The SecuGen red LED turns on. Hold finger steady until capture finishes.
8. Confirm: `Accept and save this fingerprint? [Y/n]: y`.
9. The image is saved to `FingerSure_Dataset/clean/Rahul_RIGHT_THUMB_clean.png`.

### Step 2: Collect DISTORTED Fingerprint for the SAME Finger
1. At the completion prompt, choose: `1. Collect another fingerprint`.
2. Enter the same participant name: `Rahul`.
3. Select Finger: `1` (Right Thumb).
4. Select Capture Type: `2` (Distorted Fingerprint).
5. Place the finger with slight natural variation in placement or pressure.
6. Press **ENTER**, hold finger, inspect, and confirm acceptance.
7. The image is saved to `FingerSure_Dataset/distorted/Rahul_RIGHT_THUMB_distorted.png`.
8. The system displays:
   ```text
   Right Thumb:
   Clean Capture:     Completed
   Distorted Capture: Completed
   ```
   **The paired dataset for Rahul's Right Thumb is complete!**

---

## 4. Duplicate Protection & Data Integrity

1. **Duplicate Warning**:
   If you select a condition that already exists for that participant and finger, the script warns you:
   ```text
   A clean fingerprint already exists for Rahul - RIGHT_THUMB.
   1. Cancel
   2. Save a new numbered capture without overwriting the existing image.
   ```
   Selecting option 2 safely creates `Rahul_RIGHT_THUMB_clean_v2.png` without overwriting the original.
2. **Missing Clean Warning**:
   If you attempt to capture a distorted thumb before a clean one exists, the script warns you to ensure all distortions are paired.
3. **Lossless Saving**:
   Images are written directly as 8-bit uncompressed grayscale PNGs preserving raw sensor pixels without normalization, enhancement, or resizing.
4. **Preserved Integrity**:
   Existing datasets, DDRNet models, MySQL tables, and AFIS galleries remain completely untouched.
