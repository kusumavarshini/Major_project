"""
Script to migrate metadata.csv to the 10-finger schema.
"""

import csv
import shutil
from pathlib import Path

ds = Path(r"c:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\FingerSure_Dataset")
meta = ds / "metadata.csv"
backup = ds / "metadata_backup_pre_10finger.csv"

if not backup.exists():
    shutil.copyfile(meta, backup)
    print("Backup created at:", backup.name)

rows = []
with open(meta, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        p_id = r.get("participant_id", "").strip()
        orig_name = r.get("original_participant_name", "").strip() or p_id
        old_finger = r.get("finger", "").strip().upper()
        hand = "RIGHT" if "RIGHT" in old_finger else "LEFT"
        finger = "THUMB"
        for f_name in ["INDEX", "MIDDLE", "RING", "LITTLE", "THUMB"]:
            if f_name in old_finger:
                finger = f_name
                break

        c_type = r.get("capture_type", "").strip().lower()
        if c_type:
            cond = 0 if c_type == "clean" else 1
        else:
            cond_val = str(r.get("condition", "0")).strip().lower()
            cond = 0 if cond_val in ("0", "clean") else 1

        img_fn = r.get("image_filename", "").strip() or r.get("image_path", "").strip()
        cond_label = "CLEAN" if cond == 0 else "DISTORTED"
        fname = r.get("filename", "").strip() or (Path(img_fn).name if img_fn else f"{p_id}_{hand}_{finger}_{cond_label}.png")
        ipath = img_fn if img_fn else f"{cond_label.lower()}/{fname}"

        rows.append({
            "participant_id": p_id,
            "hand": hand,
            "finger": finger,
            "condition": cond,
            "filename": fname,
            "image_path": ipath,
            "capture_timestamp": r.get("capture_timestamp", ""),
            "image_width": int(r.get("image_width", 300) or 300),
            "image_height": int(r.get("image_height", 400) or 400),
            "capture_status": r.get("capture_status", "accepted") or "accepted",
            "original_participant_name": orig_name,
            "scanner_model": r.get("scanner_model", "SecuGen Hamster Pro 20"),
            "scanner_serial_number": r.get("scanner_serial_number", "Unknown"),
            "image_quality": r.get("image_quality", "N/A"),
            "nfiq": r.get("nfiq", "N/A"),
            "distortion_condition": r.get("distortion_condition", "Normal Relaxed"),
        })

NEW_FIELDNAMES = [
    "participant_id",
    "hand",
    "finger",
    "condition",
    "filename",
    "image_path",
    "capture_timestamp",
    "image_width",
    "image_height",
    "capture_status",
    "original_participant_name",
    "scanner_model",
    "scanner_serial_number",
    "image_quality",
    "nfiq",
    "distortion_condition",
]

with open(meta, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=NEW_FIELDNAMES)
    w.writeheader()
    w.writerows(rows)

print(f"Successfully migrated {len(rows)} rows to new metadata schema!")

