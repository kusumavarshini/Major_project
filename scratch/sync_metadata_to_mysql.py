"""
Sync existing metadata.csv records into MySQL physical_dataset_collection table.
"""

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import database

csv_path = PROJECT_ROOT / "FingerSure_Dataset" / "metadata.csv"
records = []
with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        records.append({
            "participant_id": r["participant_id"],
            "hand": r["hand"],
            "finger": r["finger"],
            "condition": int(r["condition"]),
            "filename": r["filename"],
            "image_path": r["image_path"],
            "capture_timestamp": r["capture_timestamp"].replace("T", " "),
            "image_width": int(r["image_width"]),
            "image_height": int(r["image_height"]),
            "capture_status": r["capture_status"],
            "scanner_model": r["scanner_model"],
            "scanner_serial": r["scanner_serial_number"],
            "image_quality": int(r["image_quality"]) if str(r["image_quality"]).isdigit() else 0,
        })

print(f"Syncing {len(records)} records to MySQL physical_dataset_collection...")
success_count = 0
for rec in records:
    if database.save_physical_capture_to_db(rec):
        success_count += 1

print(f"Successfully synced {success_count}/{len(records)} records into MySQL!")
stats = database.get_physical_collection_stats_from_db()
print("Updated MySQL Collection Stats:", stats)

