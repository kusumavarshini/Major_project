"""
FingerSure Physical Fingerprint Dataset Collection Validation Report Generator
=============================================================================
Audits and validates the physical SecuGen fingerprint dataset against the
target of 42 participants x 20 captures (10 CLEAN, 10 DISTORTED = 840 total images).

Computes:
1. Total participants, CLEAN, DISTORTED, and total images (disk, CSV, MySQL).
2. Per-participant completeness (20 captures: 10 clean + 10 distorted).
3. Per-finger capture counts across all 10 fingers.
4. Missing captures per participant.
5. Duplicate record checks.
6. Failed capture logs audit.
7. Physical image file integrity checks (readability, dimensions, zero-pixel corruption).
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import cv2
import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager import (
    DEFAULT_DATASET_DIR_NAME,
    DatasetManager,
    ALL_10_FINGERS,
    ALL_10_FINGER_CODES,
    CONDITION_CLEAN,
    CONDITION_DISTORTED,
    CONDITION_NAMES,
    parse_hand_and_finger,
)


def run_dataset_validation(
    dataset_dir: Path | str | None = None,
    output_reports: bool = True
) -> Dict[str, Any]:
    """Execute complete dataset validation and return structured audit results."""
    if dataset_dir is None:
        dataset_dir = PROJECT_ROOT / DEFAULT_DATASET_DIR_NAME
    else:
        dataset_dir = Path(dataset_dir)

    clean_dir = dataset_dir / "clean"
    dist_dir = dataset_dir / "distorted"
    meta_path = dataset_dir / "metadata.csv"
    fail_path = dataset_dir / "failed_captures.csv"

    mgr = DatasetManager(dataset_dir)
    records = mgr.load_all_records()

    # 1. Disk scan of physical image files
    clean_files = sorted(list(clean_dir.glob("*.png"))) if clean_dir.exists() else []
    dist_files = sorted(list(dist_dir.glob("*.png"))) if dist_dir.exists() else []
    all_disk_files = clean_files + dist_files

    # 2. File Integrity Audit
    corrupted_files: List[Dict[str, str]] = []
    dimension_mismatches: List[Dict[str, Any]] = []
    valid_disk_images = 0

    for p in all_disk_files:
        try:
            if p.stat().st_size == 0:
                corrupted_files.append({"filename": p.name, "reason": "Zero-byte file"})
                continue

            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                corrupted_files.append({"filename": p.name, "reason": "Failed to decode PNG pixels"})
                continue

            h, w = img.shape
            if (h, w) != (400, 300) and (h, w) != (224, 224):
                dimension_mismatches.append({"filename": p.name, "dimensions": f"{w}x{h}"})

            # Check for completely blank / dead sensor capture
            if float(np.std(img)) < 1.0:
                corrupted_files.append({"filename": p.name, "reason": "Zero variance / blank white or black frame"})
                continue

            valid_disk_images += 1
        except Exception as e:
            corrupted_files.append({"filename": p.name, "reason": str(e)})

    # 3. Duplicate Records Detection
    seen_keys: Dict[Tuple[str, str, str, int], List[str]] = {}
    duplicate_records: List[Dict[str, Any]] = []

    for r in records:
        if r.capture_status != "accepted":
            continue
        key = (r.participant_id.upper(), r.hand.upper(), r.finger.upper(), r.condition)
        seen_keys.setdefault(key, []).append(r.filename)

    for (p_id, hand, finger, cond), fnames in seen_keys.items():
        if len(fnames) > 1:
            duplicate_records.append({
                "participant_id": p_id,
                "hand": hand,
                "finger": finger,
                "condition": CONDITION_NAMES.get(cond, "UNKNOWN"),
                "count": len(fnames),
                "filenames": fnames,
            })

    # 4. Failed Captures Log Audit
    failed_captures_count = 0
    recent_failed_captures: List[Dict[str, str]] = []
    if fail_path.exists():
        try:
            df_fail = pd.read_csv(fail_path)
            failed_captures_count = len(df_fail)
            recent_failed_captures = df_fail.tail(10).to_dict(orient="records")
        except Exception:
            pass

    # 5. Participant Breakdown & Missing Captures
    participants: Dict[str, Dict[Tuple[str, str], Dict[str, Any]]] = {}
    for r in records:
        if r.capture_status != "accepted":
            continue
        p_id = r.participant_id.upper()
        if p_id not in participants:
            participants[p_id] = {}
            for h, f in ALL_10_FINGERS:
                participants[p_id][(h, f)] = {"clean": False, "distorted": False}

        h, f = r.hand.upper(), r.finger.upper()
        if (h, f) in participants[p_id]:
            if r.condition == CONDITION_CLEAN:
                participants[p_id][(h, f)]["clean"] = True
            elif r.condition == CONDITION_DISTORTED:
                participants[p_id][(h, f)]["distorted"] = True

    participant_summaries: List[Dict[str, Any]] = []
    missing_captures_list: List[Dict[str, Any]] = []

    for p_id, finger_map in sorted(participants.items()):
        c_count = sum(1 for v in finger_map.values() if v["clean"])
        d_count = sum(1 for v in finger_map.values() if v["distorted"])
        tot = c_count + d_count

        missing_for_p = []
        for (h, f), st in finger_map.items():
            if not st["clean"]:
                missing_for_p.append(f"{h}_{f}_CLEAN")
                missing_captures_list.append({"participant_id": p_id, "hand": h, "finger": f, "condition": "CLEAN"})
            if not st["distorted"]:
                missing_for_p.append(f"{h}_{f}_DISTORTED")
                missing_captures_list.append({"participant_id": p_id, "hand": h, "finger": f, "condition": "DISTORTED"})

        participant_summaries.append({
            "participant_id": p_id,
            "clean_count": c_count,
            "distorted_count": d_count,
            "total_completed": tot,
            "remaining": 20 - tot,
            "is_complete": tot == 20,
            "missing_captures_count": len(missing_for_p),
            "missing_captures": missing_for_p,
        })

    # 6. Per-Finger Distribution
    per_finger_stats: Dict[str, Dict[str, int]] = {}
    for h, f in ALL_10_FINGERS:
        code = f"{h}_{f}"
        clean_sum = sum(1 for p_data in participants.values() if p_data.get((h, f), {}).get("clean"))
        dist_sum = sum(1 for p_data in participants.values() if p_data.get((h, f), {}).get("distorted"))
        per_finger_stats[code] = {
            "clean": clean_sum,
            "distorted": dist_sum,
            "total": clean_sum + dist_sum,
        }

    # 7. Aggregate Metrics vs 42-Participant Target
    total_participants = len(participants)
    total_clean = len(clean_files)
    total_distorted = len(dist_files)
    total_images = len(all_disk_files)

    complete_participants = sum(1 for p in participant_summaries if p["is_complete"])
    incomplete_participants = total_participants - complete_participants

    target_participants = 42
    target_clean = 420
    target_distorted = 420
    target_total = 840

    # Query MySQL database stats if available
    mysql_stats: Dict[str, Any] = {}
    try:
        import database
        mysql_stats = database.get_physical_collection_stats_from_db()
    except Exception as e:
        mysql_stats = {"error": str(e)}

    results = {
        "metadata": {
            "report_title": "FingerSure Physical Dataset Collection Validation Report",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset_directory": str(dataset_dir),
            "target_benchmark": {
                "target_participants": target_participants,
                "target_clean_per_participant": 10,
                "target_distorted_per_participant": 10,
                "target_total_per_participant": 20,
                "target_clean_images": target_clean,
                "target_distorted_images": target_distorted,
                "target_total_images": target_total,
            }
        },
        "summary": {
            "total_participants": total_participants,
            "complete_participants_count": complete_participants,
            "incomplete_participants_count": incomplete_participants,
            "total_clean_images": total_clean,
            "total_distorted_images": total_distorted,
            "total_images": total_images,
            "valid_disk_images": valid_disk_images,
            "corrupted_images_count": len(corrupted_files),
            "duplicate_records_count": len(duplicate_records),
            "failed_captures_count": failed_captures_count,
            "missing_captures_count": len(missing_captures_list),
            "progress_pct_total_images": round(float(total_images / target_total * 100), 2),
            "progress_pct_participants": round(float(complete_participants / target_participants * 100), 2),
        },
        "mysql_database_status": mysql_stats,
        "per_finger_counts": per_finger_stats,
        "per_participant_counts": participant_summaries,
        "corrupted_images": corrupted_files,
        "duplicate_records": duplicate_records,
        "failed_captures_recent": recent_failed_captures,
        "missing_captures_sample": missing_captures_list[:50],
    }

    if output_reports:
        # Save JSON Report
        json_path = dataset_dir / "dataset_collection_report.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[REPORT] Saved JSON report to {json_path}")

        # Save Markdown Report
        md_path = dataset_dir / "dataset_collection_report.md"
        generate_markdown_report(results, md_path)
        print(f"[REPORT] Saved Markdown report to {md_path}")

    return results


def generate_markdown_report(data: Dict[str, Any], output_path: Path):
    """Format audit results as a comprehensive GitHub Flavored Markdown document."""
    s = data["summary"]
    t = data["metadata"]["target_benchmark"]
    mysql_s = data.get("mysql_database_status", {})

    md = f"""# FingerSure – Physical Fingerprint Dataset Collection Validation Report

**Report Generated:** {data['metadata']['timestamp']}  
**Dataset Directory:** [`{data['metadata']['dataset_directory']}`](file:///{Path(data['metadata']['dataset_directory']).as_posix()})  
**Target Milestone:** 42 Participants $\\times$ 20 Captures (10 Clean, 10 Distorted) = **840 Total Images**  

---

## 1. Executive Summary & Progress Toward Target (42 Participants)

| Metric | Target Benchmark | Actual Recorded Count | Progress (%) | Audit Status |
| :--- | :---: | :---: | :---: | :---: |
| **Total Participants** | {t['target_participants']} | **{s['total_participants']}** | {s['total_participants']/t['target_participants']*100:.1f}% | {"ONGOING" if s['total_participants'] < t['target_participants'] else "MET / EXCEEDED"} |
| **Fully Complete Participants (20/20)** | {t['target_participants']} | **{s['complete_participants_count']}** | {s['progress_pct_participants']}% | {"PENDING FULL COMPLETION" if s['complete_participants_count'] < t['target_participants'] else "COMPLETE"} |
| **Incomplete Participants** | 0 | **{s['incomplete_participants_count']}** | — | {f"{s['incomplete_participants_count']} participants require remaining fingers" if s['incomplete_participants_count'] > 0 else "All enrolled complete"} |
| **Total CLEAN Images** | {t['target_clean_images']} | **{s['total_clean_images']}** | {s['total_clean_images']/t['target_clean_images']*100:.1f}% | Disk verified |
| **Total DISTORTED Images** | {t['target_distorted_images']} | **{s['total_distorted_images']}** | {s['total_distorted_images']/t['target_distorted_images']*100:.1f}% | Disk verified |
| **Total Fingerprint Images** | **{t['target_total_images']}** | **{s['total_images']}** | **{s['progress_pct_total_images']}%** | Physical PNG files |
| **Corrupted / Invalid Images** | 0 | **{s['corrupted_images_count']}** | — | {"VERIFIED CLEAN (0 errors)" if s['corrupted_images_count'] == 0 else f"WARNING: {s['corrupted_images_count']} issues"} |
| **Duplicate Records** | 0 | **{s['duplicate_records_count']}** | — | {"VERIFIED NO DUPLICATES" if s['duplicate_records_count'] == 0 else f"{s['duplicate_records_count']} duplicates"} |
| **Logged Failed Captures** | — | **{s['failed_captures_count']}** | — | Logged separately in `failed_captures.csv` |

---

## 2. MySQL Database Synchronization Status

- **Database Name:** `fingersure_eval_db`
- **Collection Table:** `physical_dataset_collection`
- **Total Images in MySQL:** **{mysql_s.get('total_images', 'N/A')}**
- **Clean Images in MySQL:** **{mysql_s.get('clean_images', 'N/A')}**
- **Distorted Images in MySQL:** **{mysql_s.get('distorted_images', 'N/A')}**
- **Participants in MySQL:** **{mysql_s.get('total_participants', 'N/A')}**
- **Failed Captures in MySQL:** **{mysql_s.get('failed_captures', 'N/A')}**

---

## 3. Per-Finger Capture Distribution (All 10 Fingers)

| Hand | Finger | Finger Code | Clean Count | Distorted Count | Total Images | Completeness |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""

    for code, counts in data["per_finger_counts"].items():
        hand, finger = code.split("_")
        tot = counts["total"]
        status = "Active" if tot > 0 else "Pending"
        md += f"| **{hand}** | {finger} | `{code}` | {counts['clean']} | {counts['distorted']} | **{tot}** | {status} |\n"

    md += """
---

## 4. Per-Participant Capture Breakdown

| Participant ID | Clean Captures (/10) | Distorted Captures (/10) | Total Completed (/20) | Remaining (/20) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""

    for p in data["per_participant_counts"]:
        st = "**COMPLETE (20/20)**" if p["is_complete"] else f"Pending ({p['remaining']} remaining)"
        md += f"| `{p['participant_id']}` | {p['clean_count']}/10 | {p['distorted_count']}/10 | **{p['total_completed']}/20** | {p['remaining']} | {st} |\n"

    md += """
---

## 5. Data Integrity, Corrupted Images & Duplicate Audit

### A. Corrupted or Unreadable Images
"""
    if data["corrupted_images"]:
        for c in data["corrupted_images"]:
            md += f"- **{c['filename']}:** {c['reason']}\n"
    else:
        md += "- **Zero corrupted images detected.** All physical PNG files decoded successfully with valid pixel gradients and non-zero dimensions.\n"

    md += "\n### B. Duplicate Record Audit\n"
    if data["duplicate_records"]:
        for d in data["duplicate_records"]:
            md += f"- **Participant `{d['participant_id']}` - {d['hand']} {d['finger']} ({d['condition']}):** {d['count']} occurrences ({', '.join(d['filenames'])})\n"
    else:
        md += "- **Zero duplicate entries detected.** All captures follow unique constraint `(participant_id, hand, finger, condition)`.\n"

    md += f"""\n### C. Failed Captures Log
- Total capture failures/rejections recorded: **{s['failed_captures_count']}**.
- Log file: [`FingerSure_Dataset/failed_captures.csv`](file:///{Path(data['metadata']['dataset_directory'] + '/failed_captures.csv').as_posix()}).
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md)


if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING FINGERSURE DATASET COLLECTION VALIDATION AUDIT")
    print("=" * 70)
    results = run_dataset_validation()
    s = results["summary"]
    print("\n[VALIDATION SUMMARY]")
    print(f"  Total Participants:       {s['total_participants']} (Complete: {s['complete_participants_count']}, Incomplete: {s['incomplete_participants_count']})")
    print(f"  Total Clean Images:       {s['total_clean_images']}")
    print(f"  Total Distorted Images:   {s['total_distorted_images']}")
    print(f"  Total Images on Disk:     {s['total_images']} / 840 target ({s['progress_pct_total_images']}%)")
    print(f"  Corrupted Images:         {s['corrupted_images_count']}")
    print(f"  Duplicate Records:        {s['duplicate_records_count']}")
    print(f"  Failed Captures Logged:   {s['failed_captures_count']}")
    print("=" * 70)

