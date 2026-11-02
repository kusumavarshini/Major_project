#!/usr/bin/env python3
"""
Enroll FingerSure Gallery
Standalone SourceAFIS enrollment script for FingerSure clean reference fingerprints.
Extracts minutiae templates from FingerSure_Dataset/clean/ and saves them to
FingerSure_Dataset/sourceafis_gallery/ with an exact manifest CSV and verification.
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np

# Set project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcher import SourceAFISMatcher

DATASET_ROOT = PROJECT_ROOT / "FingerSure_Dataset"
CLEAN_DIR = DATASET_ROOT / "clean"
GALLERY_DIR = DATASET_ROOT / "sourceafis_gallery"
MANIFEST_CSV_PATH = GALLERY_DIR / "gallery_manifest.csv"


def parse_filename(filename: str) -> Tuple[str, str, str]:
    """
    Parse participant_id, finger, and identity from clean image filename.
    Format: <Participant>_<FINGER>_clean.png
    Example: Kusuma_RIGHT_THUMB_clean.png
             identity: Kusuma_RIGHT_THUMB
             participant_id: Kusuma
             finger: RIGHT_THUMB
    """
    base = filename[:-10] if filename.endswith("_clean.png") else filename.replace(".png", "")
    identity = base

    if "_RIGHT_THUMB" in base:
        participant_id = base.replace("_RIGHT_THUMB", "")
        finger = "RIGHT_THUMB"
    elif "_LEFT_THUMB" in base:
        participant_id = base.replace("_LEFT_THUMB", "")
        finger = "LEFT_THUMB"
    else:
        participant_id = base
        finger = "UNKNOWN"

    return participant_id, finger, identity


def enroll_gallery(force: bool = False):
    print("=" * 75)
    print("      FINGERSURE SOURCEAFIS GALLERY ENROLLMENT")
    print("=" * 75)
    print(f"Clean Source Directory: {CLEAN_DIR}")
    print(f"Gallery Output Directory: {GALLERY_DIR}")
    print(f"Manifest Output Path:     {MANIFEST_CSV_PATH}")
    print("=" * 75)

    GALLERY_DIR.mkdir(parents=True, exist_ok=True)

    clean_images = sorted(list(CLEAN_DIR.glob("*_clean.png")))
    total_clean_images = len(clean_images)

    print(f"\n[Step 1/3] Discovered {total_clean_images} clean reference image(s).")
    if total_clean_images == 0:
        print("[!] No clean images found in FingerSure_Dataset/clean/")
        return

    matcher = SourceAFISMatcher()

    manifest_records: List[Dict[str, Any]] = []
    successful_extractions = 0
    failed_extractions = 0
    skipped_count = 0
    enrolled_identities = []

    print("\n[Step 2/3] Extracting SourceAFIS templates and saving to gallery...")

    t0_all = time.time()

    for idx, img_path in enumerate(clean_images, start=1):
        filename = img_path.name
        part_id, finger, identity = parse_filename(filename)
        tpl_filename = f"{identity}.dat"
        tpl_path = GALLERY_DIR / tpl_filename

        print(f"  [{idx:02d}/{total_clean_images:02d}] Identity: {identity:<26} ... ", end="", flush=True)

        # Check existing without force
        if tpl_path.exists() and not force:
            print("SKIPPED (Already enrolled, use --force to overwrite)")
            skipped_count += 1
            # Load existing template to read minutiae count
            try:
                with open(tpl_path, "rb") as f:
                    cached_tpl = pickle.load(f)
                mc = cached_tpl.get("minutiae_count", 0)
                status = "SUCCESS"
                err_msg = ""
                successful_extractions += 1
                enrolled_identities.append(identity)
            except Exception as e:
                mc = 0
                status = "FAILED"
                err_msg = f"Existing template corrupt: {e}"
                failed_extractions += 1

            manifest_records.append({
                "participant_id": part_id,
                "finger": finger,
                "identity": identity,
                "source_image": filename,
                "template_filename": tpl_filename,
                "minutiae_count": mc,
                "extraction_status": status,
                "error_message": err_msg,
            })
            continue

        # Load clean image
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None or img.size == 0:
            print("FAILED (Cannot read image file)")
            failed_extractions += 1
            manifest_records.append({
                "participant_id": part_id,
                "finger": finger,
                "identity": identity,
                "source_image": filename,
                "template_filename": tpl_filename,
                "minutiae_count": 0,
                "extraction_status": "FAILED",
                "error_message": "Failed to read or decode image file",
            })
            continue

        try:
            t_start = time.time()
            template = matcher.extract_template(img)
            t_elapsed = (time.time() - t_start) * 1000.0

            minutiae_cnt = template.get("minutiae_count", 0)
            if minutiae_cnt < 2:
                raise ValueError(f"Insufficient minutiae detected: {minutiae_cnt} < 2")

            # Serialize using project-standard pickle protocol
            tpl_bytes = pickle.dumps(template, protocol=pickle.HIGHEST_PROTOCOL)

            # Atomic save to gallery directory
            temp_path = tpl_path.with_suffix(".tmp")
            with open(temp_path, "wb") as f:
                f.write(tpl_bytes)
            temp_path.replace(tpl_path)

            successful_extractions += 1
            enrolled_identities.append(identity)
            print(f"ENROLLED ({minutiae_cnt} minutiae, {t_elapsed:.1f}ms)")

            manifest_records.append({
                "participant_id": part_id,
                "finger": finger,
                "identity": identity,
                "source_image": filename,
                "template_filename": tpl_filename,
                "minutiae_count": minutiae_cnt,
                "extraction_status": "SUCCESS",
                "error_message": "",
            })

        except Exception as e:
            failed_extractions += 1
            print(f"FAILED ({e})")
            manifest_records.append({
                "participant_id": part_id,
                "finger": finger,
                "identity": identity,
                "source_image": filename,
                "template_filename": tpl_filename,
                "minutiae_count": 0,
                "extraction_status": "FAILED",
                "error_message": str(e),
            })

    total_enroll_time = time.time() - t0_all

    # Write gallery manifest CSV
    print(f"\nWriting gallery manifest to: {MANIFEST_CSV_PATH}")
    fieldnames = [
        "participant_id",
        "finger",
        "identity",
        "source_image",
        "template_filename",
        "minutiae_count",
        "extraction_status",
        "error_message",
    ]
    with open(MANIFEST_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_records)

    # Step 3: Verification of Enrolled Templates
    print("\n[Step 3/3] Verifying that every enrolled template loads and is accepted by SourceAFIS...")
    verified_count = 0
    verification_errors = []

    for rec in manifest_records:
        if rec["extraction_status"] != "SUCCESS":
            continue

        tpl_path = GALLERY_DIR / rec["template_filename"]
        try:
            with open(tpl_path, "rb") as f:
                loaded = pickle.load(f)

            # Check template integrity
            assert isinstance(loaded, dict), f"Loaded template is not a dict: {type(loaded)}"
            assert "minutiae" in loaded, "Missing 'minutiae' key in template"
            assert "minutiae_count" in loaded, "Missing 'minutiae_count' key in template"
            assert loaded["minutiae_count"] == rec["minutiae_count"], "Minutiae count mismatch"

            # Execute matcher self-verification test (1:1 self-match)
            match_res = matcher.match_1to1(loaded, loaded)
            assert match_res.match is True, f"Self-match failed to meet threshold: score={match_res.score}"
            assert match_res.matched_minutiae_count == loaded["minutiae_count"], "Self-matched minutiae mismatch"

            verified_count += 1
        except Exception as e:
            verification_errors.append((rec["identity"], str(e)))

    if verification_errors:
        print(f"[!] Warning: {len(verification_errors)} template(s) failed verification check:")
        for ident, err in verification_errors:
            print(f"    - {ident}: {err}")
    else:
        print(f"[OK] All {verified_count} enrolled templates verified successfully with SourceAFIS!")

    # Summary Report
    print("\n" + "=" * 75)
    print("                 GALLERY ENROLLMENT SUMMARY REPORT")
    print("=" * 75)
    print(f"  Total Clean Images:              {total_clean_images}")
    print(f"  Successful Template Extractions: {successful_extractions}")
    print(f"  Failed Extractions:              {failed_extractions}")
    print(f"  Total Enrolled Identities:       {len(enrolled_identities)}")
    print(f"  Total Verified Usable Templates: {verified_count}")
    print(f"  Total Processing Time:           {total_enroll_time:.2f} seconds")
    print(f"  Gallery Directory:               {GALLERY_DIR}")
    print(f"  Gallery Manifest CSV:            {MANIFEST_CSV_PATH}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="Enroll clean fingerprints into SourceAFIS gallery.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing enrolled templates.")
    args = parser.parse_args()

    enroll_gallery(force=args.force)


if __name__ == "__main__":
    main()

