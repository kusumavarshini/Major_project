"""
Batch Rectification Pipeline for FingerSure Dataset using DDRNet
Applies fine-tuned DDRNet_DIR (experiments/ddrnet_fvc_finetuned/best_model.pth)
via the exact production rectify_with_ddrnet() function in authentication.py.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np

# Set project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "DDRNet"))

from authentication import rectify_with_ddrnet, DDRNET_CHECKPOINT

DATASET_ROOT = PROJECT_ROOT / "FingerSure_Dataset"
DISTORTED_DIR = DATASET_ROOT / "distorted"
CLEAN_DIR = DATASET_ROOT / "clean"
RECTIFIED_DIR = DATASET_ROOT / "rectified"
REPORT_CSV_PATH = DATASET_ROOT / "rectification_report.csv"
CONTACT_SHEET_PATH = DATASET_ROOT / "contact_sheet.png"


def run_batch_rectification():
    print("=" * 75)
    print("      FINGERSURE DATASET BATCH RECTIFICATION PIPELINE (DDRNet)")
    print("=" * 75)
    print(f"Active Checkpoint: {DDRNET_CHECKPOINT}")
    print(f"Input Directory:   {DISTORTED_DIR}")
    print(f"Clean Directory:   {CLEAN_DIR}")
    print(f"Output Directory:  {RECTIFIED_DIR}")
    print("=" * 75)

    RECTIFIED_DIR.mkdir(parents=True, exist_ok=True)

    distorted_files = sorted(list(DISTORTED_DIR.glob("*_distorted.png")))
    total_inputs = len(distorted_files)
    print(f"\nFound {total_inputs} distorted image(s) to process.")

    if total_inputs == 0:
        print("[!] No distorted images found in input directory.")
        return

    report_rows = []
    successful_count = 0
    failed_count = 0
    skipped_count = 0
    start_total_time = time.time()

    for idx, dist_path in enumerate(distorted_files, start=1):
        filename_dist = dist_path.name
        filename_rect = filename_dist.replace("_distorted.png", "_rectified.png")
        rect_path = RECTIFIED_DIR / filename_rect

        # Extract identity parts
        # Format: <Participant>_<FINGER>_distorted.png
        base_no_ext = filename_dist[:-14] # strip '_distorted.png'
        if "_RIGHT_THUMB" in base_no_ext:
            part_id = base_no_ext.replace("_RIGHT_THUMB", "")
            finger = "RIGHT_THUMB"
        elif "_LEFT_THUMB" in base_no_ext:
            part_id = base_no_ext.replace("_LEFT_THUMB", "")
            finger = "LEFT_THUMB"
        else:
            part_id = base_no_ext
            finger = "UNKNOWN"

        print(f"[{idx:02d}/{total_inputs:02d}] Processing: {filename_dist} ... ", end="", flush=True)

        # Check if already exists (Do not overwrite)
        if rect_path.exists():
            print("SKIPPED (Already exists)")
            skipped_count += 1
            # Read existing image to record stats
            existing_img = cv2.imread(str(rect_path), cv2.IMREAD_GRAYSCALE)
            h, w = existing_img.shape if existing_img is not None else (0, 0)
            report_rows.append({
                "participant_id": part_id,
                "finger": finger,
                "distorted_filename": filename_dist,
                "rectified_filename": filename_rect,
                "status": "SKIPPED",
                "inference_time_sec": 0.0,
                "input_width": w,
                "input_height": h,
                "output_width": w,
                "output_height": h,
                "input_mean": "",
                "output_mean": f"{float(np.mean(existing_img)):.2f}" if existing_img is not None else "",
                "output_std": f"{float(np.std(existing_img)):.2f}" if existing_img is not None else "",
                "checkpoint": str(DDRNET_CHECKPOINT),
                "error_message": "Output file already exists",
            })
            continue

        # Load input image
        input_img = cv2.imread(str(dist_path), cv2.IMREAD_GRAYSCALE)
        if input_img is None:
            print("FAILED (Could not read image)")
            failed_count += 1
            report_rows.append({
                "participant_id": part_id,
                "finger": finger,
                "distorted_filename": filename_dist,
                "rectified_filename": filename_rect,
                "status": "FAILED",
                "inference_time_sec": 0.0,
                "input_width": 0,
                "input_height": 0,
                "output_width": 0,
                "output_height": 0,
                "input_mean": "",
                "output_mean": "",
                "output_std": "",
                "checkpoint": str(DDRNET_CHECKPOINT),
                "error_message": "Failed to decode input image",
            })
            continue

        in_h, in_w = input_img.shape
        t0 = time.time()
        try:
            # Execute exact production rectification function
            rectified_img = rectify_with_ddrnet(input_img)
            elapsed = time.time() - t0

            # Validate output
            if rectified_img is None or rectified_img.size == 0:
                raise ValueError("Model produced empty output")

            out_h, out_w = rectified_img.shape
            if (out_h, out_w) != (in_h, in_w):
                raise ValueError(f"Output dimensions ({out_w}x{out_h}) do not match input ({in_w}x{in_h})")

            out_std = float(np.std(rectified_img))
            if out_std < 2.0:
                raise ValueError("Output image appears blank or uniform")

            # Save lossless PNG without modification or thresholding
            cv2.imwrite(str(rect_path), rectified_img)
            successful_count += 1
            print(f"DONE ({elapsed:.2f}s, {out_w}x{out_h})")

            report_rows.append({
                "participant_id": part_id,
                "finger": finger,
                "distorted_filename": filename_dist,
                "rectified_filename": filename_rect,
                "status": "SUCCESS",
                "inference_time_sec": f"{elapsed:.3f}",
                "input_width": in_w,
                "input_height": in_h,
                "output_width": out_w,
                "output_height": out_h,
                "input_mean": f"{float(np.mean(input_img)):.2f}",
                "output_mean": f"{float(np.mean(rectified_img)):.2f}",
                "output_std": f"{out_std:.2f}",
                "checkpoint": str(DDRNET_CHECKPOINT),
                "error_message": "",
            })

        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED ({e})")
            failed_count += 1
            report_rows.append({
                "participant_id": part_id,
                "finger": finger,
                "distorted_filename": filename_dist,
                "rectified_filename": filename_rect,
                "status": "FAILED",
                "inference_time_sec": f"{elapsed:.3f}",
                "input_width": in_w,
                "input_height": in_h,
                "output_width": 0,
                "output_height": 0,
                "input_mean": f"{float(np.mean(input_img)):.2f}",
                "output_mean": "",
                "output_std": "",
                "checkpoint": str(DDRNET_CHECKPOINT),
                "error_message": str(e),
            })

    total_time = time.time() - start_total_time

    # Write rectification_report.csv
    print(f"\nWriting rectification report to: {REPORT_CSV_PATH}")
    fieldnames = [
        "participant_id",
        "finger",
        "distorted_filename",
        "rectified_filename",
        "status",
        "inference_time_sec",
        "input_width",
        "input_height",
        "output_width",
        "output_height",
        "input_mean",
        "output_mean",
        "output_std",
        "checkpoint",
        "error_message",
    ]
    with open(REPORT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)

    # Generate Contact Sheet
    print("Generating comprehensive contact sheet...")
    generate_contact_sheet(distorted_files)

    # Final Verification of All Outputs
    print("\nRunning post-rectification integrity validation...")
    verify_all_outputs(total_inputs)

    # Summary
    print("\n" + "=" * 75)
    print("                 RECTIFICATION EXECUTION SUMMARY")
    print("=" * 75)
    print(f"  Checkpoint Used:          {DDRNET_CHECKPOINT}")
    print(f"  Total Input Images:       {total_inputs}")
    print(f"  Successfully Rectified:   {successful_count}")
    print(f"  Skipped (Existed):        {skipped_count}")
    print(f"  Failed:                   {failed_count}")
    print(f"  Total Processing Time:    {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    if successful_count > 0:
        times = [float(r["inference_time_sec"]) for r in report_rows if r["status"] == "SUCCESS"]
        print(f"  Average Time Per Image:   {np.mean(times):.2f} seconds")
    print("=" * 75)


def generate_contact_sheet(distorted_files: List[Path]):
    """
    Generate a high-resolution contact sheet comparing Clean, Distorted,
    and Rectified images for all 42 fingerprint identities.
    Organized as a multi-column visual comparison grid with text headers.
    """
    thumb_w, thumb_h = 150, 200
    padding = 10
    header_h = 40
    row_h = thumb_h + 30
    
    # 42 rows, 3 columns: [CLEAN | DISTORTED | RECTIFIED]
    col_w = thumb_w + 20
    total_w = (col_w * 3) + 200 # extra margin for identity label
    total_h = header_h + (row_h * len(distorted_files)) + 30

    # Create white canvas
    sheet = np.full((total_h, total_w), 255, dtype=np.uint8)

    # Top Header Banner
    cv2.putText(sheet, "FINGERSURE DATASET: CLEAN vs. DISTORTED vs. DDRNet RECTIFIED", (20, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,), 2, cv2.LINE_AA)

    # Column Labels
    col1_x = 220
    col2_x = col1_x + col_w
    col3_x = col2_x + col_w

    cv2.putText(sheet, "CLEAN (ORIGINAL)", (col1_x, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,), 1, cv2.LINE_AA)
    cv2.putText(sheet, "DISTORTED (INPUT)", (col2_x, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,), 1, cv2.LINE_AA)
    cv2.putText(sheet, "DDRNet RECTIFIED", (col3_x, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,), 1, cv2.LINE_AA)

    for i, dist_p in enumerate(distorted_files):
        fname_dist = dist_p.name
        fname_clean = fname_dist.replace("_distorted.png", "_clean.png")
        fname_rect = fname_dist.replace("_distorted.png", "_rectified.png")

        clean_p = CLEAN_DIR / fname_clean
        rect_p = RECTIFIED_DIR / fname_rect

        y_top = header_h + (i * row_h)

        # Identity label
        label = fname_dist.replace("_distorted.png", "")
        cv2.putText(sheet, f"#{i+1:02d} {label}", (10, y_top + thumb_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,), 1, cv2.LINE_AA)

        # 1. Clean thumbnail
        if clean_p.exists():
            c_img = cv2.imread(str(clean_p), cv2.IMREAD_GRAYSCALE)
            if c_img is not None:
                c_small = cv2.resize(c_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                sheet[y_top:y_top + thumb_h, col1_x:col1_x + thumb_w] = c_small

        # 2. Distorted thumbnail
        d_img = cv2.imread(str(dist_p), cv2.IMREAD_GRAYSCALE)
        if d_img is not None:
            d_small = cv2.resize(d_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            sheet[y_top:y_top + thumb_h, col2_x:col2_x + thumb_w] = d_small

        # 3. Rectified thumbnail
        if rect_p.exists():
            r_img = cv2.imread(str(rect_p), cv2.IMREAD_GRAYSCALE)
            if r_img is not None:
                r_small = cv2.resize(r_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                sheet[y_top:y_top + thumb_h, col3_x:col3_x + thumb_w] = r_small

        # Row separator line
        cv2.line(sheet, (10, y_top + thumb_h + 15), (total_w - 10, y_top + thumb_h + 15), (220,), 1)

    cv2.imwrite(str(CONTACT_SHEET_PATH), sheet)
    print(f"Contact sheet saved: {CONTACT_SHEET_PATH} ({total_w}x{total_h} pixels)")

    # Also generate a paginated 2-page version for convenient standard viewing
    rows_per_page = 21
    for page in [1, 2]:
        start_idx = (page - 1) * rows_per_page
        end_idx = start_idx + rows_per_page
        page_files = distorted_files[start_idx:end_idx]
        page_h = header_h + (row_h * len(page_files)) + 30
        page_sheet = np.full((page_h, total_w), 255, dtype=np.uint8)

        cv2.putText(page_sheet, f"FINGERSURE DATASET: RECTIFICATION COMPARISON (PAGE {page}/2)", (20, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,), 2, cv2.LINE_AA)
        cv2.putText(page_sheet, "CLEAN (ORIGINAL)", (col1_x, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,), 1, cv2.LINE_AA)
        cv2.putText(page_sheet, "DISTORTED (INPUT)", (col2_x, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,), 1, cv2.LINE_AA)
        cv2.putText(page_sheet, "DDRNet RECTIFIED", (col3_x, header_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,), 1, cv2.LINE_AA)

        for pi, dist_p in enumerate(page_files):
            fname_dist = dist_p.name
            fname_clean = fname_dist.replace("_distorted.png", "_clean.png")
            fname_rect = fname_dist.replace("_distorted.png", "_rectified.png")
            clean_p = CLEAN_DIR / fname_clean
            rect_p = RECTIFIED_DIR / fname_rect
            y_top = header_h + (pi * row_h)
            global_idx = start_idx + pi + 1
            label = fname_dist.replace("_distorted.png", "")

            cv2.putText(page_sheet, f"#{global_idx:02d} {label}", (10, y_top + thumb_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,), 1, cv2.LINE_AA)

            if clean_p.exists():
                c_img = cv2.imread(str(clean_p), cv2.IMREAD_GRAYSCALE)
                if c_img is not None:
                    page_sheet[y_top:y_top + thumb_h, col1_x:col1_x + thumb_w] = cv2.resize(c_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            d_img = cv2.imread(str(dist_p), cv2.IMREAD_GRAYSCALE)
            if d_img is not None:
                page_sheet[y_top:y_top + thumb_h, col2_x:col2_x + thumb_w] = cv2.resize(d_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            if rect_p.exists():
                r_img = cv2.imread(str(rect_p), cv2.IMREAD_GRAYSCALE)
                if r_img is not None:
                    page_sheet[y_top:y_top + thumb_h, col3_x:col3_x + thumb_w] = cv2.resize(r_img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

            cv2.line(page_sheet, (10, y_top + thumb_h + 15), (total_w - 10, y_top + thumb_h + 15), (220,), 1)

        page_path = DATASET_ROOT / f"contact_sheet_page_{page}.png"
        cv2.imwrite(str(page_path), page_sheet)
        print(f"Paginated contact sheet saved: {page_path}")


def verify_all_outputs(expected_count: int):
    """
    Verify every output file in FingerSure_Dataset/rectified:
    - Exists
    - Readable via cv2
    - Non-blank (std > 5)
    - Valid dimensions matching original 300x400
    - uint8 grayscale format
    """
    rectified_files = sorted(list(RECTIFIED_DIR.glob("*_rectified.png")))
    print(f"Total rectified files in directory: {len(rectified_files)} (Expected: {expected_count})")
    assert len(rectified_files) == expected_count, f"Count mismatch: {len(rectified_files)} != {expected_count}"

    for p in rectified_files:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        assert img is not None, f"Unreadable file: {p.name}"
        assert img.ndim == 2, f"Expected 2D grayscale, got {img.shape}: {p.name}"
        assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}: {p.name}"
        assert img.shape == (400, 300), f"Expected (400, 300), got {img.shape}: {p.name}"
        std_val = float(np.std(img))
        assert std_val > 5.0, f"Image appears blank (std={std_val:.2f}): {p.name}"

    print(f"[OK] All {len(rectified_files)} rectified images passed full integrity checks!")


if __name__ == "__main__":
    run_batch_rectification()

