"""
Populate and validate newDB.external_fvc_clean_gallery using the 317 clean images
from data/clean_fvc_new/ (1.png to 317.png).
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mysql.connector
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from matcher import SourceAFISMatcher

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
TARGET_DB = "newdb"
TARGET_TABLE = "external_fvc_clean_gallery"
DATA_DIR = PROJECT_ROOT / "data" / "clean_fvc_new"
TOTAL_IMAGES_EXPECTED = 317

def get_db_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=TARGET_DB,
    )

def ensure_table():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{TARGET_TABLE}` (
            `fingerprint_id` INT PRIMARY KEY,
            `template_data` LONGBLOB NULL,
            `minutiae_count` INT DEFAULT 0,
            `template_version` VARCHAR(50) DEFAULT 'sourceafis_v1',
            `generation_status` VARCHAR(50) DEFAULT 'pending',
            `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            `updated_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    )
    conn.commit()
    cursor.close()
    conn.close()

def run_pipeline():
    ensure_table()
    conn = get_db_connection()
    cursor = conn.cursor()

    matcher = SourceAFISMatcher()

    upsert_sql = f"""
        INSERT INTO `{TARGET_TABLE}` (
            `fingerprint_id`,
            `template_data`,
            `minutiae_count`,
            `template_version`,
            `generation_status`
        ) VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            `template_data` = VALUES(`template_data`),
            `minutiae_count` = VALUES(`minutiae_count`),
            `template_version` = VALUES(`template_version`),
            `generation_status` = VALUES(`generation_status`);
    """

    print(f"Starting template generation for {TOTAL_IMAGES_EXPECTED} images...")
    start_time = time.time()

    processed_count = 0
    successful_count = 0
    failed_count = 0
    failed_ids = []
    
    # Process images 1..317
    for fid in range(1, TOTAL_IMAGES_EXPECTED + 1):
        filename = f"{fid}.png"
        img_path = DATA_DIR / filename
        processed_count += 1

        if not img_path.exists():
            cursor.execute(upsert_sql, (fid, None, 0, "sourceafis_v1", "failed"))
            failed_count += 1
            failed_ids.append((fid, "File not found on disk"))
            continue

        try:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                raise ValueError("Image read failed / empty image")

            template = matcher.extract_template(image)
            minutiae_cnt = template.get("minutiae_count", 0)

            if minutiae_cnt > 0:
                tpl_bytes = pickle.dumps(template, protocol=pickle.HIGHEST_PROTOCOL)
                version_str = template.get("version", "sourceafis_v1")
                cursor.execute(upsert_sql, (fid, tpl_bytes, minutiae_cnt, version_str, "generated"))
                successful_count += 1
            else:
                cursor.execute(upsert_sql, (fid, None, 0, "sourceafis_v1", "failed"))
                failed_count += 1
                failed_ids.append((fid, "0 minutiae extracted"))
        except Exception as e:
            cursor.execute(upsert_sql, (fid, None, 0, "sourceafis_v1", "failed"))
            failed_count += 1
            failed_ids.append((fid, str(e)))

        if fid % 50 == 0 or fid == TOTAL_IMAGES_EXPECTED:
            conn.commit()
            print(f"Processed {fid}/{TOTAL_IMAGES_EXPECTED} images ({successful_count} succeeded, {failed_count} failed)...")

    conn.commit()
    total_time = time.time() - start_time
    print(f"Template extraction and storage completed in {total_time:.2f}s.\n")

    # ==========================================
    # DATABASE & DESERIALIZATION VALIDATION
    # ==========================================
    print("Running full database & deserialization validation...")
    cursor.execute(f"SELECT fingerprint_id, template_data, minutiae_count, template_version, generation_status FROM `{TARGET_TABLE}` ORDER BY fingerprint_id ASC;")
    rows = cursor.fetchall()

    total_rows = len(rows)
    db_success = 0
    db_failed = 0
    null_templates = 0
    zero_minutiae = 0
    minutiae_list = []
    template_sizes = []
    versions = set()

    deser_success = 0
    deser_failures = 0
    count_mismatches = 0
    mismatch_details = []

    gallery_dict = {}

    for row in rows:
        fid, tpl_data, db_mc, tpl_ver, gen_status = row
        versions.add(tpl_ver)

        if gen_status == "generated" and tpl_data is not None:
            db_success += 1
            minutiae_list.append(db_mc)
            sz = len(tpl_data)
            template_sizes.append(sz)

            # Test deserialization
            try:
                deser_tpl = pickle.loads(tpl_data)
                deser_mc = deser_tpl.get("minutiae_count", len(deser_tpl.get("minutiae", [])))
                deser_success += 1
                gallery_dict[fid] = deser_tpl

                if deser_mc != db_mc:
                    count_mismatches += 1
                    mismatch_details.append((fid, db_mc, deser_mc))
            except Exception as e:
                deser_failures += 1
        else:
            db_failed += 1
            if tpl_data is None:
                null_templates += 1
            if db_mc == 0:
                zero_minutiae += 1

    # Check ID range and duplicates
    all_fids = [r[0] for r in rows]
    unique_fids = set(all_fids)
    missing_fids = sorted(set(range(1, TOTAL_IMAGES_EXPECTED + 1)) - unique_fids)
    duplicate_fids = len(all_fids) - len(unique_fids)

    # Min/Max/Avg calculations
    min_m = min(minutiae_list) if minutiae_list else 0
    max_m = max(minutiae_list) if minutiae_list else 0
    avg_m = sum(minutiae_list) / len(minutiae_list) if minutiae_list else 0.0

    min_sz = min(template_sizes) if template_sizes else 0
    max_sz = max(template_sizes) if template_sizes else 0
    avg_sz = sum(template_sizes) / len(template_sizes) if template_sizes else 0.0
    total_sz = sum(template_sizes)

    # ==========================================
    # SAMPLE VALIDATION (1, 2, 100, 200, 317)
    # ==========================================
    sample_ids = [1, 2, 100, 200, 317]
    sample_reports = {}
    for sid in sample_ids:
        cursor.execute(f"SELECT fingerprint_id, template_data, minutiae_count, template_version, generation_status FROM `{TARGET_TABLE}` WHERE fingerprint_id = %s;", (sid,))
        s_row = cursor.fetchone()
        if s_row:
            fid, tdata, dmc, tver, gstat = s_row
            if tdata:
                try:
                    dtpl = pickle.loads(tdata)
                    dmc_actual = dtpl.get("minutiae_count", 0)
                    dstatus = "Success"
                except Exception as e:
                    dmc_actual = 0
                    dstatus = f"Failed ({e})"
                sample_reports[sid] = {
                    "fid": fid,
                    "filename": f"{fid}.png",
                    "db_minutiae": dmc,
                    "deser_minutiae": dmc_actual,
                    "size_bytes": len(tdata),
                    "version": tver,
                    "status": dstatus
                }
            else:
                sample_reports[sid] = {
                    "fid": fid,
                    "filename": f"{fid}.png",
                    "db_minutiae": 0,
                    "deser_minutiae": 0,
                    "size_bytes": 0,
                    "version": tver,
                    "status": "NULL template"
                }

    # ==========================================
    # CLEAN SELF-MATCH SANITY CHECK
    # ==========================================
    print("Performing 1:N clean self-match sanity check for IDs 1, 2, 100, 200, 317...")
    self_match_results = []
    
    # Prepare gallery list for 1:N matcher
    gallery_list = []
    for g_fid, g_tpl in gallery_dict.items():
        gallery_list.append({"fingerprint_id": g_fid, "template": g_tpl})

    for qid in sample_ids:
        q_img_path = DATA_DIR / f"{qid}.png"
        if not q_img_path.exists():
            continue
        q_img = cv2.imread(str(q_img_path), cv2.IMREAD_GRAYSCALE)
        
        t0 = time.time()
        probe_tpl = matcher.extract_template(q_img)
        
        # Match against all items in gallery
        scores = []
        for g_item in gallery_list:
            cand_id = g_item["fingerprint_id"]
            cand_tpl = g_item["template"]
            res = matcher.match_1to1(probe_tpl, cand_tpl)
            scores.append((cand_id, res.raw_score, res.score))
        
        search_time_ms = (time.time() - t0) * 1000
        
        # Rank by raw_score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        top1_id, top1_raw, top1_score = scores[0]
        
        # Find expected ID rank and score
        expected_rank = -1
        expected_raw = 0.0
        for rank, (c_id, c_raw, c_sc) in enumerate(scores, start=1):
            if c_id == qid:
                expected_rank = rank
                expected_raw = c_raw
                break
                
        self_match_results.append({
            "query_id": qid,
            "expected_id": qid,
            "top1_id": top1_id,
            "top1_raw": top1_raw,
            "expected_raw": expected_raw,
            "expected_rank": expected_rank,
            "runtime_ms": search_time_ms
        })

    cursor.close()
    conn.close()

    # Output dictionary for consumption
    res_summary = {
        "source_images": TOTAL_IMAGES_EXPECTED,
        "processed": processed_count,
        "total_rows": total_rows,
        "successful_count": successful_count,
        "failed_count": failed_count,
        "failed_ids": failed_ids,
        "min_m": min_m,
        "max_m": max_m,
        "avg_m": avg_m,
        "min_sz": min_sz,
        "max_sz": max_sz,
        "avg_sz": avg_sz,
        "total_sz": total_sz,
        "versions": list(versions),
        "missing_fids": missing_fids,
        "duplicate_fids": duplicate_fids,
        "null_templates": null_templates,
        "zero_minutiae": zero_minutiae,
        "deser_success": deser_success,
        "deser_failures": deser_failures,
        "count_mismatches": count_mismatches,
        "sample_reports": sample_reports,
        "self_match_results": self_match_results
    }
    
    # Save results as json for exact reporting
    import json
    with open("scratch/newdb_gallery_report.json", "w") as f:
        json.dump(res_summary, f, indent=2)

    print("ALL DONE. Results written to scratch/newdb_gallery_report.json")

if __name__ == "__main__":
    run_pipeline()
