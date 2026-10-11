"""
Populate MySQL database `external_fingerprint_db.external_clean_gallery`
using 316 clean fingerprint images from data/clean_fvc/ (1.png to 316.png).
Uses existing SourceAFIS implementation in the project.
Idempotent insertion/update.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mysql.connector
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

from matcher import SourceAFISMatcher

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
TARGET_DB = "external_fingerprint_db"
TARGET_TABLE = "external_clean_gallery"
DATA_DIR = PROJECT_ROOT / "data" / "clean_fvc"
TOTAL_IMAGES_EXPECTED = 316


def get_db_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=TARGET_DB,
    )


def ensure_table_schema(conn):
    """Ensure the table exists and allows NULL for template_data when generation fails."""
    cursor = conn.cursor()
    cursor.execute(
        f"""
        CREATE TABLE IF NOT EXISTS `{TARGET_TABLE}` (
            `fingerprint_id` INT NOT NULL,
            `template_data` LONGBLOB NULL,
            `minutiae_count` INT DEFAULT 0,
            `template_version` VARCHAR(50) DEFAULT 'sourceafis_v1',
            `generation_status` VARCHAR(50) DEFAULT 'generated',
            `created_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP,
            `updated_at` TIMESTAMP NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (`fingerprint_id`)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
    )
    # Ensure template_data column is nullable
    cursor.execute(f"ALTER TABLE `{TARGET_TABLE}` MODIFY COLUMN `template_data` LONGBLOB NULL;")
    conn.commit()
    cursor.close()


def populate_gallery():
    print("=" * 70)
    print("POPULATING external_fingerprint_db.external_clean_gallery")
    print("=" * 70)
    print(f"Target Database : {TARGET_DB}")
    print(f"Target Table    : {TARGET_TABLE}")
    print(f"Image Directory : {DATA_DIR}")
    print(f"Expected Range  : 1.png -> {TOTAL_IMAGES_EXPECTED}.png (IDs 1..{TOTAL_IMAGES_EXPECTED})")
    print()

    conn = get_db_connection()
    ensure_table_schema(conn)

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

    cursor = conn.cursor()

    total_images_processed = 0
    successful_templates = 0
    failed_templates = 0
    failed_records: List[Dict[str, Any]] = []
    minutiae_counts: List[int] = []

    t_start = time.time()

    for fid in range(1, TOTAL_IMAGES_EXPECTED + 1):
        filename = f"{fid}.png"
        img_path = DATA_DIR / filename
        total_images_processed += 1

        if not img_path.exists():
            # File missing on disk
            cursor.execute(
                upsert_sql,
                (
                    fid,
                    None,
                    0,
                    "sourceafis_v1",
                    "failed",
                ),
            )
            failed_templates += 1
            failed_records.append({
                "fingerprint_id": fid,
                "filename": filename,
                "reason": "File does not exist on disk"
            })
            continue

        try:
            image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                raise ValueError("Could not read image / corrupted file")

            template = matcher.extract_template(image)
            minutiae_cnt = template.get("minutiae_count", 0)

            if minutiae_cnt > 0:
                tpl_bytes = pickle.dumps(template, protocol=pickle.HIGHEST_PROTOCOL)
                cursor.execute(
                    upsert_sql,
                    (
                        fid,
                        tpl_bytes,
                        minutiae_cnt,
                        template.get("version", "sourceafis_v1"),
                        "generated",
                    ),
                )
                successful_templates += 1
                minutiae_counts.append(minutiae_cnt)
            else:
                cursor.execute(
                    upsert_sql,
                    (
                        fid,
                        None,
                        0,
                        "sourceafis_v1",
                        "failed",
                    ),
                )
                failed_templates += 1
                failed_records.append({
                    "fingerprint_id": fid,
                    "filename": filename,
                    "reason": "SourceAFIS extracted 0 minutiae"
                })

        except Exception as e:
            cursor.execute(
                upsert_sql,
                (
                    fid,
                    None,
                    0,
                    "sourceafis_v1",
                    "failed",
                ),
            )
            failed_templates += 1
            failed_records.append({
                "fingerprint_id": fid,
                "filename": filename,
                "reason": str(e)
            })

        if fid % 50 == 0 or fid == TOTAL_IMAGES_EXPECTED:
            conn.commit()
            print(f"Progress: {fid}/{TOTAL_IMAGES_EXPECTED} images processed (Success: {successful_templates}, Failed: {failed_templates})")

    conn.commit()
    elapsed = time.time() - t_start
    print(f"\nProcessing finished in {elapsed:.2f} seconds.")

    # Verification queries
    cursor.execute(f"SELECT COUNT(*) FROM `{TARGET_TABLE}`;")
    total_rows = cursor.fetchone()[0]

    cursor.execute(
        f"""
        SELECT `generation_status`, COUNT(*)
        FROM `{TARGET_TABLE}`
        GROUP BY `generation_status`;
        """
    )
    status_counts = cursor.fetchall()

    min_minutiae = min(minutiae_counts) if minutiae_counts else 0
    max_minutiae = max(minutiae_counts) if minutiae_counts else 0
    avg_minutiae = sum(minutiae_counts) / len(minutiae_counts) if minutiae_counts else 0.0

    print("\n" + "=" * 70)
    print("FINAL SUMMARY REPORT")
    print("=" * 70)
    print(f"Total images processed         : {total_images_processed}")
    print(f"Successful templates           : {successful_templates}")
    print(f"Failed templates               : {failed_templates}")
    print(f"Failed IDs/filenames           : {[(r['fingerprint_id'], r['filename'], r['reason']) for r in failed_records]}")
    print(f"Minimum minutiae count         : {min_minutiae}")
    print(f"Maximum minutiae count         : {max_minutiae}")
    print(f"Average minutiae count         : {avg_minutiae:.2f}")
    print(f"Total rows in external_clean_gallery : {total_rows}")
    print()

    print("SQL QUERY: SELECT COUNT(*) FROM external_clean_gallery;")
    print(f"Result: {total_rows}")
    print()

    print("SQL QUERY: SELECT generation_status, COUNT(*) FROM external_clean_gallery GROUP BY generation_status;")
    for status, count in status_counts:
        print(f"  {status}: {count}")
    print("=" * 70)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    populate_gallery()
