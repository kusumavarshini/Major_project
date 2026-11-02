#!/usr/bin/env python3
"""
Migrate SourceAFIS reference templates from local .dat files into MySQL database 'fingersure_eval_db'.
Completely isolated from the production database.
"""

from __future__ import annotations

import csv
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mysql.connector
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcher import SourceAFISMatcher

# Load database credentials from .env
load_dotenv(PROJECT_ROOT / ".env", override=True)

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
TARGET_DB = "fingersure_eval_db"
TARGET_TABLE = "fingerprint_gallery"

DATASET_ROOT = PROJECT_ROOT / "FingerSure_Dataset"
GALLERY_DIR = DATASET_ROOT / "sourceafis_gallery"
MANIFEST_PATH = GALLERY_DIR / "gallery_manifest.csv"
if not MANIFEST_PATH.exists():
    MANIFEST_PATH = DATASET_ROOT / "gallery_manifest.csv"


def get_mysql_server_connection():
    """Connect to MySQL server instance without selecting a database."""
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        charset="utf8mb4",
        use_pure=True,
    )


def get_eval_db_connection():
    """Connect directly to fingersure_eval_db."""
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=TARGET_DB,
        charset="utf8mb4",
        use_pure=True,
    )


def setup_database_and_table(drop_if_exists: bool = False) -> None:
    """Create the evaluation database and reference gallery table."""
    conn = get_mysql_server_connection()
    cursor = conn.cursor()
    try:
        # Create database
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{TARGET_DB}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        conn.commit()
        print(f"[OK] Database '{TARGET_DB}' verified / created.")
    finally:
        cursor.close()
        conn.close()

    # Create table inside fingersure_eval_db
    conn = get_eval_db_connection()
    cursor = conn.cursor()
    try:
        if drop_if_exists:
            cursor.execute(f"DROP TABLE IF EXISTS `{TARGET_TABLE}`;")
            conn.commit()
            print(f"[!] Dropped previous '{TARGET_TABLE}' table (--reset specified).")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS `{TARGET_TABLE}` (
            id INT AUTO_INCREMENT PRIMARY KEY,
            participant_id VARCHAR(100) NOT NULL,
            finger VARCHAR(50) NOT NULL,
            identity VARCHAR(150) NOT NULL UNIQUE,
            source_image VARCHAR(255) NOT NULL,
            minutiae_count INT NOT NULL,
            template_data LONGBLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_participant (participant_id),
            INDEX idx_finger (finger)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"[OK] Table '{TARGET_TABLE}' verified / created with schema.")
    finally:
        cursor.close()
        conn.close()


def migrate_templates(allow_overwrite: bool = False) -> Dict[str, Any]:
    """Read .dat files and insert serialized templates into MySQL."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Gallery manifest not found at: {MANIFEST_PATH}")
    if not GALLERY_DIR.exists():
        raise FileNotFoundError(f"Gallery directory not found at: {GALLERY_DIR}")

    # Read manifest
    records_to_insert = []
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("extraction_status", "").upper() == "SUCCESS":
                records_to_insert.append(row)

    print(f"\n[Step 1] Read {len(records_to_insert)} successful entries from manifest.")

    conn = get_eval_db_connection()
    cursor = conn.cursor(dictionary=True)

    inserted_count = 0
    skipped_count = 0
    failed_count = 0

    try:
        # Check existing identities to prevent duplicates and accidental overwrites
        cursor.execute(f"SELECT identity FROM `{TARGET_TABLE}`")
        existing_identities = {r["identity"] for r in cursor.fetchall()}
        print(f"[*] Currently existing records in `{TARGET_TABLE}`: {len(existing_identities)}")

        insert_sql = f"""
        INSERT INTO `{TARGET_TABLE}` (
            participant_id, finger, identity, source_image, minutiae_count, template_data
        ) VALUES (%s, %s, %s, %s, %s, %s);
        """

        update_sql = f"""
        UPDATE `{TARGET_TABLE}`
        SET participant_id = %s, finger = %s, source_image = %s, minutiae_count = %s, template_data = %s
        WHERE identity = %s;
        """

        for idx, row in enumerate(records_to_insert, 1):
            identity = row["identity"]
            participant_id = row["participant_id"]
            finger = row["finger"]
            source_image = row["source_image"]
            tpl_filename = row["template_filename"]
            dat_path = GALLERY_DIR / tpl_filename

            if not dat_path.exists():
                print(f"  [{idx:02d}/{len(records_to_insert)}] FAILED: Missing file {dat_path.name}")
                failed_count += 1
                continue

            with open(dat_path, "rb") as f:
                raw_bytes = f.read()

            # Verify deserializability before inserting
            try:
                tpl_obj = pickle.loads(raw_bytes)
                minutiae_count = int(tpl_obj.get("minutiae_count", len(tpl_obj.get("minutiae", []))))
            except Exception as e:
                print(f"  [{idx:02d}/{len(records_to_insert)}] FAILED: Deserialization check failed for {identity}: {e}")
                failed_count += 1
                continue

            if identity in existing_identities:
                if allow_overwrite:
                    cursor.execute(update_sql, (participant_id, finger, source_image, minutiae_count, raw_bytes, identity))
                    print(f"  [{idx:02d}/{len(records_to_insert)}] UPDATED: {identity} ({minutiae_count} minutiae, {len(raw_bytes)} bytes)")
                    inserted_count += 1
                else:
                    print(f"  [{idx:02d}/{len(records_to_insert)}] SKIPPED (Already exists): {identity}")
                    skipped_count += 1
            else:
                cursor.execute(insert_sql, (participant_id, finger, identity, source_image, minutiae_count, raw_bytes))
                print(f"  [{idx:02d}/{len(records_to_insert)}] INSERTED: {identity} ({minutiae_count} minutiae, {len(raw_bytes)} bytes)")
                inserted_count += 1
                existing_identities.add(identity)

        conn.commit()
    finally:
        cursor.close()
        conn.close()

    return {
        "manifest_count": len(records_to_insert),
        "inserted_or_updated": inserted_count,
        "skipped": skipped_count,
        "failed": failed_count,
    }


def verify_database_gallery() -> Dict[str, Any]:
    """Retrieve all templates from MySQL, deserialize, and run 1:1 match verification."""
    print("\n[Step 2] Verifying records in MySQL database...")
    conn = get_eval_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            f"""
            SELECT id, participant_id, finger, identity, source_image, minutiae_count, template_data
            FROM `{TARGET_TABLE}`
            ORDER BY identity ASC;
            """
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    total_rows = len(rows)
    print(f"[*] Retrieved {total_rows} row(s) from `{TARGET_DB}.{TARGET_TABLE}`.")

    deserialized_templates = []
    deserialization_failures = 0

    for r in rows:
        identity = r["identity"]
        raw_blob = r["template_data"]
        try:
            tpl = pickle.loads(raw_blob)
            if not isinstance(tpl, dict) or "minutiae" not in tpl:
                raise ValueError("Loaded object is not a valid SourceAFIS template dict")
            deserialized_templates.append({
                "id": r["id"],
                "identity": identity,
                "participant_id": r["participant_id"],
                "finger": r["finger"],
                "source_image": r["source_image"],
                "minutiae_count": r["minutiae_count"],
                "template": tpl,
            })
        except Exception as e:
            print(f"  [ERROR] Failed to deserialize template for '{identity}': {e}")
            deserialization_failures += 1

    print(f"[*] Successfully deserialized {len(deserialized_templates)}/{total_rows} templates.")

    # 1:1 Matching Verification using SourceAFIS
    matcher = SourceAFISMatcher()
    sample_tests = []
    if len(deserialized_templates) >= 2:
        # Test 1: Self-match (genuine pair)
        tpl_a = deserialized_templates[0]
        res_self = matcher.match_1to1(tpl_a["template"], tpl_a["template"])
        sample_tests.append({
            "test": "Genuine Self-Match",
            "identity_a": tpl_a["identity"],
            "identity_b": tpl_a["identity"],
            "score": res_self.score,
            "decision": res_self.match,
            "matched_minutiae": res_self.matched_minutiae_count,
        })

        # Test 2: Another genuine self-match from middle of gallery
        mid_idx = len(deserialized_templates) // 2
        tpl_mid = deserialized_templates[mid_idx]
        res_mid = matcher.match_1to1(tpl_mid["template"], tpl_mid["template"])
        sample_tests.append({
            "test": "Genuine Self-Match",
            "identity_a": tpl_mid["identity"],
            "identity_b": tpl_mid["identity"],
            "score": res_mid.score,
            "decision": res_mid.match,
            "matched_minutiae": res_mid.matched_minutiae_count,
        })

        # Test 3: Impostor cross-match (different persons)
        tpl_b = deserialized_templates[1]
        res_cross = matcher.match_1to1(tpl_a["template"], tpl_b["template"])
        sample_tests.append({
            "test": "Impostor Cross-Match",
            "identity_a": tpl_a["identity"],
            "identity_b": tpl_b["identity"],
            "score": res_cross.score,
            "decision": res_cross.match,
            "matched_minutiae": res_cross.matched_minutiae_count,
        })

    print("\n[Step 3] Sample 1:1 SourceAFIS Verification Matches with MySQL-Retrieved Templates:")
    for t in sample_tests:
        print(f"  - [{t['test']}] {t['identity_a']} vs {t['identity_b']}: "
              f"Score = {t['score']:.2f}, Match = {t['decision']}, Matched Minutiae = {t['matched_minutiae']}")

    return {
        "total_rows": total_rows,
        "deserialized_count": len(deserialized_templates),
        "deserialization_failures": deserialization_failures,
        "sample_tests": sample_tests,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Migrate FingerSure clean reference gallery to MySQL.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate the gallery table if it already exists.")
    parser.add_argument("--overwrite", action="store_true", help="Allow updating existing identities.")
    args = parser.parse_args()

    print("===========================================================================")
    print("      FINGERSURE SOURCEAFIS MYSQL GALLERY MIGRATION")
    print("===========================================================================")
    print(f"Target Database:    {TARGET_DB}")
    print(f"Target Table:       {TARGET_TABLE}")
    print(f"Source Gallery Dir: {GALLERY_DIR}")
    print(f"Source Manifest:    {MANIFEST_PATH}")
    print("===========================================================================\n")

    setup_database_and_table(drop_if_exists=args.reset)
    migration_stats = migrate_templates(allow_overwrite=args.overwrite)
    verification_stats = verify_database_gallery()

    print("\n===========================================================================")
    print("                 MIGRATION & VERIFICATION SUMMARY")
    print("===========================================================================")
    print(f"  Target Database:               {TARGET_DB}")
    print(f"  Target Table:                  {TARGET_TABLE}")
    print(f"  Manifest Entries:              {migration_stats['manifest_count']}")
    print(f"  Inserted / Updated in MySQL:   {migration_stats['inserted_or_updated']}")
    print(f"  Skipped (Existing):            {migration_stats['skipped']}")
    print(f"  Failed Insertions:             {migration_stats['failed']}")
    print(f"  Total Valid Records in DB:     {verification_stats['total_rows']}")
    print(f"  Successfully Deserialized:     {verification_stats['deserialized_count']}")
    print(f"  Deserialization Failures:      {verification_stats['deserialization_failures']}")
    print("===========================================================================")


if __name__ == "__main__":
    main()

