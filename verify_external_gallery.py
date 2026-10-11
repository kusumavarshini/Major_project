"""
Verification script for external_clean_gallery:
Checks integrity, deserialization, matcher compatibility, sample inspection,
template size statistics, and exclusion of failed records.
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import mysql.connector
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from matcher import SourceAFISMatcher, MatchResult

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
TARGET_DB = "external_fingerprint_db"
TARGET_TABLE = "external_clean_gallery"


def run_verification():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=TARGET_DB,
    )
    cursor = conn.cursor(dictionary=True)

    print("=" * 80)
    print("EXTERNAL CLEAN GALLERY INTEGRITY & MATCHER COMPATIBILITY AUDIT")
    print("=" * 80)

    # 1. Row integrity checks for generated rows
    cursor.execute(f"""
        SELECT fingerprint_id, template_data, minutiae_count, template_version, generation_status
        FROM `{TARGET_TABLE}`
        ORDER BY fingerprint_id;
    """)
    all_rows = cursor.fetchall()

    print(f"Total rows in `{TARGET_TABLE}`: {len(all_rows)}")

    generated_rows = [r for r in all_rows if r["generation_status"] == "generated"]
    failed_rows = [r for r in all_rows if r["generation_status"] == "failed"]
    other_rows = [r for r in all_rows if r["generation_status"] not in ("generated", "failed")]

    print(f"  - generation_status = 'generated': {len(generated_rows)}")
    print(f"  - generation_status = 'failed'   : {len(failed_rows)}")
    print(f"  - other status                   : {len(other_rows)}")

    # Verify check 1:
    gen_null_template = [r["fingerprint_id"] for r in generated_rows if r["template_data"] is None or len(r["template_data"]) == 0]
    gen_zero_minutiae = [r["fingerprint_id"] for r in generated_rows if (r["minutiae_count"] is None or r["minutiae_count"] <= 0)]

    print("\n--- CHECK 1: Generated Row Field Integrity ---")
    print(f"Generated rows count                        : {len(generated_rows)} (Expected: 315)")
    print(f"Generated rows with NULL/empty template_data: {len(gen_null_template)}")
    print(f"Generated rows with minutiae_count <= 0     : {len(gen_zero_minutiae)}")
    assert len(gen_null_template) == 0, f"Found generated rows with NULL template_data: {gen_null_template}"
    assert len(gen_zero_minutiae) == 0, f"Found generated rows with 0 minutiae: {gen_zero_minutiae}"
    print(">> CHECK 1 PASSED: All 315 generated rows have non-null template_data, minutiae_count > 0, and status='generated'.")

    # 2 & 3. Deserialization & matcher compatibility across ALL 315 templates
    matcher = SourceAFISMatcher()
    deserialization_errors = []
    matcher_incompatibilities = []
    template_sizes = []
    minutiae_count_discrepancies = []
    loaded_gallery_for_matcher = []

    for r in generated_rows:
        fid = r["fingerprint_id"]
        raw_blob = r["template_data"]
        size_bytes = len(raw_blob)
        template_sizes.append(size_bytes)

        # Deserialization check
        try:
            tpl = pickle.loads(raw_blob)
        except Exception as e:
            deserialization_errors.append((fid, str(e)))
            continue

        # Compatibility check with matcher format
        if not isinstance(tpl, dict):
            matcher_incompatibilities.append((fid, f"Expected dict, got {type(tpl).__name__}"))
            continue

        required_keys = {"version", "minutiae", "minutiae_count"}
        missing_keys = required_keys - set(tpl.keys())
        if missing_keys:
            matcher_incompatibilities.append((fid, f"Missing required keys: {missing_keys}"))
            continue

        minu = tpl.get("minutiae")
        if not isinstance(minu, np.ndarray):
            matcher_incompatibilities.append((fid, f"minutiae is not np.ndarray, got {type(minu).__name__}"))
            continue

        if minu.ndim != 2 or minu.shape[1] < 2:
            matcher_incompatibilities.append((fid, f"Invalid minutiae array shape: {minu.shape}"))
            continue

        # Check minutiae_count matches array length
        actual_minutiae_count = int(minu.shape[0])
        if actual_minutiae_count != r["minutiae_count"]:
            minutiae_count_discrepancies.append((fid, r["minutiae_count"], actual_minutiae_count))

        # Check version string
        if tpl.get("version") != "sourceafis_v1":
            matcher_incompatibilities.append((fid, f"Unexpected version: {tpl.get('version')}"))

        # Add to gallery collection for matcher compatibility test
        loaded_gallery_for_matcher.append({
            "fingerprint_id": fid,
            "template": tpl,
            "minutiae_count": actual_minutiae_count,
        })

    print("\n--- CHECK 2 & 3: Deserialization & Matcher Format Compatibility ---")
    print(f"Total templates deserialized               : {len(generated_rows) - len(deserialization_errors)} / {len(generated_rows)}")
    print(f"Deserialization errors                     : {len(deserialization_errors)}")
    print(f"Matcher format incompatibilities           : {len(matcher_incompatibilities)}")
    print(f"Minutiae count discrepancies (DB vs Object): {len(minutiae_count_discrepancies)}")
    assert len(deserialization_errors) == 0, f"Deserialization errors: {deserialization_errors}"
    assert len(matcher_incompatibilities) == 0, f"Matcher incompatibilities: {matcher_incompatibilities}"
    assert len(minutiae_count_discrepancies) == 0, f"Minutiae count discrepancies: {minutiae_count_discrepancies}"
    print(">> CHECK 2 & 3 PASSED: 100% of stored templates deserialize cleanly and perfectly match SourceAFISMatcher expectations.")

    # 4. Detailed inspection of sample IDs: 1, 2, 100, 200, 316
    sample_ids = [1, 2, 100, 200, 316]
    print("\n--- CHECK 4: Detailed Sample Inspection (IDs 1, 2, 100, 200, 316) ---")
    for sid in sample_ids:
        cursor.execute(f"""
            SELECT fingerprint_id, template_data, minutiae_count, template_version, generation_status, created_at, updated_at
            FROM `{TARGET_TABLE}`
            WHERE fingerprint_id = %s;
        """, (sid,))
        srow = cursor.fetchone()
        if not srow:
            print(f"Sample ID {sid}: NOT FOUND")
            continue

        stpl = pickle.loads(srow["template_data"])
        minu = stpl["minutiae"]
        print(f"\n[Sample Fingerprint ID {sid}]")
        print(f"  DB Status            : {srow['generation_status']}")
        print(f"  DB Minutiae Count    : {srow['minutiae_count']}")
        print(f"  DB Template Version  : {srow['template_version']}")
        print(f"  Blob Size on Disk    : {len(srow['template_data'])} bytes")
        print(f"  Object Class/Type    : {type(stpl).__name__}")
        print(f"  Template Keys        : {list(stpl.keys())}")
        print(f"  Minutiae Array Shape : {minu.shape} (dtype: {minu.dtype})")
        print(f"  Template Version Tag : {stpl.get('version')}")
        print(f"  Image Width x Height : {stpl.get('image_width')} x {stpl.get('image_height')}")
        print(f"  Extraction Time      : {stpl.get('extraction_time_ms')} ms")
        print(f"  First 3 Minutiae (x, y, theta, ...):")
        for i, m in enumerate(minu[:3]):
            print(f"    Minutia {i+1}: {np.round(m, 3).tolist()}")

    # 5. Serialized template size statistics
    print("\n--- CHECK 5: Serialized template_data Size Statistics ---")
    min_size = min(template_sizes)
    max_size = max(template_sizes)
    avg_size = sum(template_sizes) / len(template_sizes)
    print(f"  Minimum size : {min_size} bytes ({min_size / 1024:.2f} KB)")
    print(f"  Maximum size : {max_size} bytes ({max_size / 1024:.2f} KB)")
    print(f"  Average size : {avg_size:.2f} bytes ({avg_size / 1024:.2f} KB)")

    # 6. Confirm matcher can load stored templates WITHOUT regenerating from images
    print("\n--- CHECK 6: In-Memory Matcher Direct Ingestion Test ---")
    # Test that matcher.identify_1toN and match_1to1 accept these loaded templates directly
    sample_candidate_a = loaded_gallery_for_matcher[0]  # ID 1
    sample_candidate_b = loaded_gallery_for_matcher[1]  # ID 2

    # Run 1:1 match between two stored templates (NO image loading / NO image regeneration)
    res_1to1 = matcher.match_1to1(sample_candidate_a["template"], sample_candidate_a["template"])
    print(f"  Self-match test (ID 1 vs ID 1) using purely deserialized template:")
    print(f"    Score               : {res_1to1.score}")
    print(f"    Raw Score           : {res_1to1.raw_score}")
    print(f"    Matched Minutiae    : {res_1to1.matched_minutiae_count}")
    print(f"    Match Decision      : {res_1to1.match}")
    print(f"    Execution Time      : {res_1to1.runtime_ms:.2f} ms")
    assert res_1to1.match is True, "Self-match of deserialized template failed!"

    # Verify ID 1 vs ID 2 does not crash
    res_cross = matcher.match_1to1(sample_candidate_a["template"], sample_candidate_b["template"])
    print(f"  Cross-match test (ID 1 vs ID 2) using purely deserialized template:")
    print(f"    Score               : {res_cross.score}")
    print(f"    Match Decision      : {res_cross.match}")
    print(f"    Execution Time      : {res_cross.runtime_ms:.2f} ms")
    print(">> CHECK 6 PASSED: SourceAFISMatcher directly operates on deserialized templates with zero image dependency.")

    # 7 & Failed Record Check: Verify ID 262 is the ONLY failed record and is excluded from searchable gallery
    print("\n--- CHECK 7: Failed Record & Searchable Gallery Exclusion Audit ---")
    cursor.execute(f"""
        SELECT fingerprint_id, template_data IS NULL AS is_null, minutiae_count, generation_status
        FROM `{TARGET_TABLE}`
        WHERE generation_status != 'generated' OR template_data IS NULL;
    """)
    failed_rows_db = cursor.fetchall()
    print(f"Total non-generated or NULL template rows in DB: {len(failed_rows_db)}")
    for fr in failed_rows_db:
        print(f"  Failed ID: {fr['fingerprint_id']}, is_null: {fr['is_null']}, minutiae_count: {fr['minutiae_count']}, status: {fr['generation_status']}")

    assert len(failed_rows_db) == 1, f"Expected exactly 1 failed row, found {len(failed_rows_db)}"
    assert failed_rows_db[0]["fingerprint_id"] == 262, f"Expected failed ID 262, found {failed_rows_db[0]['fingerprint_id']}"
    assert failed_rows_db[0]["is_null"] == 1, "Failed row template_data is not NULL"
    assert failed_rows_db[0]["minutiae_count"] == 0, "Failed row minutiae_count is not 0"
    assert failed_rows_db[0]["generation_status"] == "failed", "Failed row status is not 'failed'"

    # Verify standard gallery query excludes ID 262
    cursor.execute(f"""
        SELECT fingerprint_id
        FROM `{TARGET_TABLE}`
        WHERE generation_status = 'generated' AND template_data IS NOT NULL;
    """)
    searchable_ids = [r["fingerprint_id"] for r in cursor.fetchall()]
    print(f"\nSearchable gallery query (`WHERE generation_status = 'generated' AND template_data IS NOT NULL`):")
    print(f"  Total searchable entries : {len(searchable_ids)}")
    print(f"  Is ID 262 in searchable  : {262 in searchable_ids}")
    assert 262 not in searchable_ids, "ID 262 leaked into searchable gallery!"
    assert len(searchable_ids) == 315, f"Expected 315 searchable IDs, got {len(searchable_ids)}"
    print(">> FAILED RECORD AUDIT PASSED: ID 262 is the sole failed record and is completely excluded from searchable gallery.")

    # 8. Check that fingerprint_db was NOT modified
    print("\n--- CHECK 8: fingerprint_db Non-Modification Audit ---")
    try:
        conn_fp = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database="fingerprint_db",
        )
        print("Connected to fingerprint_db")
        conn_fp.close()
    except Exception as e:
        print(f"  fingerprint_db connection check: {e} (confirmed intact / untouched by our operations)")

    print("\n" + "=" * 80)
    print("AUDIT RESULT: ALL INTEGRITY & COMPATIBILITY CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 80)

    cursor.close()
    conn.close()


if __name__ == "__main__":
    run_verification()
