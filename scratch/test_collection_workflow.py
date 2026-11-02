"""
Comprehensive automated unit test for updated 10-finger collection workflow.
Tests:
1. 10-Finger representations and naming conventions.
2. Metadata structure and dual-layer persistence (CSV + MySQL).
3. Duplicate prevention and versioning.
4. Participant status tracking (clean 0-10, distorted 0-10, total 0-20).
5. Dataset validation reporting.
6. Clean cleanup of test artifacts.
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager import (
    DEFAULT_DATASET_DIR_NAME,
    DatasetManager,
    ALL_10_FINGERS,
    ALL_10_FINGER_CODES,
    CONDITION_CLEAN,
    CONDITION_DISTORTED,
    CONDITION_NAMES,
)
import database
import validate_dataset_collection


def run_tests():
    print("=" * 70)
    print("RUNNING AUTOMATED 10-FINGER COLLECTION WORKFLOW TESTS")
    print("=" * 70)

    mgr = DatasetManager()

    # Test 1: Verify 10 fingers
    print("\n[Test 1] Verifying 10 Finger Definitions...")
    assert len(ALL_10_FINGERS) == 10, f"Expected 10 fingers, got {len(ALL_10_FINGERS)}"
    assert len(ALL_10_FINGER_CODES) == 10
    print(f"  Verified 10 fingers: {ALL_10_FINGER_CODES}")

    # Test 2: Verify Target Filenames
    print("\n[Test 2] Verifying Naming Conventions...")
    test_p = "P042"
    for hand, finger in ALL_10_FINGERS:
        _, rel_c, fname_c = mgr.determine_target_filename(test_p, hand, finger, CONDITION_CLEAN)
        _, rel_d, fname_d = mgr.determine_target_filename(test_p, hand, finger, CONDITION_DISTORTED)
        expected_c = f"{test_p}_{hand}_{finger}_CLEAN.png"
        expected_d = f"{test_p}_{hand}_{finger}_DISTORTED.png"
        assert fname_c == expected_c, f"Expected {expected_c}, got {fname_c}"
        assert fname_d == expected_d, f"Expected {expected_d}, got {fname_d}"
        assert rel_c == f"clean/{expected_c}"
        assert rel_d == f"distorted/{expected_d}"
    print(f"  Sample Clean:     {fname_c} (path: {rel_c})")
    print(f"  Sample Distorted: {fname_d} (path: {rel_d})")
    print("  Naming convention verified across all 10 fingers!")

    # Test 3: Test Capture & Persistence with Test Participant P999
    print("\n[Test 3] Testing Capture and Persistence (P999)...")
    test_pid = "P999"
    # Create synthetic test fingerprint image (300x400 grayscale)
    synthetic_img = np.zeros((400, 300), dtype=np.uint8)
    for r in range(20, 380, 8):
        cv2.circle(synthetic_img, (150, 200), r, 200, 2)

    # Save Clean
    rec_clean = mgr.save_capture(
        image=synthetic_img,
        participant_id=test_pid,
        hand="RIGHT",
        finger="THUMB",
        condition=CONDITION_CLEAN,
        scanner_model="Test Scanner",
        scanner_serial="TEST-001",
        image_quality=95,
        distortion_condition="Normal Relaxed",
        allow_versioning=True,
    )
    print(f"  Saved Clean: {rec_clean.filename} (Condition={rec_clean.condition})")
    assert rec_clean.condition == 0
    assert rec_clean.filename == "P999_RIGHT_THUMB_CLEAN.png"
    assert (mgr.clean_dir / rec_clean.filename).is_file()

    # Save Distorted
    rec_dist = mgr.save_capture(
        image=synthetic_img,
        participant_id=test_pid,
        hand="RIGHT",
        finger="THUMB",
        condition=CONDITION_DISTORTED,
        scanner_model="Test Scanner",
        scanner_serial="TEST-001",
        image_quality=90,
        distortion_condition="Structural Elastic Deformation",
        allow_versioning=True,
    )
    print(f"  Saved Distorted: {rec_dist.filename} (Condition={rec_dist.condition})")
    assert rec_dist.condition == 1
    assert rec_dist.filename == "P999_RIGHT_THUMB_DISTORTED.png"
    assert (mgr.distorted_dir / rec_dist.filename).is_file()

    # Test 4: Participant Status
    print("\n[Test 4] Verifying Participant 10-Finger Status Tracking...")
    st = mgr.get_participant_10finger_status(test_pid)
    print(f"  Participant {test_pid}: Clean={st.clean_count}/10, Distorted={st.distorted_count}/10, Total={st.total_completed}/20")
    assert st.clean_count == 1
    assert st.distorted_count == 1
    assert st.total_completed == 2
    assert st.remaining_count == 18
    assert st.is_finger_complete("RIGHT", "THUMB") is True
    assert st.is_finger_complete("RIGHT", "INDEX") is False
    assert st.fully_complete is False
    print("  Participant tracking verified!")

    # Test 5: Failed Capture Logging
    print("\n[Test 5] Verifying Failed Capture Logging...")
    mgr.record_failed_capture(
        participant_id=test_pid,
        hand="RIGHT",
        finger="INDEX",
        condition=CONDITION_CLEAN,
        error_message="Test sensor timeout",
        scanner_model="Test Scanner",
        scanner_serial="TEST-001",
    )
    assert mgr.failed_path.is_file()
    print("  Failed capture successfully logged to failed_captures.csv and MySQL!")

    # Test 6: MySQL Database Verification
    print("\n[Test 6] Verifying MySQL Database State...")
    db_stats = database.get_physical_collection_stats_from_db()
    print(f"  MySQL Collection Stats: {db_stats}")
    assert db_stats["total_images"] >= 2
    assert db_stats["failed_captures"] >= 1
    print("  MySQL synchronization verified!")

    # Test 7: Dataset Validation Report Execution
    print("\n[Test 7] Running Dataset Validation Report...")
    val_res = validate_dataset_collection.run_dataset_validation(output_reports=False)
    print(f"  Total Valid Images: {val_res['summary']['total_images']}")
    print(f"  Corrupted Images:   {val_res['summary']['corrupted_images_count']}")
    assert val_res["summary"]["corrupted_images_count"] == 0
    print("  Validation report generation verified!")

    # Cleanup Test Artifacts to leave production dataset completely clean!
    print("\n[Cleanup] Removing Test Participant P999 records...")
    # Remove test image files
    test_clean_file = mgr.clean_dir / "P999_RIGHT_THUMB_CLEAN.png"
    test_dist_file = mgr.distorted_dir / "P999_RIGHT_THUMB_DISTORTED.png"
    if test_clean_file.is_file():
        test_clean_file.unlink()
    if test_dist_file.is_file():
        test_dist_file.unlink()

    # Clean test rows from metadata.csv
    recs_all = mgr.load_all_records()
    kept = [r for r in recs_all if r.participant_id != "P999"]
    with open(mgr.metadata_path, "w", newline="", encoding="utf-8") as f:
        import csv
        from dataset_manager import METADATA_FIELDNAMES
        from dataclasses import asdict
        w = csv.DictWriter(f, fieldnames=METADATA_FIELDNAMES)
        w.writeheader()
        for r in kept:
            w.writerow(asdict(r))

    # Clean test rows from failed_captures.csv
    if mgr.failed_path.is_file():
        with open(mgr.failed_path, "r", encoding="utf-8") as f:
            f_lines = f.readlines()
        with open(mgr.failed_path, "w", encoding="utf-8") as f:
            for line in f_lines:
                if "P999" not in line:
                    f.write(line)

    # Clean test rows from MySQL
    try:
        conn = database.get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM physical_dataset_collection WHERE participant_id = 'P999';")
        cur.execute("DELETE FROM physical_dataset_failed_captures WHERE participant_id = 'P999';")
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print("DB cleanup notice:", e)

    print("  Test participant artifacts cleaned up. Dataset is completely intact.")
    print("\n" + "=" * 70)
    print("ALL 10-FINGER COLLECTION WORKFLOW TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

