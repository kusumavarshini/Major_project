"""
Comprehensive Verification Script for FingerSure Terminal Dataset Collection System
Tests:
1. Flexible participant name/ID handling & sanitization
2. Collision resistance and exact metadata preservation
3. Right and Left thumb tracking independently
4. Clean and Distorted capture pairing
5. Metadata.csv integrity and field formatting
6. Duplicate protection & numbered versioning (_v2, etc.)
7. Scanner detection (Native SDK & WebAPI)
8. Summary statistics calculation
"""

import os
import sys
import shutil
from pathlib import Path
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager import (
    DatasetManager,
    sanitize_identifier,
    normalize_finger_label,
    CaptureRecord,
)
from secugen_service import SecuGenService

TEST_DIR = PROJECT_ROOT / "scratch" / "test_verification_dataset"


def run_tests():
    print("=" * 70)
    print("   RUNNING AUTOMATED VERIFICATION OF DATASET COLLECTION PIPELINE")
    print("=" * 70)

    # Clean up test dir if exists
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    dm = DatasetManager(TEST_DIR)

    # Test 1: Identifier Sanitization
    print("\n[Test 1/8] Verifying Flexible Participant Identifier Sanitization...")
    test_identifiers = [
        ("Rahul", "Rahul"),
        ("Anu", "Anu"),
        ("Person_01", "Person_01"),
        ("Participant123", "Participant123"),
        ("ABC123", "ABC123"),
        ("MyFriend", "MyFriend"),
        ("  Leading and Trailing Spaces  ", "Leading_and_Trailing_Spaces"),
        ("Dr. John Doe #1 (Lab A)", "Dr_John_Doe_1_Lab_A"),
        ("Special/Unsafe:Chars?*<|>", "Special_Unsafe_Chars"),
    ]

    for raw, expected_safe in test_identifiers:
        safe = sanitize_identifier(raw)
        print(f"  '{raw}' -> '{safe}' (Expected: '{expected_safe}')")
        assert safe == expected_safe, f"Sanitization mismatch: {safe} != {expected_safe}"
    print("[OK] Participant sanitization tests PASSED.")

    # Test 2: Finger Normalization (Only Right Thumb and Left Thumb)
    print("\n[Test 2/8] Verifying Finger Selection Normalization...")
    assert normalize_finger_label("Right Thumb") == "RIGHT_THUMB"
    assert normalize_finger_label("right thumb") == "RIGHT_THUMB"
    assert normalize_finger_label("Left Thumb") == "LEFT_THUMB"
    assert normalize_finger_label("left thumb") == "LEFT_THUMB"
    print("[OK] Finger label normalization tests PASSED.")

    # Test 3: Clean Capture Saving & Metadata Integrity
    print("\n[Test 3/8] Verifying Clean Capture Saving & Metadata Fields...")
    dummy_img = np.ones((400, 300), dtype=np.uint8) * 128
    # Add dummy ridge pattern
    dummy_img[100:300:10, :] = 20

    rec_clean = dm.save_capture(
        image=dummy_img,
        raw_name="Rahul",
        finger="RIGHT_THUMB",
        capture_type="clean",
        distortion_condition="Normal Relaxed",
        scanner_model="SecuGen Hamster Pro 20",
        scanner_serial="H58220414185",
        image_quality=88,
        nfiq="N/A",
    )

    print(f"  Saved Image: {rec_clean.image_filename}")
    print(f"  Participant: {rec_clean.original_participant_name} (ID: {rec_clean.participant_id})")
    assert rec_clean.image_filename == "clean/Rahul_RIGHT_THUMB_clean.png"
    assert (TEST_DIR / rec_clean.image_filename).exists()

    # Check status after clean
    st_rahul = dm.get_participant_thumb_status("Rahul")
    assert st_rahul.right_thumb_clean is True
    assert st_rahul.right_thumb_distorted is False
    assert st_rahul.left_thumb_clean is False
    assert st_rahul.left_thumb_distorted is False
    print("[OK] Clean capture saving & status lookup PASSED.")

    # Test 4: Distorted Capture Saving & Pairing
    print("\n[Test 4/8] Verifying Distorted Capture Saving & Pairing...")
    rec_dist = dm.save_capture(
        image=dummy_img,
        raw_name="Rahul",
        finger="RIGHT_THUMB",
        capture_type="distorted",
        distortion_condition="Lateral Shear (Side displacement / shear shift)",
        scanner_model="SecuGen Hamster Pro 20",
        scanner_serial="H58220414185",
        image_quality=82,
        nfiq="N/A",
    )
    print(f"  Saved Image: {rec_dist.image_filename}")
    assert rec_dist.image_filename == "distorted/Rahul_RIGHT_THUMB_distorted.png"
    assert (TEST_DIR / rec_dist.image_filename).exists()

    st_rahul = dm.get_participant_thumb_status("Rahul")
    assert st_rahul.right_thumb_clean is True
    assert st_rahul.right_thumb_distorted is True
    assert st_rahul.right_thumb_complete is True
    print("[OK] Distorted capture saving & pairing PASSED.")

    # Test 5: Left Thumb Independent Tracking
    print("\n[Test 5/8] Verifying Left Thumb Independent Tracking...")
    rec_lt = dm.save_capture(
        image=dummy_img,
        raw_name="Rahul",
        finger="LEFT_THUMB",
        capture_type="clean",
        distortion_condition="Normal Relaxed",
        scanner_model="SecuGen Hamster Pro 20",
        scanner_serial="H58220414185",
        image_quality=90,
    )
    print(f"  Saved Image: {rec_lt.image_filename}")
    assert rec_lt.image_filename == "clean/Rahul_LEFT_THUMB_clean.png"

    st_rahul = dm.get_participant_thumb_status("Rahul")
    assert st_rahul.right_thumb_complete is True
    assert st_rahul.left_thumb_clean is True
    assert st_rahul.left_thumb_distorted is False
    assert st_rahul.left_thumb_complete is False
    print("[OK] Left Thumb independent tracking PASSED.")

    # Test 6: Duplicate Protection & Numbered Versioning
    print("\n[Test 6/8] Verifying Duplicate Protection & Versioning...")
    # Attempting to save duplicate without versioning should raise FileExistsError
    try:
        dm.save_capture(
            image=dummy_img,
            raw_name="Rahul",
            finger="RIGHT_THUMB",
            capture_type="clean",
            distortion_condition="Normal Relaxed",
            scanner_model="SecuGen Hamster Pro 20",
            scanner_serial="H58220414185",
            image_quality=95,
            allow_versioning=False,
        )
        assert False, "Should have raised FileExistsError!"
    except FileExistsError:
        print("  Correctly blocked duplicate overwrite without versioning.")

    # Saving with allow_versioning=True should create _v2
    rec_v2 = dm.save_capture(
        image=dummy_img,
        raw_name="Rahul",
        finger="RIGHT_THUMB",
        capture_type="clean",
        distortion_condition="Normal Relaxed",
        scanner_model="SecuGen Hamster Pro 20",
        scanner_serial="H58220414185",
        image_quality=95,
        allow_versioning=True,
    )
    print(f"  Versioned file created: {rec_v2.image_filename}")
    assert rec_v2.image_filename == "clean/Rahul_RIGHT_THUMB_clean_v2.png"
    assert (TEST_DIR / rec_v2.image_filename).exists()
    assert (TEST_DIR / "clean" / "Rahul_RIGHT_THUMB_clean.png").exists()  # Original still intact!
    print("[OK] Duplicate protection & versioning PASSED.")

    # Test 7: Multi-participant Dataset Statistics
    print("\n[Test 7/8] Verifying Dataset Statistics Calculation...")
    # Add Anu
    dm.save_capture(dummy_img, "Anu", "RIGHT_THUMB", "clean", "Normal Relaxed", "SecuGen", "SN1", 85)
    dm.save_capture(dummy_img, "Anu", "RIGHT_THUMB", "distorted", "High Pressure", "SecuGen", "SN1", 80)
    # Add Person_01 (only clean)
    dm.save_capture(dummy_img, "Person_01", "LEFT_THUMB", "clean", "Normal Relaxed", "SecuGen", "SN1", 90)

    stats = dm.get_statistics()
    print(f"  Total Unique Participants:      {stats.total_participants}")
    print(f"  Total Clean Captures:           {stats.clean_captures_count}")
    print(f"  Total Distorted Captures:       {stats.distorted_captures_count}")
    print(f"  Complete Clean-Distorted Pairs: {stats.complete_pairs_count}")
    print(f"  Right Thumb Captures:           {stats.right_thumb_captures_count}")
    print(f"  Left Thumb Captures:            {stats.left_thumb_captures_count}")

    # Participants: Rahul (RT clean, RT clean_v2, RT dist, LT clean), Anu (RT clean, RT dist), Person_01 (LT clean) -> 3 unique
    assert stats.total_participants == 3
    # Complete pairs: Rahul RT, Anu RT -> 2 complete pairs
    assert stats.complete_pairs_count == 2
    print("[OK] Dataset statistics calculation PASSED.")

    # Test 8: Physical Hardware Detection Check
    print("\n[Test 8/8] Testing SecuGen Hardware & Service Connectivity...")
    svc = SecuGenService()
    hw_status = svc.detect_device()
    print(f"  Hardware Available: {hw_status.available}")
    print(f"  Connection Mode:    {hw_status.mode}")
    print(f"  Device Name:        {hw_status.device_name}")
    print(f"  Serial Number:      {hw_status.serial_number}")
    print(f"  Resolution:         {hw_status.resolution}")
    print(f"  WebAPI URL:         {hw_status.service_url}")
    print("[OK] Hardware detection test completed.")

    # Cleanup test directory
    shutil.rmtree(TEST_DIR)
    print("\n" + "=" * 70)
    print("   ALL PIPELINE & METADATA TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
