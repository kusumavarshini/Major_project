#!/usr/bin/env python3
"""
FingerSure Automated Physical Fingerprint Dataset Collection System (10-Finger Architecture)
==========================================================================================
Captures paired CLEAN (0) and DISTORTED (1) fingerprints across all 10 fingers:
  Right Hand: Thumb, Index, Middle, Ring, Little (1-5)
  Left Hand:  Thumb, Index, Middle, Ring, Little (6-10)
Target: 20 images per participant (10 CLEAN, 10 DISTORTED).
Target Dataset Size: 42 participants x 20 images = 840 total images.
Supports arbitrary additional participants without code modifications.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

# Ensure project root is in sys.path
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
    Participant10FingerStatus,
    sanitize_identifier,
)
from secugen_service import CaptureResult, ScannerDeviceStatus, SecuGenService
import validate_dataset_collection


def print_banner(scanner_status: ScannerDeviceStatus):
    print("\n" + "=" * 74)
    print("      FINGERSURE - 10-FINGER BIOMETRIC DATASET COLLECTION SYSTEM")
    print("           Paired CLEAN & DISTORTED Physical Dataset Collector")
    print("=" * 74)
    if scanner_status.available:
        if scanner_status.mode == "native_hardware":
            print(f" [OK] Scanner: {scanner_status.device_name} (SN: {scanner_status.serial_number})")
            print(f"      Mode: Direct Native SDK Driver | Resolution: {scanner_status.resolution}")
        else:
            print(f" [OK] Scanner: {scanner_status.device_name} ({scanner_status.service_url})")
            print(f"      Mode: SecuGen Bio Server WebAPI | Resolution: 300x400 @ 500 DPI")
    else:
        print(" [!] SCANNER STATUS: Disconnected / Not Detected")
        print("     " + (scanner_status.error_message or "Ensure USB cable is connected or Bio Server is running."))
    print("=" * 74)


def prompt_participant_id(dataset_mgr: DatasetManager) -> Tuple[str, Participant10FingerStatus]:
    """
    Prompt operator for participant ID with automatic format suggestion (e.g. P001, P002).
    Accepts arbitrary IDs while enforcing clean filesystem naming.
    """
    suggested = dataset_mgr.get_next_participant_id()
    while True:
        try:
            raw = input(f"\nEnter Participant ID [Press ENTER for '{suggested}']: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nOperation cancelled.")
            sys.exit(0)

        if not raw:
            raw = suggested

        try:
            p_id = sanitize_identifier(raw).upper()
            if p_id.isdigit():
                p_id = f"P{int(p_id):03d}"
        except ValueError as e:
            print(f"[!] Invalid identifier: {e}")
            continue

        status = dataset_mgr.get_participant_10finger_status(p_id)
        return p_id, status


def display_participant_dashboard(p_id: str, status: Participant10FingerStatus):
    """Render comprehensive 10-finger status dashboard for a participant."""
    print("\n" + "=" * 74)
    print(f" PARTICIPANT: {p_id:<8} | CLEAN: {status.clean_count}/10 | DISTORTED: {status.distorted_count}/10 | TOTAL: {status.total_completed}/20")
    print("=" * 74)

    def format_status(done: bool, fname: Optional[str]) -> str:
        return "[DONE]" if done else "Pending"

    print(" RIGHT HAND:")
    for idx, (hand, finger) in enumerate(ALL_10_FINGERS[:5], 1):
        cap = status.captures.get((hand, finger), {})
        c_st = format_status(cap.get("clean"), cap.get("clean_file"))
        d_st = format_status(cap.get("distorted"), cap.get("distorted_file"))
        code = f"{hand} {finger.title()}".ljust(15)
        print(f"   {idx:2d}. {code}  | Clean: {c_st:<8} | Distorted: {d_st:<8} | Pair: {'[COMPLETE]' if c_st == '[DONE]' and d_st == '[DONE]' else 'Pending'}")

    print("\n LEFT HAND:")
    for idx, (hand, finger) in enumerate(ALL_10_FINGERS[5:], 6):
        cap = status.captures.get((hand, finger), {})
        c_st = format_status(cap.get("clean"), cap.get("clean_file"))
        d_st = format_status(cap.get("distorted"), cap.get("distorted_file"))
        code = f"{hand} {finger.title()}".ljust(15)
        print(f"   {idx:2d}. {code}  | Clean: {c_st:<8} | Distorted: {d_st:<8} | Pair: {'[COMPLETE]' if c_st == '[DONE]' and d_st == '[DONE]' else 'Pending'}")

    print("-" * 74)
    if status.fully_complete:
        print(" 🎉 STATUS: ALL 20 CAPTURES COMPLETED FOR THIS PARTICIPANT!")
    else:
        print(f" Progress: {status.total_completed}/20 completed ({status.remaining_count} remaining)")
    print("=" * 74)


def select_finger_and_condition(status: Participant10FingerStatus) -> Tuple[str, str, int]:
    """Allow operator to select a specific finger (1-10) and condition (Clean/Distorted)."""
    while True:
        print("\nSelect finger to capture (1-10):")
        print("  1. Right Thumb   2. Right Index   3. Right Middle   4. Right Ring   5. Right Little")
        print("  6. Left Thumb    7. Left Index    8. Left Middle    9. Left Ring   10. Left Little")

        try:
            choice = input("Enter finger number (1-10, or 'c' to cancel): ").strip()
        except (KeyboardInterrupt, EOFError):
            return "", "", -1

        if choice.lower() in ("c", "cancel", "q", "quit"):
            return "", "", -1

        if not choice.isdigit() or not (1 <= int(choice) <= 10):
            print("[!] Please enter a valid finger number between 1 and 10.")
            continue

        finger_idx = int(choice) - 1
        hand, finger = ALL_10_FINGERS[finger_idx]

        # Select condition
        while True:
            print(f"\nTarget Finger: {hand} {finger.title()}")
            print("Select Condition:")
            print("  1. CLEAN Fingerprint (Normal, relaxed, natural contact)")
            print("  2. DISTORTED Fingerprint (Safe structural elastic deformation / lateral shear)")
            try:
                c_ch = input("Enter condition (1 or 2, 'c' to cancel): ").strip()
            except (KeyboardInterrupt, EOFError):
                return "", "", -1

            if c_ch == "1":
                return hand, finger, CONDITION_CLEAN
            elif c_ch == "2":
                return hand, finger, CONDITION_DISTORTED
            elif c_ch.lower() in ("c", "cancel"):
                break
            else:
                print("[!] Please enter 1 for CLEAN or 2 for DISTORTED.")


def print_ascii_thumbnail(img: np.ndarray, width: int = 36, height: int = 14):
    """Render compact ASCII visual representation of the captured fingerprint ridges."""
    try:
        small = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
        print("\n  [Captured Ridge Pattern Preview]")
        print("  +" + "-" * width + "+")
        for row in small:
            line = "".join(chars[min(val // 26, len(chars) - 1)] for val in row)
            print(f"  |{line}|")
        print("  +" + "-" * width + "+")
    except Exception:
        pass


def execute_capture(
    secugen_svc: SecuGenService,
    participant_id: str,
    hand: str,
    finger: str,
    condition: int,
    allow_mock: bool = False,
    dataset_mgr: Optional[DatasetManager] = None,
) -> Optional[CaptureResult]:
    """Display instructions and trigger capture from SecuGen scanner."""
    cond_name = CONDITION_NAMES.get(condition, "CLEAN")
    finger_display = f"{hand} {finger.title()}"

    print("\n" + "=" * 60)
    print(f"   {cond_name} FINGERPRINT CAPTURE")
    print(f"   Participant: {participant_id} | Finger: {finger_display}")
    print("=" * 60)

    if condition == CONDITION_CLEAN:
        print("Instructions:")
        print(" • Clean and dry the finger.")
        print(" • Place finger naturally and flat on the glass prism.")
        print(" • Keep finger relaxed with gentle, even contact.")
        print(" • Ordinary tilt or rotation is acceptable; DO NOT force distortion.")
    else:
        print("Instructions:")
        print(" • Use the SAME finger as the clean capture.")
        print(" • Apply safe, natural lateral skin shear or slight non-rigid bending.")
        print(" • DO NOT twist forcefully or cause participant discomfort/injury.")
        print(" • If genuine structural distortion cannot be obtained safely, abort capture.")

    try:
        input("\nPress ENTER when ready to scan (or Ctrl+C to abort)... ")
    except (KeyboardInterrupt, EOFError):
        print("\nScan cancelled.")
        return None

    timeout_sec = 20
    print(f"\n📡 Sensor active (waiting up to {timeout_sec}s for finger contact)...")
    print("   👉 Red optical LED should be ON. Place finger firmly on glass prism.")

    while True:
        res = secugen_svc.capture(timeout_sec=timeout_sec, quality_min=30, allow_mock=allow_mock)

        if not res.success:
            print(f"\n[X] SCANNER CAPTURE FAILED: {res.error_message}")
            if dataset_mgr is not None:
                dataset_mgr.record_failed_capture(
                    participant_id=participant_id,
                    hand=hand,
                    finger=finger,
                    condition=condition,
                    error_message=res.error_message or "Scanner timeout / sensor failure",
                    scanner_model=res.scanner_model,
                    scanner_serial=res.serial_number,
                )

            print("\nTroubleshooting Tips:")
            print(" • Ensure finger covers the center of the glass prism flatly.")
            print(" • Apply firm, even pressure.")
            print(" • If finger skin is dry, gently rub against palm or breathe warm air on it.")
            print(" • Verify prism glass is clean.")

            try:
                retry = input("\nWould you like to try capturing again? [Y/n]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return None

            if retry in ("", "y", "yes"):
                print(f"\n📡 Sensor active (waiting up to {timeout_sec}s)... Hold finger firmly.")
                continue
            else:
                return None

        # Capture succeeded!
        print(f"\n[OK] Fingerprint captured successfully via {res.capture_method}!")
        print(f"     Image Dimensions: {res.width} x {res.height} pixels")
        print(f"     Reported Quality: {res.quality}/100")
        print(f"     Hardware Model:   {res.scanner_model} (SN: {res.serial_number})")

        if res.warning_message:
            print(f"     [!] Notice: {res.warning_message}")

        if res.is_mock:
            print("     [!] NOTE: MOCK SIMULATION MODE (No physical scanner used).")

        # Display ASCII ridge preview
        if res.image is not None:
            print_ascii_thumbnail(res.image)

        # Operator preview and confirmation
        try:
            accept = input("\nAccept and save this fingerprint? [Y/n]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            return None

        if accept in ("", "y", "yes"):
            return res
        else:
            print("Fingerprint capture rejected by operator.")
            if dataset_mgr is not None:
                dataset_mgr.record_failed_capture(
                    participant_id=participant_id,
                    hand=hand,
                    finger=finger,
                    condition=condition,
                    error_message="Rejected by operator preview",
                    scanner_model=res.scanner_model,
                    scanner_serial=res.serial_number,
                )

            try:
                recapture = input("Would you like to re-scan this finger? [Y/n]: ").strip().lower()
            except (KeyboardInterrupt, EOFError):
                return None
            if recapture in ("", "y", "yes"):
                print(f"\n📡 Sensor active (waiting up to {timeout_sec}s)...")
                continue
            return None


def run_guided_collection_session(
    p_id: str,
    status: Participant10FingerStatus,
    dataset_mgr: DatasetManager,
    secugen_svc: SecuGenService,
    allow_mock: bool = False,
):
    """
    Guided collection workflow:
    Iterates systematically through all 10 fingers.
    Captures CLEAN first, then DISTORTED for the same finger.
    Allows resuming at any point.
    """
    print("\n" + "=" * 70)
    print(f"   STARTING GUIDED COLLECTION SESSION FOR PARTICIPANT: {p_id}")
    print("=" * 70)

    for idx, (hand, finger) in enumerate(ALL_10_FINGERS, 1):
        finger_name = f"{hand} {finger.title()}"

        # 1. Clean capture if pending
        if not status.is_clean_done(hand, finger):
            print(f"\n>>> Step {idx}A: Capturing CLEAN impression for {finger_name}")
            cap_res = execute_capture(
                secugen_svc=secugen_svc,
                participant_id=p_id,
                hand=hand,
                finger=finger,
                condition=CONDITION_CLEAN,
                allow_mock=allow_mock,
                dataset_mgr=dataset_mgr,
            )
            if cap_res and cap_res.success:
                try:
                    rec = dataset_mgr.save_capture(
                        image=cap_res.image,
                        participant_id=p_id,
                        hand=hand,
                        finger=finger,
                        condition=CONDITION_CLEAN,
                        scanner_model=cap_res.scanner_model,
                        scanner_serial=cap_res.serial_number,
                        image_quality=cap_res.quality,
                        allow_versioning=True,
                    )
                    status = dataset_mgr.get_participant_10finger_status(p_id)
                    print(f"[SAVED] {rec.filename} -> FingerSure_Dataset/{rec.image_path}")
                except Exception as e:
                    print(f"[!] Error saving capture: {e}")
            else:
                print(f"[!] Clean capture for {finger_name} skipped/incomplete.")
                try:
                    cont = input("Continue to next step? [Y/n]: ").strip().lower()
                    if cont in ("n", "no"):
                        return
                except (KeyboardInterrupt, EOFError):
                    return

        # 2. Distorted capture if pending
        if not status.is_distorted_done(hand, finger):
            print(f"\n>>> Step {idx}B: Capturing DISTORTED impression for {finger_name}")
            cap_res = execute_capture(
                secugen_svc=secugen_svc,
                participant_id=p_id,
                hand=hand,
                finger=finger,
                condition=CONDITION_DISTORTED,
                allow_mock=allow_mock,
                dataset_mgr=dataset_mgr,
            )
            if cap_res and cap_res.success:
                try:
                    rec = dataset_mgr.save_capture(
                        image=cap_res.image,
                        participant_id=p_id,
                        hand=hand,
                        finger=finger,
                        condition=CONDITION_DISTORTED,
                        scanner_model=cap_res.scanner_model,
                        scanner_serial=cap_res.serial_number,
                        image_quality=cap_res.quality,
                        distortion_condition="Structural Elastic Deformation",
                        allow_versioning=True,
                    )
                    status = dataset_mgr.get_participant_10finger_status(p_id)
                    print(f"[SAVED] {rec.filename} -> FingerSure_Dataset/{rec.image_path}")
                except Exception as e:
                    print(f"[!] Error saving capture: {e}")
            else:
                print(f"[!] Distorted capture for {finger_name} skipped/incomplete.")
                try:
                    cont = input("Continue to next step? [Y/n]: ").strip().lower()
                    if cont in ("n", "no"):
                        return
                except (KeyboardInterrupt, EOFError):
                    return

        # Show mini progress
        print(f"\n[PROGRESS] Participant {p_id}: {status.total_completed}/20 captures completed ({status.remaining_count} remaining).")


def main():
    """Main terminal dataset collector entrypoint."""
    dataset_mgr = DatasetManager(PROJECT_ROOT / DEFAULT_DATASET_DIR_NAME)
    secugen_svc = SecuGenService()

    scanner_status = secugen_svc.detect_device()
    print_banner(scanner_status)

    allow_mock = False
    if not scanner_status.available:
        print("\n[?] Scanner is not connected.")
        print("    1. Retry hardware detection")
        print("    2. Enable offline mock simulator (for testing/validation)")
        print("    3. Exit")
        try:
            opt = input("Select option (1-3): ").strip()
            if opt == "1":
                scanner_status = secugen_svc.detect_device()
                print_banner(scanner_status)
            elif opt == "2":
                allow_mock = True
                print("[!] Mock simulator enabled for offline testing.")
            else:
                sys.exit(0)
        except (KeyboardInterrupt, EOFError):
            sys.exit(0)

    # Main operational loop
    while True:
        print("\n" + "=" * 50)
        print("             MAIN SELECTION MENU")
        print("=" * 50)
        print(" 1. Collect / Resume Participant (10 Fingers x 2 Conditions)")
        print(" 2. View Dataset Progress Dashboard")
        print(" 3. Generate & View Dataset Validation Report")
        print(" 4. Exit")
        print("=" * 50)

        try:
            menu_choice = input("Enter choice (1-4): ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Thank you!")
            break

        if menu_choice == "2":
            stats = dataset_mgr.get_statistics()
            print("\n" + "=" * 60)
            print("          FINGERSURE DATASET COLLECTION SUMMARY")
            print("=" * 60)
            print(f" Total Unique Participants:       {stats.total_participants}")
            print(f" Complete Participants (20/20):   {stats.complete_participants_count} / {stats.target_participants} target")
            print(f" Incomplete Participants:         {stats.incomplete_participants_count}")
            print(f" Total CLEAN Captures:            {stats.clean_captures_count} / {stats.target_clean_images} target")
            print(f" Total DISTORTED Captures:        {stats.distorted_captures_count} / {stats.target_distorted_images} target")
            print(f" Total Physical Images:           {stats.total_images_count} / {stats.target_total_images} target ({stats.total_images_count/stats.target_total_images*100:.1f}%)")
            print("=" * 60)
            input("\nPress ENTER to return to menu...")
            continue

        elif menu_choice == "3":
            print("\nGenerating Dataset Validation Report...")
            validate_dataset_collection.run_dataset_validation()
            input("\nPress ENTER to return to menu...")
            continue

        elif menu_choice == "4":
            print("\nExiting FingerSure Dataset Collector. Goodbye!")
            break

        elif menu_choice != "1":
            print("[!] Invalid option. Please enter 1, 2, 3, or 4.")
            continue

        # Option 1: Collect / Resume Participant
        p_id, status = prompt_participant_id(dataset_mgr)
        display_participant_dashboard(p_id, status)

        # Completion check
        if status.fully_complete:
            print(f"\n[NOTE] Participant {p_id} already has all 20 captures complete.")
            try:
                again = input("Do you want to recapture any finger? [y/N]: ").strip().lower()
                if again not in ("y", "yes"):
                    continue
            except (KeyboardInterrupt, EOFError):
                continue

        # Participant capture sub-menu
        while True:
            display_participant_dashboard(p_id, status)
            print("\nParticipant Capture Options:")
            print(" 1. Start / Resume Guided Full Capture (All 10 fingers sequentially)")
            print(" 2. Selectively Capture or Recapture a Specific Finger")
            print(" 3. Return to Main Menu")

            try:
                p_opt = input("\nEnter option (1-3): ").strip()
            except (KeyboardInterrupt, EOFError):
                break

            if p_opt == "1":
                run_guided_collection_session(
                    p_id=p_id,
                    status=status,
                    dataset_mgr=dataset_mgr,
                    secugen_svc=secugen_svc,
                    allow_mock=allow_mock,
                )
                status = dataset_mgr.get_participant_10finger_status(p_id)

            elif p_opt == "2":
                hand, finger, cond = select_finger_and_condition(status)
                if not hand or cond == -1:
                    continue

                cond_name = CONDITION_NAMES.get(cond, "CLEAN")
                is_done = status.is_clean_done(hand, finger) if cond == CONDITION_CLEAN else status.is_distorted_done(hand, finger)

                allow_versioning = False
                if is_done:
                    print(f"\n[!] A {cond_name} capture already exists for {p_id} - {hand} {finger}.")
                    print("  1. Cancel")
                    print("  2. Replace / Recapture with a new version")
                    try:
                        ov = input("Enter choice (1 or 2): ").strip()
                        if ov == "2":
                            allow_versioning = True
                        else:
                            print("Capture cancelled.")
                            continue
                    except (KeyboardInterrupt, EOFError):
                        continue

                cap_res = execute_capture(
                    secugen_svc=secugen_svc,
                    participant_id=p_id,
                    hand=hand,
                    finger=finger,
                    condition=cond,
                    allow_mock=allow_mock,
                    dataset_mgr=dataset_mgr,
                )

                if cap_res and cap_res.success:
                    try:
                        rec = dataset_mgr.save_capture(
                            image=cap_res.image,
                            participant_id=p_id,
                            hand=hand,
                            finger=finger,
                            condition=cond,
                            scanner_model=cap_res.scanner_model,
                            scanner_serial=cap_res.serial_number,
                            image_quality=cap_res.quality,
                            distortion_condition="Structural Elastic Deformation" if cond == CONDITION_DISTORTED else "Normal Relaxed",
                            allow_versioning=allow_versioning,
                        )
                        print(f"\n[SAVED] {rec.filename} -> FingerSure_Dataset/{rec.image_path}")
                        status = dataset_mgr.get_participant_10finger_status(p_id)
                    except Exception as e:
                        print(f"[!] Error saving capture: {e}")

            elif p_opt == "3":
                break

            # Check if participant just finished all 20 captures
            if status.fully_complete:
                print("\n" + "=" * 74)
                print(f" 🎉 CONGRATULATIONS! ALL 20 CAPTURES COMPLETED FOR PARTICIPANT {p_id}!")
                print(f"    - 10 CLEAN captures verified")
                print(f"    - 10 DISTORTED captures verified")
                print("=" * 74)
                input("\nPress ENTER to continue...")
                break


if __name__ == "__main__":
    main()
