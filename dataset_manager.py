"""
FingerSure Dataset Manager (10-Finger Extended Architecture)
============================================================
Handles filesystem organization, standardized filename generation,
10-finger metadata tracking (CLEAN=0, DISTORTED=1), duplicate protection,
participant capture progress, and dual-layer persistence (metadata.csv + MySQL).
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

# Standard dataset directory structure
DEFAULT_DATASET_DIR_NAME = "FingerSure_Dataset"
METADATA_FILENAME = "metadata.csv"
FAILED_CAPTURES_FILENAME = "failed_captures.csv"

# Hands and Fingers definitions
HANDS = ["RIGHT", "LEFT"]
FINGER_NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"]

ALL_10_FINGERS: List[Tuple[str, str]] = [
    ("RIGHT", "THUMB"),
    ("RIGHT", "INDEX"),
    ("RIGHT", "MIDDLE"),
    ("RIGHT", "RING"),
    ("RIGHT", "LITTLE"),
    ("LEFT", "THUMB"),
    ("LEFT", "INDEX"),
    ("LEFT", "MIDDLE"),
    ("LEFT", "RING"),
    ("LEFT", "LITTLE"),
]

# Finger codes e.g. "RIGHT_THUMB", "RIGHT_INDEX", etc.
ALL_10_FINGER_CODES = [f"{hand}_{finger}" for hand, finger in ALL_10_FINGERS]

# Binary Condition Constants
CONDITION_CLEAN = 0
CONDITION_DISTORTED = 1
CONDITION_NAMES = {0: "CLEAN", 1: "DISTORTED"}
CONDITION_BY_NAME = {"clean": 0, "distorted": 1, "0": 0, "1": 1}

# Metadata Fields (exactly matching specification)
METADATA_FIELDNAMES = [
    "participant_id",
    "hand",
    "finger",
    "condition",
    "filename",
    "image_path",
    "capture_timestamp",
    "image_width",
    "image_height",
    "capture_status",
    # Backward-compatible metadata
    "original_participant_name",
    "scanner_model",
    "scanner_serial_number",
    "image_quality",
    "nfiq",
    "distortion_condition",
]

FAILED_CAPTURES_FIELDNAMES = [
    "participant_id",
    "hand",
    "finger",
    "condition",
    "error_message",
    "capture_timestamp",
    "scanner_model",
    "scanner_serial_number",
]


@dataclass
class CaptureRecord:
    participant_id: str
    hand: str                          # "RIGHT" or "LEFT"
    finger: str                        # "THUMB", "INDEX", "MIDDLE", "RING", "LITTLE"
    condition: int                     # 0 for CLEAN, 1 for DISTORTED
    filename: str                      # e.g. "P001_RIGHT_THUMB_CLEAN.png"
    image_path: str                    # e.g. "clean/P001_RIGHT_THUMB_CLEAN.png"
    capture_timestamp: str             # ISO-8601 format
    image_width: int
    image_height: int
    capture_status: str = "accepted"
    original_participant_name: str = ""
    scanner_model: str = "SecuGen Hamster Pro 20"
    scanner_serial_number: str = "Unknown"
    image_quality: str = "N/A"
    nfiq: str = "N/A"
    distortion_condition: str = "Normal Relaxed"


@dataclass
class Participant10FingerStatus:
    participant_id: str
    original_name: str = ""
    captures: Dict[Tuple[str, str], Dict[str, Any]] = None

    def __post_init__(self):
        if self.captures is None:
            self.captures = {}
            for hand, finger in ALL_10_FINGERS:
                self.captures[(hand, finger)] = {
                    "clean": False,
                    "distorted": False,
                    "clean_file": None,
                    "distorted_file": None,
                }

    @property
    def clean_count(self) -> int:
        return sum(1 for v in self.captures.values() if v["clean"])

    @property
    def distorted_count(self) -> int:
        return sum(1 for v in self.captures.values() if v["distorted"])

    @property
    def total_completed(self) -> int:
        return self.clean_count + self.distorted_count

    @property
    def remaining_count(self) -> int:
        return 20 - self.total_completed

    @property
    def fully_complete(self) -> bool:
        return self.total_completed == 20

    def is_finger_complete(self, hand: str, finger: str) -> bool:
        h = hand.upper().strip()
        f = finger.upper().strip()
        entry = self.captures.get((h, f), {})
        return bool(entry.get("clean") and entry.get("distorted"))

    def is_clean_done(self, hand: str, finger: str) -> bool:
        h = hand.upper().strip()
        f = finger.upper().strip()
        return bool(self.captures.get((h, f), {}).get("clean"))

    def is_distorted_done(self, hand: str, finger: str) -> bool:
        h = hand.upper().strip()
        f = finger.upper().strip()
        return bool(self.captures.get((h, f), {}).get("distorted"))


@dataclass
class Dataset10FingerStatistics:
    total_participants: int
    clean_captures_count: int
    distorted_captures_count: int
    total_images_count: int
    complete_participants_count: int
    incomplete_participants_count: int
    target_participants: int = 42
    target_clean_images: int = 420
    target_distorted_images: int = 420
    target_total_images: int = 840
    per_finger_counts: Dict[str, Dict[str, int]] = None


def sanitize_identifier(raw_name: str) -> str:
    """
    Generate a safe filesystem identifier while preserving alphanumeric characters,
    underscores, and hyphens. Formats e.g. "P001", "Rahul", etc.
    """
    trimmed = raw_name.strip()
    if not trimmed:
        raise ValueError("Participant Name / ID cannot be empty.")
    safe = re.sub(r"[^\w\-]", "_", trimmed)
    safe = re.sub(r"_+", "_", safe)
    safe = safe.strip("_")
    return safe or "Participant"


def parse_hand_and_finger(finger_str: str) -> Tuple[str, str]:
    """
    Parse hand and finger from input string.
    Supports formats like:
      - "RIGHT_THUMB" -> ("RIGHT", "THUMB")
      - "left index" -> ("LEFT", "INDEX")
      - "thumb" -> ("RIGHT", "THUMB") [defaults to RIGHT]
    """
    norm = finger_str.strip().upper()
    hand = "RIGHT" if "RIGHT" in norm else ("LEFT" if "LEFT" in norm else "RIGHT")

    finger = "THUMB"
    for f in ["INDEX", "MIDDLE", "RING", "LITTLE", "THUMB"]:
        if f in norm:
            finger = f
            break

    return hand, finger


class DatasetManager:
    """Manages the FingerSure_Dataset directory, metadata, and MySQL synchronization."""

    def __init__(self, base_dir: Path | str | None = None):
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent / DEFAULT_DATASET_DIR_NAME
        else:
            base_dir = Path(base_dir)

        self.dataset_dir = base_dir
        self.clean_dir = self.dataset_dir / "clean"
        self.distorted_dir = self.dataset_dir / "distorted"
        self.metadata_path = self.dataset_dir / METADATA_FILENAME
        self.failed_path = self.dataset_dir / FAILED_CAPTURES_FILENAME

        self._ensure_directories()
        self._ensure_metadata_file()
        self._ensure_failed_file()

        # Attempt to initialize MySQL tables non-destructively
        try:
            import database
            database.init_physical_collection_db()
        except Exception:
            pass

    def _ensure_directories(self) -> None:
        """Create dataset directories if they do not exist."""
        self.clean_dir.mkdir(parents=True, exist_ok=True)
        self.distorted_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_metadata_file(self) -> None:
        """Create metadata.csv with headers if it does not exist."""
        if not self.metadata_path.exists():
            with open(self.metadata_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(METADATA_FIELDNAMES)

    def _ensure_failed_file(self) -> None:
        """Create failed_captures.csv with headers if it does not exist."""
        if not self.failed_path.exists():
            with open(self.failed_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(FAILED_CAPTURES_FIELDNAMES)

    def load_all_records(self) -> List[CaptureRecord]:
        """Read and parse all rows from metadata.csv, with legacy format backward compatibility."""
        records: List[CaptureRecord] = []
        if not self.metadata_path.exists():
            return records

        with open(self.metadata_path, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    p_id = row.get("participant_id", "").strip()
                    orig_name = row.get("original_participant_name", "").strip() or p_id

                    # Check if hand/finger are separated or unified
                    hand = row.get("hand", "").strip().upper()
                    finger = row.get("finger", "").strip().upper()

                    if not hand:
                        hand, finger = parse_hand_and_finger(finger)

                    # Condition: support int 0/1 or string clean/distorted
                    cond_raw = str(row.get("condition", "")).strip().lower()
                    if cond_raw in ("0", "clean"):
                        cond = CONDITION_CLEAN
                    elif cond_raw in ("1", "distorted"):
                        cond = CONDITION_DISTORTED
                    else:
                        c_type = row.get("capture_type", "").strip().lower()
                        cond = CONDITION_CLEAN if c_type == "clean" else CONDITION_DISTORTED

                    fname = row.get("filename", "").strip()
                    ipath = row.get("image_path", "").strip()
                    legacy_img = row.get("image_filename", "").strip()
                    if not fname and not ipath and legacy_img:
                        fname = Path(legacy_img).name
                        ipath = legacy_img
                    elif not fname and ipath:
                        fname = Path(ipath).name
                    elif not ipath and fname:
                        sub = "clean" if cond == CONDITION_CLEAN else "distorted"
                        ipath = f"{sub}/{fname}"

                    records.append(
                        CaptureRecord(
                            participant_id=p_id,
                            hand=hand,
                            finger=finger,
                            condition=cond,
                            filename=fname,
                            image_path=ipath,
                            capture_timestamp=row.get("capture_timestamp", "").strip(),
                            image_width=int(row.get("image_width", 0) or 0),
                            image_height=int(row.get("image_height", 0) or 0),
                            capture_status=row.get("capture_status", "accepted").strip(),
                            original_participant_name=orig_name,
                            scanner_model=row.get("scanner_model", "SecuGen Hamster Pro 20").strip(),
                            scanner_serial_number=row.get("scanner_serial_number", "Unknown").strip(),
                            image_quality=str(row.get("image_quality", "N/A")).strip(),
                            nfiq=str(row.get("nfiq", "N/A")).strip(),
                            distortion_condition=row.get("distortion_condition", "Normal Relaxed").strip(),
                        )
                    )
                except Exception:
                    continue
        return records

    def get_participant_10finger_status(self, raw_id: str) -> Participant10FingerStatus:
        """
        Retrieve capture status for all ten fingers (CLEAN and DISTORTED) for a participant.
        """
        orig_name = raw_id.strip()
        safe_id = sanitize_identifier(orig_name) if orig_name else ""

        status = Participant10FingerStatus(participant_id=safe_id, original_name=orig_name)
        if not safe_id:
            return status

        records = self.load_all_records()
        for rec in records:
            if rec.capture_status != "accepted":
                continue

            # Match participant by ID or original name (case-insensitive)
            if rec.participant_id.lower() == safe_id.lower() or rec.original_participant_name.lower() == orig_name.lower():
                hand, finger = rec.hand.upper(), rec.finger.upper()
                if (hand, finger) in status.captures:
                    if rec.condition == CONDITION_CLEAN:
                        status.captures[(hand, finger)]["clean"] = True
                        status.captures[(hand, finger)]["clean_file"] = rec.filename
                    elif rec.condition == CONDITION_DISTORTED:
                        status.captures[(hand, finger)]["distorted"] = True
                        status.captures[(hand, finger)]["distorted_file"] = rec.filename

        return status

    def get_existing_participant_ids(self) -> List[str]:
        """Return sorted list of all unique participant IDs currently in metadata."""
        records = self.load_all_records()
        seen = set()
        for r in records:
            if r.participant_id:
                seen.add(r.participant_id)
        return sorted(list(seen))

    def get_next_participant_id(self) -> str:
        """
        Detect existing sequential participant IDs (e.g. P001, P002, ...)
        and return the next recommended ID formatted as P{:03d}.
        """
        existing = self.get_existing_participant_ids()
        numbers = []
        for pid in existing:
            m = re.match(r"^P(\d+)$", pid, re.IGNORECASE)
            if m:
                numbers.append(int(m.group(1)))

        if not numbers:
            return "P001"
        next_num = max(numbers) + 1
        return f"P{next_num:03d}"

    def determine_target_filename(
        self,
        participant_id: str,
        hand: str,
        finger: str,
        condition: int,
        allow_versioning: bool = False,
    ) -> Tuple[Path, str, str]:
        """
        Determine target file path, relative image path, and filename.
        Strict naming convention:
          CLEAN:     {participant_id}_{HAND}_{FINGER}_CLEAN.png
          DISTORTED: {participant_id}_{HAND}_{FINGER}_DISTORTED.png
        Organized under:
          FingerSure_Dataset/clean/
          FingerSure_Dataset/distorted/
        """
        p_id = sanitize_identifier(participant_id).upper()
        h = hand.upper().strip()
        f = finger.upper().strip()
        cond_str = CONDITION_NAMES.get(condition, "CLEAN")

        sub_dir = self.clean_dir if condition == CONDITION_CLEAN else self.distorted_dir
        base_name = f"{p_id}_{h}_{f}_{cond_str}"
        target_file = sub_dir / f"{base_name}.png"

        if target_file.exists():
            if not allow_versioning:
                rel_path = f"{cond_str.lower()}/{target_file.name}"
                return target_file, rel_path, target_file.name

            # Versioning fallback (_v2, _v3, etc.)
            v = 2
            while True:
                v_file = sub_dir / f"{base_name}_v{v}.png"
                if not v_file.exists():
                    target_file = v_file
                    break
                v += 1

        rel_path = f"{cond_str.lower()}/{target_file.name}"
        return target_file, rel_path, target_file.name

    def save_capture(
        self,
        image: np.ndarray,
        participant_id: str,
        hand: str,
        finger: str,
        condition: int,
        scanner_model: str = "SecuGen Hamster Pro 20",
        scanner_serial: str = "Unknown",
        image_quality: str | int = "N/A",
        nfiq: str | int = "N/A",
        distortion_condition: str = "Normal Relaxed",
        allow_versioning: bool = False,
        original_participant_name: str = "",
    ) -> CaptureRecord:
        """
        Save the captured fingerprint image without loss to disk,
        append metadata to metadata.csv, and sync with MySQL.
        """
        if image is None or image.size == 0:
            raise ValueError("Cannot save empty or corrupted image.")

        p_id = sanitize_identifier(participant_id).upper()
        h = hand.upper().strip()
        f = finger.upper().strip()
        assert (h, f) in ALL_10_FINGERS, f"Invalid hand/finger combination: {h} {f}"
        assert condition in (CONDITION_CLEAN, CONDITION_DISTORTED), f"Invalid condition: {condition}"

        target_path, rel_path, fname = self.determine_target_filename(
            participant_id=p_id,
            hand=h,
            finger=f,
            condition=condition,
            allow_versioning=allow_versioning,
        )

        if target_path.exists() and not allow_versioning:
            raise FileExistsError(
                f"Capture image already exists: {target_path.name}. "
                "Recapturing must be confirmed to replace or version."
            )

        # Write 8-bit uncompressed lossless grayscale PNG
        success = cv2.imwrite(str(target_path), image)
        if not success:
            from PIL import Image
            pil_img = Image.fromarray(image)
            pil_img.save(str(target_path), format="PNG")

        h_px, w_px = image.shape[:2]
        timestamp = datetime.now().isoformat(timespec="seconds")

        if condition == CONDITION_CLEAN:
            cond_desc = "Normal Relaxed"
        else:
            cond_desc = distortion_condition.strip() or "Structural Elastic Deformation"

        record = CaptureRecord(
            participant_id=p_id,
            hand=h,
            finger=f,
            condition=condition,
            filename=fname,
            image_path=rel_path,
            capture_timestamp=timestamp,
            image_width=int(w_px),
            image_height=int(h_px),
            capture_status="accepted",
            original_participant_name=original_participant_name or p_id,
            scanner_model=scanner_model or "SecuGen Hamster Pro 20",
            scanner_serial_number=scanner_serial or "Unknown",
            image_quality=str(image_quality) if image_quality not in (None, "") else "N/A",
            nfiq=str(nfiq) if nfiq not in (None, "") else "N/A",
            distortion_condition=cond_desc,
        )

        # 1. Append to CSV
        with open(self.metadata_path, mode="a", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=METADATA_FIELDNAMES)
            writer.writerow(asdict(record))

        # 2. Sync to MySQL database
        try:
            import database
            db_record = {
                "participant_id": p_id,
                "hand": h,
                "finger": f,
                "condition": condition,
                "filename": fname,
                "image_path": rel_path,
                "capture_timestamp": timestamp.replace("T", " "),
                "image_width": int(w_px),
                "image_height": int(h_px),
                "capture_status": "accepted",
                "scanner_model": scanner_model,
                "scanner_serial": scanner_serial,
                "image_quality": int(image_quality) if str(image_quality).isdigit() else 0,
            }
            database.save_physical_capture_to_db(db_record)
        except Exception as e:
            print(f"[DB Notice] MySQL save skipped: {e}")

        return record

    def record_failed_capture(
        self,
        participant_id: str,
        hand: str,
        finger: str,
        condition: int,
        error_message: str,
        scanner_model: str = "SecuGen Hamster Pro 20",
        scanner_serial: str = "Unknown",
    ) -> None:
        """Log a failed or rejected capture attempt in failed_captures.csv and MySQL."""
        timestamp = datetime.now().isoformat(timespec="seconds")
        p_id = sanitize_identifier(participant_id).upper()
        h = hand.upper().strip()
        f = finger.upper().strip()

        row = {
            "participant_id": p_id,
            "hand": h,
            "finger": f,
            "condition": condition,
            "error_message": error_message,
            "capture_timestamp": timestamp,
            "scanner_model": scanner_model,
            "scanner_serial_number": scanner_serial,
        }

        # 1. Append to failed_captures.csv
        with open(self.failed_path, mode="a", newline="", encoding="utf-8") as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=FAILED_CAPTURES_FIELDNAMES)
            writer.writerow(row)

        # 2. Sync to MySQL failed captures table
        try:
            import database
            fail_db = {
                "participant_id": p_id,
                "hand": h,
                "finger": f,
                "condition": condition,
                "error_message": error_message,
                "scanner_model": scanner_model,
                "scanner_serial": scanner_serial,
            }
            database.record_failed_physical_capture_to_db(fail_db)
        except Exception:
            pass

    def get_statistics(self) -> Dataset10FingerStatistics:
        """Compute full 10-finger dataset progress statistics against 42-participant target."""
        records = [r for r in self.load_all_records() if r.capture_status == "accepted"]

        participants: Dict[str, Participant10FingerStatus] = {}
        for r in records:
            p_id = r.participant_id
            if p_id not in participants:
                participants[p_id] = Participant10FingerStatus(participant_id=p_id)

            hand, finger = r.hand.upper(), r.finger.upper()
            if (hand, finger) in participants[p_id].captures:
                if r.condition == CONDITION_CLEAN:
                    participants[p_id].captures[(hand, finger)]["clean"] = True
                elif r.condition == CONDITION_DISTORTED:
                    participants[p_id].captures[(hand, finger)]["distorted"] = True

        clean_total = sum(p.clean_count for p in participants.values())
        dist_total = sum(p.distorted_count for p in participants.values())
        total_images = clean_total + dist_total

        complete_parts = sum(1 for p in participants.values() if p.fully_complete)
        incomplete_parts = len(participants) - complete_parts

        # Per finger counts
        per_finger: Dict[str, Dict[str, int]] = {}
        for h, f in ALL_10_FINGERS:
            code = f"{h}_{f}"
            c_cnt = 0
            d_cnt = 0
            for p in participants.values():
                if p.captures.get((h, f), {}).get("clean"):
                    c_cnt += 1
                if p.captures.get((h, f), {}).get("distorted"):
                    d_cnt += 1
            per_finger[code] = {"clean": c_cnt, "distorted": d_cnt, "total": c_cnt + d_cnt}

        return Dataset10FingerStatistics(
            total_participants=len(participants),
            clean_captures_count=clean_total,
            distorted_captures_count=dist_total,
            total_images_count=total_images,
            complete_participants_count=complete_parts,
            incomplete_participants_count=incomplete_parts,
            per_finger_counts=per_finger,
        )
