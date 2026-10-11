
"""
MySQL repository for the existing SourceAFIS gallery and new enrollments.
The existing 317 gallery templates remain in external_fvc_clean_gallery.
New enrollments are stored separately in enrolled_fingerprints.
"""

import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import mysql.connector
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

_GALLERY_CACHE: Optional[List[Dict[str, Any]]] = None
_LAST_CACHE_TIME = 0.0
CACHE_TTL_SECONDS = 300.0


def _connect():
    """Connect to the existing gallery database."""
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database="newdb",
        charset="utf8mb4",
        use_pure=True,
    )


def get_person_templates(
    subject_id: str,
    mysql_password: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve all templates enrolled for a subject."""
    conn = _connect()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT enrollment_id, subject_id, filename, image_path,
                   template_data, minutiae_count, image_width, image_height
            FROM enrolled_fingerprints
            WHERE subject_id = %s
            ORDER BY enrollment_id
            """,
            (subject_id,),
        )
        rows = cursor.fetchall()
    finally:
        cursor.close()
        conn.close()

    templates = []
    for row in rows:
        try:
            tpl = pickle.loads(row["template_data"])
        except Exception:
            continue

        templates.append({
            "fingerprint_id": row["enrollment_id"],
            "filename": row["filename"],
            "image_path": row["image_path"],
            "template": tpl,
            "minutiae_count": row["minutiae_count"],
            "image_width": row["image_width"],
            "image_height": row["image_height"],
        })

    if not templates:
        raise ValueError(
            f"No enrolled fingerprint templates found for subject: {subject_id}"
        )

    return {
        "subject_id": subject_id,
        "family_id": None,
        "member_type": "ENROLLED",
        "templates": templates,
    }


def get_all_gallery_templates(
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Load the original gallery and new enrollments for 1:N identification."""
    global _GALLERY_CACHE, _LAST_CACHE_TIME

    now = time.time()
    if (
        not force_refresh
        and _GALLERY_CACHE is not None
        and now - _LAST_CACHE_TIME < CACHE_TTL_SECONDS
    ):
        return _GALLERY_CACHE

    conn = _connect()
    cursor = conn.cursor(dictionary=True)
    gallery: List[Dict[str, Any]] = []

    try:
        # Preserve and read all original gallery templates.
        cursor.execute(
            """
            SELECT fingerprint_id, template_data, minutiae_count,
                   template_version, generation_status
            FROM external_fvc_clean_gallery
            WHERE generation_status = 'generated'
            """
        )

        for row in cursor.fetchall():
            try:
                tpl = pickle.loads(row["template_data"])
            except Exception:
                continue

            if not isinstance(tpl, dict) or len(tpl.get("minutiae", [])) < 2:
                continue

            gallery.append({
                "fingerprint_id": row["fingerprint_id"],
                "subject_id": f"GALLERY-{row['fingerprint_id']}",
                "family_id": None,
                "member_type": "EXISTING_GALLERY",
                "filename": f"gallery_{row['fingerprint_id']}",
                "image_path": None,
                "minutiae_count": row["minutiae_count"],
                "template": tpl,
            })

        # Add newly enrolled templates.
        cursor.execute(
            """
            SELECT enrollment_id, subject_id, filename, image_path,
                   template_data, minutiae_count, image_width, image_height
            FROM enrolled_fingerprints
            ORDER BY enrollment_id
            """
        )

        for row in cursor.fetchall():
            try:
                tpl = pickle.loads(row["template_data"])
            except Exception:
                continue

            if not isinstance(tpl, dict) or len(tpl.get("minutiae", [])) < 2:
                continue

            gallery.append({
                "fingerprint_id": f"ENROLLED-{row['enrollment_id']}",
                "subject_id": row["subject_id"],
                "family_id": None,
                "member_type": "ENROLLED",
                "filename": row["filename"],
                "image_path": row["image_path"],
                "minutiae_count": row["minutiae_count"],
                "image_width": row["image_width"],
                "image_height": row["image_height"],
                "template": tpl,
            })

    finally:
        cursor.close()
        conn.close()

    _GALLERY_CACHE = gallery
    _LAST_CACHE_TIME = time.time()
    return gallery


def clear_gallery_cache():
    """Force the next 1:N search to reload the database gallery."""
    global _GALLERY_CACHE, _LAST_CACHE_TIME
    _GALLERY_CACHE = None
    _LAST_CACHE_TIME = 0.0


def enroll_fingerprint(
    subject_id: str,
    filename: str,
    image_path: str,
    image_width: int,
    image_height: int,
    template: Dict[str, Any],
) -> int:
    """Save a new SourceAFIS template without modifying the original gallery."""
    if not subject_id.strip():
        raise ValueError("Subject ID cannot be empty.")

    minutiae_count = int(template.get("minutiae_count", 0))
    if minutiae_count < 2:
        raise ValueError("Not enough minutiae detected. Please capture again.")

    serialized = pickle.dumps(template, protocol=pickle.HIGHEST_PROTOCOL)

    conn = _connect()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO enrolled_fingerprints (
                subject_id, filename, image_path, template_data,
                minutiae_count, template_version, image_width, image_height
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                subject_id.strip(),
                filename,
                image_path,
                serialized,
                minutiae_count,
                template.get("version", "sourceafis_v1"),
                image_width,
                image_height,
            ),
        )
        enrollment_id = cursor.lastrowid
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

    clear_gallery_cache()
    return int(enrollment_id)
