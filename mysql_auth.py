
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
    """Connect to the gallery database."""
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "fingersure_eval_db"),
        charset="utf8mb4",
        use_pure=True,
    )


def get_person_templates(
    subject_id: str,
    mysql_password: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve all templates enrolled for a subject from fingerprint_gallery."""
    conn = _connect()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute(
            """
            SELECT id, participant_id, finger, identity, source_image,
                   template_data, minutiae_count
            FROM fingerprint_gallery
            WHERE identity = %s OR participant_id = %s
            ORDER BY id
            """,
            (subject_id, subject_id),
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
            "fingerprint_id": row["id"],
            "filename": row["source_image"],
            "image_path": None,
            "template": tpl,
            "minutiae_count": row["minutiae_count"],
            "image_width": tpl.get("image_width", 300),
            "image_height": tpl.get("image_height", 400),
        })

    if not templates:
        raise ValueError(
            f"No enrolled fingerprint templates found for subject: {subject_id}"
        )

    return {
        "subject_id": subject_id,
        "family_id": None,
        "member_type": "GALLERY",
        "templates": templates,
    }


def get_all_gallery_templates(
    force_refresh: bool = False,
) -> List[Dict[str, Any]]:
    """Load the 42 reference gallery templates from fingersure_eval_db.fingerprint_gallery for 1:N identification."""
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
        cursor.execute(
            """
            SELECT id, participant_id, finger, identity, source_image, minutiae_count, template_data
            FROM fingerprint_gallery
            ORDER BY id ASC
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
                "fingerprint_id": row["id"],
                "id": row["id"],
                "subject_id": row["identity"],
                "identity": row["identity"],
                "participant_id": row["participant_id"],
                "finger": row["finger"],
                "family_id": None,
                "member_type": "GALLERY",
                "filename": row["source_image"],
                "image_path": None,
                "minutiae_count": row["minutiae_count"],
                "template": tpl,
            })

    finally:
        cursor.close()
        conn.close()

    # Pre-index invariant edges, buckets, and adjacency for fast 1:N comparisons
    import afis.algorithms.safis as safis
    idx_params = safis.SafisParams(
        max_neighbors=10,
        max_edge_length=320.0,
        length_tol=8.0,
        angle_tol=0.35,
        max_roots=25,
    )
    length_tol = float(idx_params.length_tol) or 1e-6
    for item in gallery:
        cand_tpl = item.get("template", item)
        if cand_tpl.get("_edges") is None:
            c_minu = cand_tpl.get("minutiae")
            if c_minu is not None and len(c_minu) >= 2:
                cand_edges = safis.build_edges(c_minu, idx_params)
                cand_buckets = {}
                cand_adj = {}
                for e in cand_edges:
                    k = int(round(e.length / length_tol))
                    cand_buckets.setdefault(k, []).append(e)
                    cand_adj.setdefault(e.ref, []).append(e)
                cand_tpl["_edges"] = cand_edges
                cand_tpl["_buckets"] = cand_buckets
                cand_tpl["_adj"] = cand_adj
            else:
                cand_tpl["_edges"] = []
                cand_tpl["_buckets"] = {}
                cand_tpl["_adj"] = {}

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
    clean_sub_id = subject_id.strip()
    if not clean_sub_id:
        raise ValueError("Subject ID cannot be empty.")

    minutiae_count = int(template.get("minutiae_count", 0))
    if minutiae_count < 2:
        raise ValueError("Not enough minutiae detected. Please capture again.")

    serialized = pickle.dumps(template, protocol=pickle.HIGHEST_PROTOCOL)

    conn = _connect()
    cursor = conn.cursor()

    try:
        # Enforce unique subject_id across enrolled fingerprints
        cursor.execute(
            "SELECT enrollment_id FROM enrolled_fingerprints WHERE subject_id = %s",
            (clean_sub_id,)
        )
        existing = cursor.fetchone()
        if existing:
            raise ValueError(f"Subject ID '{clean_sub_id}' is already enrolled (enrollment ID {existing[0]}).")

        cursor.execute(
            """
            INSERT INTO enrolled_fingerprints (
                subject_id, filename, image_path, template_data,
                minutiae_count, template_version, image_width, image_height
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                clean_sub_id,
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

