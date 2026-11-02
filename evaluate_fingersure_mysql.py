#!/usr/bin/env python3
"""
Evaluate SourceAFIS fingerprint matching accuracy BEFORE and AFTER DDRNet rectification.
Uses clean reference templates retrieved exclusively from the MySQL database 'fingersure_eval_db'.
Local .dat files are NOT read during evaluation.
"""

from __future__ import annotations

import csv
import math
import os
import pickle
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mysql.connector
import numpy as np
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcher import SourceAFISMatcher
import afis.algorithms.safis as safis

# Load database credentials and threshold from .env
load_dotenv(PROJECT_ROOT / ".env", override=True)

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
TARGET_DB = "fingersure_eval_db"
TARGET_TABLE = "fingerprint_gallery"

# Documented project threshold
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "50.0"))
if MATCH_THRESHOLD <= 1.0:
    MATCH_THRESHOLD = 50.0

DATASET_ROOT = PROJECT_ROOT / "FingerSure_Dataset"
DISTORTED_DIR = DATASET_ROOT / "distorted"
RECTIFIED_DIR = DATASET_ROOT / "rectified"
OUTPUT_DIR = DATASET_ROOT / "evaluation"


def verify_mysql_gallery() -> List[Dict[str, Any]]:
    """
    Strict verification step:
    1. Connects to MySQL server.
    2. Verifies fingersure_eval_db exists.
    3. Verifies fingerprint_gallery table exists.
    4. Confirms exactly 42 unique reference identities.
    5. Confirms all 42 template_data BLOBs are non-empty.
    6. Deserializes and validates all 42 templates using SourceAFIS format.
    Stops with RuntimeError if any check fails.
    """
    print("[Verification] Connecting to MySQL database...", flush=True)
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            charset="utf8mb4",
            use_pure=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to connect to MySQL server at {MYSQL_HOST}:{MYSQL_PORT}: {e}")

    cursor = conn.cursor(dictionary=True)
    try:
        # Check database
        cursor.execute("SHOW DATABASES LIKE %s", (TARGET_DB,))
        if not cursor.fetchall():
            raise RuntimeError(f"Database '{TARGET_DB}' does not exist.")
        print(f"[Verification] [PASS] Database '{TARGET_DB}' exists.", flush=True)

        cursor.execute(f"USE `{TARGET_DB}`;")

        # Check table
        cursor.execute(f"SHOW TABLES LIKE '{TARGET_TABLE}';")
        if not cursor.fetchall():
            raise RuntimeError(f"Table '{TARGET_TABLE}' does not exist in '{TARGET_DB}'.")
        print(f"[Verification] [PASS] Table '{TARGET_TABLE}' exists.", flush=True)

        # Check count and distinct identities
        cursor.execute(f"SELECT COUNT(*) AS total_count, COUNT(DISTINCT identity) AS unique_count FROM `{TARGET_TABLE}`;")
        counts = cursor.fetchone()
        total_count = counts["total_count"]
        unique_count = counts["unique_count"]

        if total_count != 42 or unique_count != 42:
            raise RuntimeError(
                f"Expected exactly 42 unique identities in '{TARGET_TABLE}', "
                f"found total={total_count}, unique={unique_count}."
            )
        print(f"[Verification] [PASS] Exactly 42 unique reference identities confirmed in MySQL.", flush=True)

        # Retrieve and verify all templates
        cursor.execute(
            f"""
            SELECT id, participant_id, finger, identity, source_image, minutiae_count,
                   LENGTH(template_data) AS blob_size, template_data
            FROM `{TARGET_TABLE}`
            ORDER BY identity ASC;
            """
        )
        rows = cursor.fetchall()

        gallery = []
        for r in rows:
            if not r["blob_size"] or r["blob_size"] <= 0:
                raise RuntimeError(f"Empty template_data BLOB found for identity '{r['identity']}'")

            try:
                tpl = pickle.loads(r["template_data"])
            except Exception as e:
                raise RuntimeError(f"Failed to deserialize template for '{r['identity']}': {e}")

            if not isinstance(tpl, dict) or "minutiae" not in tpl:
                raise RuntimeError(f"Deserialized object for '{r['identity']}' is not a valid SourceAFIS template dict.")

            gallery.append({
                "id": r["id"],
                "subject_id": r["identity"],
                "identity": r["identity"],
                "participant_id": r["participant_id"],
                "finger": r["finger"],
                "source_image": r["source_image"],
                "filename": f"{r['identity']}.png",
                "minutiae_count": r["minutiae_count"],
                "template": tpl,
            })

        print(f"[Verification] [PASS] All 42 reference templates retrieved and verified from MySQL.", flush=True)
        return gallery

    finally:
        cursor.close()
        conn.close()


def parse_query_filename(filename: str) -> Tuple[str, str, str]:
    """
    Extract (identity, participant_id, finger) from query filename.
    e.g. Kusuma_RIGHT_THUMB_distorted.png -> (Kusuma_RIGHT_THUMB, Kusuma, RIGHT_THUMB)
    """
    base = Path(filename).stem
    for suffix in ["_distorted", "_rectified", "_clean"]:
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break

    identity = base
    if "_RIGHT_THUMB" in identity:
        participant_id = identity.replace("_RIGHT_THUMB", "")
        finger = "RIGHT_THUMB"
    elif "_LEFT_THUMB" in identity:
        participant_id = identity.replace("_LEFT_THUMB", "")
        finger = "LEFT_THUMB"
    else:
        parts = identity.rsplit("_", 1)
        participant_id = parts[0] if len(parts) > 1 else identity
        finger = parts[1] if len(parts) > 1 else "UNKNOWN"

    return identity, participant_id, finger


def extract_single_query(
    image_path: Path,
    matcher: SourceAFISMatcher,
) -> Dict[str, Any]:
    """Read a fingerprint image and extract its SourceAFIS template."""
    identity, participant_id, finger = parse_query_filename(image_path.name)
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "filename": image_path.name,
            "path": image_path,
            "identity": identity,
            "participant_id": participant_id,
            "finger": finger,
            "template": None,
            "minutiae_count": 0,
            "extraction_success": False,
            "error": "Failed to read image file from disk",
        }

    try:
        tpl = matcher.extract_template(img)
        minu_count = tpl.get("minutiae_count", 0)
        if minu_count < 2:
            return {
                "filename": image_path.name,
                "path": image_path,
                "identity": identity,
                "participant_id": participant_id,
                "finger": finger,
                "template": tpl,
                "minutiae_count": minu_count,
                "extraction_success": False,
                "error": f"Minutiae count {minu_count} < 2",
            }
        return {
            "filename": image_path.name,
            "path": image_path,
            "identity": identity,
            "participant_id": participant_id,
            "finger": finger,
            "template": tpl,
            "minutiae_count": minu_count,
            "extraction_success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "filename": image_path.name,
            "path": image_path,
            "identity": identity,
            "participant_id": participant_id,
            "finger": finger,
            "template": None,
            "minutiae_count": 0,
            "extraction_success": False,
            "error": str(e),
        }


def run_experiment(
    query_dir: Path,
    file_suffix: str,
    gallery: List[Dict[str, Any]],
    matcher: SourceAFISMatcher,
    exp_name: str,
) -> Dict[str, Any]:
    """
    Execute 1:N matching for all query images in query_dir against the MySQL gallery.
    """
    print(f"\n[{exp_name}] Scanning query directory: {query_dir}", flush=True)
    query_files = sorted([f for f in query_dir.glob(f"*{file_suffix}") if f.is_file()])
    print(f"[{exp_name}] Discovered {len(query_files)} query image(s).", flush=True)

    if len(query_files) != 42:
        raise RuntimeError(f"Expected 42 queries in {query_dir}, found {len(query_files)}")

    # Extract all query templates
    print(f"[{exp_name}] Extracting query templates with thread pool...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        queries = list(executor.map(lambda p: extract_single_query(p, matcher), query_files))
    t_extract = time.time() - t0
    print(f"[{exp_name}] Extracted {len(queries)} templates in {t_extract:.2f}s.", flush=True)

    successful_extractions = sum(1 for q in queries if q["extraction_success"])
    failed_extractions = len(queries) - successful_extractions
    print(f"[{exp_name}] Extraction Success: {successful_extractions}/42, Failures: {failed_extractions}", flush=True)

    # Pre-index gallery templates once for ultra-fast 1:N identification
    length_tol = float(matcher.params.length_tol) or 1e-6
    for cand in gallery:
        cand_tpl = cand["template"]
        c_minu = cand_tpl.get("minutiae")
        if c_minu is not None and len(c_minu) >= 2:
            if cand_tpl.get("_edges") is None:
                cand_edges = safis.build_edges(c_minu, matcher.params)
                cand_buckets: Dict[int, List[safis._Edge]] = {}
                cand_adj: Dict[int, List[safis._Edge]] = {}
                for e in cand_edges:
                    k = int(round(e.length / length_tol))
                    cand_buckets.setdefault(k, []).append(e)
                    cand_adj.setdefault(e.ref, []).append(e)
                cand_tpl["_edges"] = cand_edges
                cand_tpl["_buckets"] = cand_buckets
                cand_tpl["_adj"] = cand_adj
                cand_tpl["_cache_params_key"] = (
                    matcher.params.max_neighbors,
                    matcher.params.max_edge_length,
                    matcher.params.length_tol,
                    matcher.params.angle_tol,
                )

    # 1:N Biometric Matching against MySQL Gallery
    print(f"[{exp_name}] Performing 1:N biometric matching against MySQL gallery...", flush=True)
    t_match_start = time.time()
    results = []
    all_genuine_scores = []
    all_impostor_pairs = []

    for q in queries:
        probe_id = q["identity"]
        probe_tpl = q["template"]

        if not q["extraction_success"] or probe_tpl is None:
            results.append({
                "query_filename": q["filename"],
                "identity": probe_id,
                "participant_id": q["participant_id"],
                "finger": q["finger"],
                "query_minutiae_count": 0,
                "extraction_success": False,
                "genuine_score": 0.0,
                "genuine_matched_minutiae": 0,
                "genuine_decision": False,
                "top1_identity": "EXTRACTION_FAILED",
                "top1_score": 0.0,
                "top1_correct": False,
                "top5_correct": False,
                "rank_of_genuine": 42,
                "best_impostor_identity": "NONE",
                "best_impostor_score": 0.0,
                "impostor_margin": 0.0,
                "impostor_scores": {cand["identity"]: 0.0 for cand in gallery if cand["identity"] != probe_id},
            })
            all_genuine_scores.append(0.0)
            for cand in gallery:
                if cand["identity"] != probe_id:
                    all_impostor_pairs.append({
                        "query_identity": probe_id,
                        "candidate_identity": cand["identity"],
                        "score": 0.0,
                    })
            continue

        # Run 1:N identification
        id_res = matcher.identify_1toN(
            probe_tpl,
            gallery,
            top_n=len(gallery),
            threshold=MATCH_THRESHOLD,
            is_rectified=False,
        )
        cands = id_res.get("candidates", [])

        # Find genuine candidate
        genuine_cand = next((c for c in cands if c["subject_id"] == probe_id), None)
        if genuine_cand is None:
            raise RuntimeError(f"Missing genuine candidate in gallery for identity '{probe_id}'")

        genuine_score = float(genuine_cand["raw_score"])
        genuine_minutiae = int(genuine_cand.get("matched_minutiae_count", 0))
        genuine_decision = bool(genuine_score >= MATCH_THRESHOLD)
        genuine_rank = next(i for i, c in enumerate(cands, 1) if c["subject_id"] == probe_id)

        # Top-1 and Top-5
        top1_cand = cands[0]
        top1_identity = top1_cand["subject_id"]
        top1_score = float(top1_cand["raw_score"])
        top1_correct = bool(top1_identity == probe_id)

        top5_identities = [c["subject_id"] for c in cands[:5]]
        top5_correct = bool(probe_id in top5_identities)

        # Impostor comparisons
        impostors = [c for c in cands if c["subject_id"] != probe_id]
        best_impostor = max(impostors, key=lambda x: x["raw_score"])
        best_impostor_score = float(best_impostor["raw_score"])
        impostor_margin = genuine_score - best_impostor_score

        impostor_dict = {}
        for imp in impostors:
            imp_score = float(imp["raw_score"])
            impostor_dict[imp["subject_id"]] = imp_score
            all_impostor_pairs.append({
                "query_identity": probe_id,
                "candidate_identity": imp["subject_id"],
                "score": imp_score,
            })

        all_genuine_scores.append(genuine_score)

        results.append({
            "query_filename": q["filename"],
            "identity": probe_id,
            "participant_id": q["participant_id"],
            "finger": q["finger"],
            "query_minutiae_count": q["minutiae_count"],
            "extraction_success": True,
            "genuine_score": genuine_score,
            "genuine_matched_minutiae": genuine_minutiae,
            "genuine_decision": genuine_decision,
            "top1_identity": top1_identity,
            "top1_score": top1_score,
            "top1_correct": top1_correct,
            "top5_correct": top5_correct,
            "rank_of_genuine": genuine_rank,
            "best_impostor_identity": best_impostor["subject_id"],
            "best_impostor_score": best_impostor_score,
            "impostor_margin": round(impostor_margin, 2),
            "impostor_scores": impostor_dict,
        })

    t_match_total = time.time() - t_match_start
    print(f"[{exp_name}] Completed matching in {t_match_total:.2f}s.", flush=True)

    # Compute identification & verification metrics
    total_queries = len(results)
    top1_correct_count = sum(1 for r in results if r["top1_correct"])
    top5_correct_count = sum(1 for r in results if r["top5_correct"])
    top1_accuracy = (top1_correct_count / total_queries) * 100.0
    top5_accuracy = (top5_correct_count / total_queries) * 100.0

    false_rejections = sum(1 for s in all_genuine_scores if s < MATCH_THRESHOLD)
    frr = (false_rejections / total_queries) * 100.0
    gar = 100.0 - frr

    total_impostors = len(all_impostor_pairs)
    false_acceptances = sum(1 for imp in all_impostor_pairs if imp["score"] >= MATCH_THRESHOLD)
    far = (false_acceptances / total_impostors) * 100.0

    gen_arr = np.array(all_genuine_scores)
    imp_scores_list = [imp["score"] for imp in all_impostor_pairs]
    imp_arr = np.array(imp_scores_list)

    summary = {
        "exp_name": exp_name,
        "total_queries": total_queries,
        "successful_extractions": successful_extractions,
        "failed_extractions": failed_extractions,
        "total_gallery": len(gallery),
        "total_genuine_comparisons": total_queries,
        "total_impostor_comparisons": total_impostors,
        "top1_correct_count": top1_correct_count,
        "top1_accuracy_pct": round(top1_accuracy, 2),
        "top5_correct_count": top5_correct_count,
        "top5_accuracy_pct": round(top5_accuracy, 2),
        "genuine_mean_score": round(float(np.mean(gen_arr)), 2),
        "genuine_median_score": round(float(np.median(gen_arr)), 2),
        "genuine_std_score": round(float(np.std(gen_arr)), 2),
        "genuine_min_score": round(float(np.min(gen_arr)), 2),
        "genuine_max_score": round(float(np.max(gen_arr)), 2),
        "impostor_mean_score": round(float(np.mean(imp_arr)), 2),
        "impostor_median_score": round(float(np.median(imp_arr)), 2),
        "impostor_std_score": round(float(np.std(imp_arr)), 2),
        "impostor_min_score": round(float(np.min(imp_arr)), 2),
        "impostor_max_score": round(float(np.max(imp_arr)), 2),
        "threshold": MATCH_THRESHOLD,
        "false_rejections": false_rejections,
        "frr_pct": round(frr, 2),
        "gar_pct": round(gar, 2),
        "false_acceptances": false_acceptances,
        "far_pct": round(far, 4),
    }

    return {
        "results": results,
        "summary": summary,
        "genuine_scores": all_genuine_scores,
        "impostor_pairs": all_impostor_pairs,
    }


def write_csv_files(
    output_dir: Path,
    exp_a: Dict[str, Any],
    exp_b: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Write all CSV artifacts required by the specification."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. before_rectification_results.csv
    f_before = output_dir / "before_rectification_results.csv"
    fieldnames_res = [
        "query_filename",
        "identity",
        "participant_id",
        "finger",
        "query_minutiae_count",
        "extraction_success",
        "genuine_score",
        "genuine_matched_minutiae",
        "genuine_decision",
        "top1_identity",
        "top1_score",
        "top1_correct",
        "top5_correct",
        "rank_of_genuine",
        "best_impostor_identity",
        "best_impostor_score",
        "impostor_margin",
    ]
    with open(f_before, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_res)
        writer.writeheader()
        for r in exp_a["results"]:
            writer.writerow({k: r[k] for k in fieldnames_res})
    print(f"[OK] Generated: {f_before}", flush=True)

    # 2. after_rectification_results.csv
    f_after = output_dir / "after_rectification_results.csv"
    with open(f_after, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_res)
        writer.writeheader()
        for r in exp_b["results"]:
            writer.writerow({k: r[k] for k in fieldnames_res})
    print(f"[OK] Generated: {f_after}", flush=True)

    # 3. paired_comparison.csv
    f_paired = output_dir / "paired_comparison.csv"
    dist_map = {r["identity"]: r for r in exp_a["results"]}
    rect_map = {r["identity"]: r for r in exp_b["results"]}
    paired_list = []

    for identity in sorted(dist_map.keys()):
        d = dist_map[identity]
        r = rect_map.get(identity)
        if not r:
            continue

        d_score = d["genuine_score"]
        r_score = r["genuine_score"]
        score_diff = round(r_score - d_score, 2)
        pct_change = round(((r_score - d_score) / (d_score if d_score > 0 else 1.0)) * 100.0, 2)
        improved = bool(score_diff > 0)

        d_dec = d["genuine_decision"]
        r_dec = r["genuine_decision"]

        if not d_dec and r_dec:
            transition = "RESCUED (Fail -> Pass)"
        elif d_dec and r_dec:
            transition = "MAINTAINED (Pass -> Pass)"
        elif not d_dec and not r_dec:
            transition = "PERSISTENT_FAIL (Fail -> Fail)"
        else:
            transition = "DEGRADED (Pass -> Fail)"

        paired_list.append({
            "identity": identity,
            "participant_id": d["participant_id"],
            "finger": d["finger"],
            "distorted_score": d_score,
            "rectified_score": r_score,
            "score_difference": score_diff,
            "percentage_change": pct_change,
            "rectification_improved": improved,
            "distorted_decision": d_dec,
            "rectified_decision": r_dec,
            "transition": transition,
            "distorted_rank": d["rank_of_genuine"],
            "rectified_rank": r["rank_of_genuine"],
            "rank_improved": bool(r["rank_of_genuine"] < d["rank_of_genuine"]),
            "distorted_top1_correct": d["top1_correct"],
            "rectified_top1_correct": r["top1_correct"],
            "distorted_top5_correct": d["top5_correct"],
            "rectified_top5_correct": r["top5_correct"],
            "distorted_minutiae": d["query_minutiae_count"],
            "rectified_minutiae": r["query_minutiae_count"],
        })

    fieldnames_paired = [
        "identity",
        "participant_id",
        "finger",
        "distorted_score",
        "rectified_score",
        "score_difference",
        "percentage_change",
        "rectification_improved",
        "distorted_decision",
        "rectified_decision",
        "transition",
        "distorted_rank",
        "rectified_rank",
        "rank_improved",
        "distorted_top1_correct",
        "rectified_top1_correct",
        "distorted_top5_correct",
        "rectified_top5_correct",
        "distorted_minutiae",
        "rectified_minutiae",
    ]
    with open(f_paired, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_paired)
        writer.writeheader()
        for p in paired_list:
            writer.writerow(p)
    print(f"[OK] Generated: {f_paired}", flush=True)

    # 4. genuine_scores.csv
    f_genuine = output_dir / "genuine_scores.csv"
    genuine_rows = []
    for p in paired_list:
        ident = p["identity"]
        d = dist_map[ident]
        r = rect_map[ident]
        genuine_rows.append({
            "identity": ident,
            "participant_id": p["participant_id"],
            "finger": p["finger"],
            "distorted_genuine_score": d["genuine_score"],
            "distorted_matched_minutiae": d["genuine_matched_minutiae"],
            "distorted_decision": d["genuine_decision"],
            "rectified_genuine_score": r["genuine_score"],
            "rectified_matched_minutiae": r["genuine_matched_minutiae"],
            "rectified_decision": r["genuine_decision"],
            "score_difference": p["score_difference"],
            "improved": p["rectification_improved"],
        })
    fieldnames_gen = [
        "identity",
        "participant_id",
        "finger",
        "distorted_genuine_score",
        "distorted_matched_minutiae",
        "distorted_decision",
        "rectified_genuine_score",
        "rectified_matched_minutiae",
        "rectified_decision",
        "score_difference",
        "improved",
    ]
    with open(f_genuine, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_gen)
        writer.writeheader()
        for g in genuine_rows:
            writer.writerow(g)
    print(f"[OK] Generated: {f_genuine}", flush=True)

    # 5. impostor_scores.csv
    f_impostor = output_dir / "impostor_scores.csv"
    impostor_rows = []
    # Index impostor pairs
    dist_imp_map = {(imp["query_identity"], imp["candidate_identity"]): imp["score"] for imp in exp_a["impostor_pairs"]}
    rect_imp_map = {(imp["query_identity"], imp["candidate_identity"]): imp["score"] for imp in exp_b["impostor_pairs"]}

    all_keys = sorted(dist_imp_map.keys())
    for (q_id, c_id) in all_keys:
        d_sc = dist_imp_map.get((q_id, c_id), 0.0)
        r_sc = rect_imp_map.get((q_id, c_id), 0.0)
        impostor_rows.append({
            "query_identity": q_id,
            "candidate_identity": c_id,
            "distorted_impostor_score": d_sc,
            "distorted_false_accept": bool(d_sc >= MATCH_THRESHOLD),
            "rectified_impostor_score": r_sc,
            "rectified_false_accept": bool(r_sc >= MATCH_THRESHOLD),
            "impostor_score_diff": round(r_sc - d_sc, 2),
        })

    fieldnames_imp = [
        "query_identity",
        "candidate_identity",
        "distorted_impostor_score",
        "distorted_false_accept",
        "rectified_impostor_score",
        "rectified_false_accept",
        "impostor_score_diff",
    ]
    with open(f_impostor, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_imp)
        writer.writeheader()
        for imp in impostor_rows:
            writer.writerow(imp)
    print(f"[OK] Generated: {f_impostor} ({len(impostor_rows)} comparisons)", flush=True)

    # 6. evaluation_summary.csv
    f_summary = output_dir / "evaluation_summary.csv"
    sum_a = exp_a["summary"]
    sum_b = exp_b["summary"]

    improved_count = sum(1 for p in paired_list if p["rectification_improved"])
    degraded_count = sum(1 for p in paired_list if p["score_difference"] < 0)
    unchanged_count = sum(1 for p in paired_list if p["score_difference"] == 0)

    rescued_count = sum(1 for p in paired_list if p["transition"] == "RESCUED (Fail -> Pass)")
    maintained_count = sum(1 for p in paired_list if p["transition"] == "MAINTAINED (Pass -> Pass)")
    failed_count = sum(1 for p in paired_list if p["transition"] == "PERSISTENT_FAIL (Fail -> Fail)")
    degraded_trans_count = sum(1 for p in paired_list if p["transition"] == "DEGRADED (Pass -> Fail)")

    summary_rows = [
        ("Reference Gallery Source", "MySQL (fingersure_eval_db.fingerprint_gallery)", "MySQL (fingersure_eval_db.fingerprint_gallery)", "N/A"),
        ("Enrolled Reference Gallery Size", str(sum_a["total_gallery"]), str(sum_b["total_gallery"]), "0"),
        ("Total Query Images", str(sum_a["total_queries"]), str(sum_b["total_queries"]), "0"),
        ("Successful Query Extractions", str(sum_a["successful_extractions"]), str(sum_b["successful_extractions"]), str(sum_b["successful_extractions"] - sum_a["successful_extractions"])),
        ("Failed Query Extractions", str(sum_a["failed_extractions"]), str(sum_b["failed_extractions"]), str(sum_b["failed_extractions"] - sum_a["failed_extractions"])),
        ("Biometric Threshold (MATCH_THRESHOLD)", str(sum_a["threshold"]), str(sum_b["threshold"]), "0"),
        ("Total Genuine Comparisons", str(sum_a["total_genuine_comparisons"]), str(sum_b["total_genuine_comparisons"]), "0"),
        ("Total Impostor Comparisons", str(sum_a["total_impostor_comparisons"]), str(sum_b["total_impostor_comparisons"]), "0"),
        ("Top-1 Identification Accuracy (%)", f"{sum_a['top1_accuracy_pct']:.2f}%", f"{sum_b['top1_accuracy_pct']:.2f}%", f"{sum_b['top1_accuracy_pct'] - sum_a['top1_accuracy_pct']:+.2f}%"),
        ("Top-5 Identification Accuracy (%)", f"{sum_a['top5_accuracy_pct']:.2f}%", f"{sum_b['top5_accuracy_pct']:.2f}%", f"{sum_b['top5_accuracy_pct'] - sum_a['top5_accuracy_pct']:+.2f}%"),
        ("False Rejection Rate (FRR @ 50.0) (%)", f"{sum_a['frr_pct']:.2f}%", f"{sum_b['frr_pct']:.2f}%", f"{sum_b['frr_pct'] - sum_a['frr_pct']:+.2f}%"),
        ("Genuine Acceptance Rate (GAR @ 50.0) (%)", f"{sum_a['gar_pct']:.2f}%", f"{sum_b['gar_pct']:.2f}%", f"{sum_b['gar_pct'] - sum_a['gar_pct']:+.2f}%"),
        ("False Acceptance Rate (FAR @ 50.0) (%)", f"{sum_a['far_pct']:.4f}%", f"{sum_b['far_pct']:.4f}%", f"{sum_b['far_pct'] - sum_a['far_pct']:+.4f}%"),
        ("Mean Genuine Score", f"{sum_a['genuine_mean_score']:.2f}", f"{sum_b['genuine_mean_score']:.2f}", f"{sum_b['genuine_mean_score'] - sum_a['genuine_mean_score']:+.2f}"),
        ("Median Genuine Score", f"{sum_a['genuine_median_score']:.2f}", f"{sum_b['genuine_median_score']:.2f}", f"{sum_b['genuine_median_score'] - sum_a['genuine_median_score']:+.2f}"),
        ("Std Dev Genuine Score", f"{sum_a['genuine_std_score']:.2f}", f"{sum_b['genuine_std_score']:.2f}", f"{sum_b['genuine_std_score'] - sum_a['genuine_std_score']:+.2f}"),
        ("Min Genuine Score", f"{sum_a['genuine_min_score']:.2f}", f"{sum_b['genuine_min_score']:.2f}", f"{sum_b['genuine_min_score'] - sum_a['genuine_min_score']:+.2f}"),
        ("Max Genuine Score", f"{sum_a['genuine_max_score']:.2f}", f"{sum_b['genuine_max_score']:.2f}", f"{sum_b['genuine_max_score'] - sum_a['genuine_max_score']:+.2f}"),
        ("Mean Impostor Score", f"{sum_a['impostor_mean_score']:.2f}", f"{sum_b['impostor_mean_score']:.2f}", f"{sum_b['impostor_mean_score'] - sum_a['impostor_mean_score']:+.2f}"),
        ("Median Impostor Score", f"{sum_a['impostor_median_score']:.2f}", f"{sum_b['impostor_median_score']:.2f}", f"{sum_b['impostor_median_score'] - sum_a['impostor_median_score']:+.2f}"),
        ("Max Impostor Score", f"{sum_a['impostor_max_score']:.2f}", f"{sum_b['impostor_max_score']:.2f}", f"{sum_b['impostor_max_score'] - sum_a['impostor_max_score']:+.2f}"),
        ("Identities with Improved Genuine Score", f"{improved_count} / {len(paired_list)} ({improved_count/len(paired_list)*100:.1f}%)", "N/A", "N/A"),
        ("Identities with Degraded Genuine Score", f"{degraded_count} / {len(paired_list)} ({degraded_count/len(paired_list)*100:.1f}%)", "N/A", "N/A"),
        ("Identities with Unchanged Genuine Score", f"{unchanged_count} / {len(paired_list)} ({unchanged_count/len(paired_list)*100:.1f}%)", "N/A", "N/A"),
        ("Verification Rescued (Fail -> Pass)", f"{rescued_count} identity(ies)", "N/A", "N/A"),
        ("Verification Maintained (Pass -> Pass)", f"{maintained_count} identity(ies)", "N/A", "N/A"),
        ("Verification Persistent Fail (Fail -> Fail)", f"{failed_count} identity(ies)", "N/A", "N/A"),
        ("Verification Degraded (Pass -> Fail)", f"{degraded_trans_count} identity(ies)", "N/A", "N/A"),
    ]

    with open(f_summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Before Rectification", "After Rectification", "Difference"])
        for row in summary_rows:
            writer.writerow(row)
    print(f"[OK] Generated: {f_summary}", flush=True)

    return paired_list, genuine_rows, impostor_rows


def write_text_report(
    output_dir: Path,
    exp_a: Dict[str, Any],
    exp_b: Dict[str, Any],
    paired_list: List[Dict[str, Any]],
) -> None:
    """Generate comprehensive evaluation report text file."""
    f_report = output_dir / "evaluation_report.txt"
    sum_a = exp_a["summary"]
    sum_b = exp_b["summary"]

    improved_count = sum(1 for p in paired_list if p["rectification_improved"])
    degraded_count = sum(1 for p in paired_list if p["score_difference"] < 0)
    unchanged_count = sum(1 for p in paired_list if p["score_difference"] == 0)
    rescued_count = sum(1 for p in paired_list if p["transition"] == "RESCUED (Fail -> Pass)")
    maintained_count = sum(1 for p in paired_list if p["transition"] == "MAINTAINED (Pass -> Pass)")
    failed_count = sum(1 for p in paired_list if p["transition"] == "PERSISTENT_FAIL (Fail -> Fail)")
    degraded_trans_count = sum(1 for p in paired_list if p["transition"] == "DEGRADED (Pass -> Fail)")

    lines = [
        "=" * 85,
        "   FINGERSURE BIOMETRIC ACCURACY EVALUATION REPORT: BEFORE vs. AFTER DDRNET",
        "          (Reference Gallery Retrieved Exclusively from MySQL Database)",
        "=" * 85,
        f"Execution Timestamp:        {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"MySQL Database:             {TARGET_DB}",
        f"MySQL Gallery Table:        {TARGET_TABLE}",
        f"Clean Enrolled Identities:  {sum_a['total_gallery']} (Verified in MySQL)",
        f"Probe Queries (Distorted):  {sum_a['total_queries']} images from FingerSure_Dataset/distorted/",
        f"Probe Queries (Rectified):  {sum_b['total_queries']} images from FingerSure_Dataset/rectified/",
        f"Biometric Matcher:          SourceAFIS (afis package / SourceAFISMatcher)",
        f"Documented Threshold:       MATCH_THRESHOLD = {MATCH_THRESHOLD:.1f}",
        "=" * 85,
        "",
        "1. FINAL COMPARISON TABLE",
        "-" * 85,
        f"{'Metric':<35} | {'Before Rectification':<20} | {'After Rectification':<20} | {'Difference':<12}",
        "-" * 85,
        f"{'Top-1 accuracy':<35} | {sum_a['top1_accuracy_pct']:>18.2f}% | {sum_b['top1_accuracy_pct']:>18.2f}% | {sum_b['top1_accuracy_pct'] - sum_a['top1_accuracy_pct']:>+10.2f}%",
        f"{'Top-5 accuracy':<35} | {sum_a['top5_accuracy_pct']:>18.2f}% | {sum_b['top5_accuracy_pct']:>18.2f}% | {sum_b['top5_accuracy_pct'] - sum_a['top5_accuracy_pct']:>+10.2f}%",
        f"{'FAR (False Acceptance Rate)':<35} | {sum_a['far_pct']:>18.4f}% | {sum_b['far_pct']:>18.4f}% | {sum_b['far_pct'] - sum_a['far_pct']:>+10.4f}%",
        f"{'FRR (False Rejection Rate)':<35} | {sum_a['frr_pct']:>18.2f}% | {sum_b['frr_pct']:>18.2f}% | {sum_b['frr_pct'] - sum_a['frr_pct']:>+10.2f}%",
        f"{'Mean genuine score':<35} | {sum_a['genuine_mean_score']:>20.2f} | {sum_b['genuine_mean_score']:>20.2f} | {sum_b['genuine_mean_score'] - sum_a['genuine_mean_score']:>+10.2f}",
        f"{'Mean impostor score':<35} | {sum_a['impostor_mean_score']:>20.2f} | {sum_b['impostor_mean_score']:>20.2f} | {sum_b['impostor_mean_score'] - sum_a['impostor_mean_score']:>+10.2f}",
        f"{'Successful query extractions':<35} | {sum_a['successful_extractions']:>20} | {sum_b['successful_extractions']:>20} | {sum_b['successful_extractions'] - sum_a['successful_extractions']:>+10}",
        f"{'Failed query extractions':<35} | {sum_a['failed_extractions']:>20} | {sum_b['failed_extractions']:>20} | {sum_b['failed_extractions'] - sum_a['failed_extractions']:>+10}",
        "-" * 85,
        "",
        "2. ADDITIONAL STATISTICAL METRICS",
        "-" * 85,
        f"{'Metric':<35} | {'Before Rectification':<20} | {'After Rectification':<20} | {'Difference':<12}",
        "-" * 85,
        f"{'Median Genuine Score':<35} | {sum_a['genuine_median_score']:>20.2f} | {sum_b['genuine_median_score']:>20.2f} | {sum_b['genuine_median_score'] - sum_a['genuine_median_score']:>+10.2f}",
        f"{'Std Dev Genuine Score':<35} | {sum_a['genuine_std_score']:>20.2f} | {sum_b['genuine_std_score']:>20.2f} | {sum_b['genuine_std_score'] - sum_a['genuine_std_score']:>+10.2f}",
        f"{'Min Genuine Score':<35} | {sum_a['genuine_min_score']:>20.2f} | {sum_b['genuine_min_score']:>20.2f} | {sum_b['genuine_min_score'] - sum_a['genuine_min_score']:>+10.2f}",
        f"{'Max Genuine Score':<35} | {sum_a['genuine_max_score']:>20.2f} | {sum_b['genuine_max_score']:>20.2f} | {sum_b['genuine_max_score'] - sum_a['genuine_max_score']:>+10.2f}",
        f"{'Genuine Acceptance Rate (GAR)':<35} | {sum_a['gar_pct']:>18.2f}% | {sum_b['gar_pct']:>18.2f}% | {sum_b['gar_pct'] - sum_a['gar_pct']:>+10.2f}%",
        f"{'Max Impostor Score':<35} | {sum_a['impostor_max_score']:>20.2f} | {sum_b['impostor_max_score']:>20.2f} | {sum_b['impostor_max_score'] - sum_a['impostor_max_score']:>+10.2f}",
        "-" * 85,
        "",
        "3. PAIRED SCORE ANALYSIS & VERIFICATION TRANSITIONS",
        "-" * 85,
        f"Total Finger Identities Evaluated:         {len(paired_list)}",
        f"Identities with Higher Genuine Score:       {improved_count} / {len(paired_list)} ({improved_count / len(paired_list) * 100:.1f}%)",
        f"Identities with Lower Genuine Score:        {degraded_count} / {len(paired_list)} ({degraded_count / len(paired_list) * 100:.1f}%)",
        f"Identities with Unchanged Score:            {unchanged_count} / {len(paired_list)} ({unchanged_count / len(paired_list) * 100:.1f}%)",
        f"Rescued Identities (Fail -> Pass @ 50.0):   {rescued_count}",
        f"Maintained Identities (Pass -> Pass @ 50.0):{maintained_count}",
        f"Persistent Failures (Fail -> Fail @ 50.0):  {failed_count}",
        f"Degraded Transitions (Pass -> Fail @ 50.0): {degraded_trans_count}",
        "-" * 85,
        "",
        "4. THRESHOLD SWEEP (FAR, FRR, GAR) ACROSS OPERATING POINTS",
        "-" * 85,
        f"{'Threshold':<12} | {'Distorted FRR':<15} {'Distorted FAR':<15} | {'Rectified FRR':<15} {'Rectified FAR':<15}",
        "-" * 85,
    ]

    # Calculate threshold sweep
    all_gen_a = exp_a["genuine_scores"]
    all_imp_a = [imp["score"] for imp in exp_a["impostor_pairs"]]
    all_gen_b = exp_b["genuine_scores"]
    all_imp_b = [imp["score"] for imp in exp_b["impostor_pairs"]]

    sweep_thresholds = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    for tau in sweep_thresholds:
        frr_a = (sum(1 for s in all_gen_a if s < tau) / len(all_gen_a)) * 100.0
        far_a = (sum(1 for s in all_imp_a if s >= tau) / len(all_imp_a)) * 100.0
        frr_b = (sum(1 for s in all_gen_b if s < tau) / len(all_gen_b)) * 100.0
        far_b = (sum(1 for s in all_imp_b if s >= tau) / len(all_imp_b)) * 100.0
        marker = " <-- (Documented Operating Point)" if tau == MATCH_THRESHOLD else ""
        lines.append(
            f"{tau:<12.1f} | {frr_a:>13.2f}% {far_a:>13.4f}% | {frr_b:>13.2f}% {far_b:>13.4f}%{marker}"
        )

    lines.extend([
        "-" * 85,
        "",
        "5. PER-IDENTITY MATCHING SCORE BREAKDOWN",
        "-" * 85,
        f"{'Identity':<30} {'Distorted':<12} {'Rectified':<12} {'Diff':<10} {'Rank Dist':<10} {'Rank Rect':<10} {'Transition'}",
        "-" * 85,
    ])

    for p in paired_list:
        lines.append(
            f"{p['identity']:<30} "
            f"{p['distorted_score']:>8.2f}    "
            f"{p['rectified_score']:>8.2f}    "
            f"{p['score_difference']:>+7.2f}   "
            f"{p['distorted_rank']:>6}     "
            f"{p['rectified_rank']:>6}     "
            f"{p['transition']}"
        )

    lines.extend([
        "-" * 85,
        "",
        "6. VALIDATION & SECURITY CONFORMANCE",
        "-" * 85,
        f"[✓] MySQL Database: Confirmed active connection to '{TARGET_DB}'.",
        f"[✓] Gallery Loading: All {sum_a['total_gallery']} reference templates fetched via SQL from '{TARGET_TABLE}'.",
        f"[✓] Zero Local .dat Dependency: Matching comparisons used memory-deserialized database records only.",
        f"[✓] Evaluation Isolation: Production database and application tables were untouched.",
        "=" * 85,
    ])

    report_text = "\n".join(lines)
    with open(f_report, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[OK] Generated: {f_report}", flush=True)


def main():
    print("===========================================================================", flush=True)
    print("      FINGERSURE DATABASE-BASED BIOMETRIC EVALUATION PIPELINE", flush=True)
    print("===========================================================================", flush=True)
    print(f"Target Database:       {TARGET_DB}", flush=True)
    print(f"Target Table:          {TARGET_TABLE}", flush=True)
    print(f"Distorted Queries:     {DISTORTED_DIR}", flush=True)
    print(f"Rectified Queries:     {RECTIFIED_DIR}", flush=True)
    print(f"Biometric Threshold:   {MATCH_THRESHOLD}", flush=True)
    print(f"Output Directory:      {OUTPUT_DIR}", flush=True)
    print("===========================================================================\n", flush=True)

    # Step 1: Strict MySQL verification and loading
    gallery = verify_mysql_gallery()
    matcher = SourceAFISMatcher()

    # Step 2: Experiment A (Before Rectification - Distorted Queries)
    print("\n" + "=" * 70, flush=True)
    print(" EXPERIMENT A: DISTORTED PROBES vs. MYSQL REFERENCE GALLERY", flush=True)
    print("=" * 70, flush=True)
    exp_a = run_experiment(
        query_dir=DISTORTED_DIR,
        file_suffix="_distorted.png",
        gallery=gallery,
        matcher=matcher,
        exp_name="Experiment A (Distorted)",
    )

    # Step 3: Experiment B (After Rectification - Rectified Queries)
    print("\n" + "=" * 70, flush=True)
    print(" EXPERIMENT B: RECTIFIED PROBES vs. MYSQL REFERENCE GALLERY", flush=True)
    print("=" * 70, flush=True)
    exp_b = run_experiment(
        query_dir=RECTIFIED_DIR,
        file_suffix="_rectified.png",
        gallery=gallery,
        matcher=matcher,
        exp_name="Experiment B (Rectified)",
    )

    # Step 4: Generate all 7 output files
    print("\n" + "=" * 70, flush=True)
    print(" GENERATING EVALUATION ARTIFACTS & REPORTS", flush=True)
    print("=" * 70, flush=True)
    paired_list, gen_rows, imp_rows = write_csv_files(OUTPUT_DIR, exp_a, exp_b)
    write_text_report(OUTPUT_DIR, exp_a, exp_b, paired_list)

    # Step 5: Print Final Comparison Table exactly as requested
    sum_a = exp_a["summary"]
    sum_b = exp_b["summary"]

    print("\n" + "=" * 85, flush=True)
    print("                           FINAL COMPARISON TABLE", flush=True)
    print("=" * 85, flush=True)
    print(f"{'Metric':<35} | {'Before Rectification':<20} | {'After Rectification':<20} | {'Difference':<12}", flush=True)
    print("-" * 85, flush=True)
    print(f"{'Top-1 accuracy':<35} | {sum_a['top1_accuracy_pct']:>18.2f}% | {sum_b['top1_accuracy_pct']:>18.2f}% | {sum_b['top1_accuracy_pct'] - sum_a['top1_accuracy_pct']:>+10.2f}%", flush=True)
    print(f"{'Top-5 accuracy':<35} | {sum_a['top5_accuracy_pct']:>18.2f}% | {sum_b['top5_accuracy_pct']:>18.2f}% | {sum_b['top5_accuracy_pct'] - sum_a['top5_accuracy_pct']:>+10.2f}%", flush=True)
    print(f"{'FAR':<35} | {sum_a['far_pct']:>18.4f}% | {sum_b['far_pct']:>18.4f}% | {sum_b['far_pct'] - sum_a['far_pct']:>+10.4f}%", flush=True)
    print(f"{'FRR':<35} | {sum_a['frr_pct']:>18.2f}% | {sum_b['frr_pct']:>18.2f}% | {sum_b['frr_pct'] - sum_a['frr_pct']:>+10.2f}%", flush=True)
    print(f"{'Mean genuine score':<35} | {sum_a['genuine_mean_score']:>20.2f} | {sum_b['genuine_mean_score']:>20.2f} | {sum_b['genuine_mean_score'] - sum_a['genuine_mean_score']:>+10.2f}", flush=True)
    print(f"{'Mean impostor score':<35} | {sum_a['impostor_mean_score']:>20.2f} | {sum_b['impostor_mean_score']:>20.2f} | {sum_b['impostor_mean_score'] - sum_a['impostor_mean_score']:>+10.2f}", flush=True)
    print(f"{'Successful query extractions':<35} | {sum_a['successful_extractions']:>20} | {sum_b['successful_extractions']:>20} | {sum_b['successful_extractions'] - sum_a['successful_extractions']:>+10}", flush=True)
    print(f"{'Failed query extractions':<35} | {sum_a['failed_extractions']:>20} | {sum_b['failed_extractions']:>20} | {sum_b['failed_extractions'] - sum_a['failed_extractions']:>+10}", flush=True)
    print("=" * 85, flush=True)
    print(f"\nAll artifacts successfully saved to: {OUTPUT_DIR}\n", flush=True)


if __name__ == "__main__":
    main()
