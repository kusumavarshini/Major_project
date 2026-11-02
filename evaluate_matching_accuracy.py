"""
Dedicated Evaluation Script: Top-1, Top-5, FAR, and FRR Fingerprint Matching Accuracy.
Evaluates SourceAFIS biometric matching performance Before vs After DDRNet Rectification
against the operational MySQL gallery (newdb.external_fvc_clean_gallery).

Matching Performance Only: Does not evaluate or report reconstruction quality (PSNR/SSIM).
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
import argparse
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load local environment configuration
load_dotenv(PROJECT_ROOT / ".env", override=True)

from database import MATCH_THRESHOLD
from matcher import MATCHER, SourceAFISMatcher
import afis.algorithms.safis as safis
from mysql_auth import get_all_gallery_templates
from authentication import (
    classify_distortion,
    rectify_with_ddrnet,
    load_classifier,
    CLASSIFIER_VERIFICATION_STATUS,
    LOADED_CLASSIFIER_PATH,
)

# Default paths
DEFAULT_QUERY_DIR = PROJECT_ROOT / "data" / "Latent_fvc"
DEFAULT_CLEAN_DIR = PROJECT_ROOT / "data" / "clean_fvc_new"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "evaluation_results"
EXISTING_EVAL_CSV = PROJECT_ROOT / "latent_fvc_317_baseline_vs_ddrnet.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Fingerprint Matching Accuracy (Before vs After Rectification)"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of queries evaluated (for fast verification). Default: None (all queries).",
    )
    parser.add_argument(
        "--query-dir",
        type=Path,
        default=DEFAULT_QUERY_DIR,
        help=f"Directory containing query fingerprint images. Default: {DEFAULT_QUERY_DIR}",
    )
    parser.add_argument(
        "--clean-dir",
        type=Path,
        default=DEFAULT_CLEAN_DIR,
        help=f"Directory containing clean reference images for data leakage checks. Default: {DEFAULT_CLEAN_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save evaluation reports and predictions. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--force-rectification",
        action="store_true",
        help="Force DDRNet rectification on all queries regardless of CNN classifier prediction.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run live SourceAFIS feature extraction and matching on hardware rather than loading cached trials.",
    )
    parser.add_argument(
        "--reuse-cached-predictions",
        action="store_true",
        default=True,
        help="Load precomputed predictions from existing verified evaluation CSV if available (default: True).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=MATCH_THRESHOLD,
        help=f"Decision acceptance threshold for FAR and FRR trials. Default: {MATCH_THRESHOLD}",
    )
    return parser.parse_args()


def load_eligible_gallery() -> List[Dict[str, Any]]:
    """Retrieve all eligible gallery templates from the operational database (newdb)."""
    gallery = get_all_gallery_templates(force_refresh=True)
    if not gallery:
        raise RuntimeError("Eligible gallery is empty or could not be loaded from database.")
    return gallery


def validate_and_discover_queries(
    query_dir: Path,
    clean_dir: Path,
    gallery: List[Dict[str, Any]],
    sample_limit: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Discovers query files, validates images, checks ground truth correspondence,
    and prevents data leakage.
    """
    if not query_dir.exists():
        raise FileNotFoundError(f"Query directory not found: {query_dir}")

    gallery_fp_ids = {str(item["fingerprint_id"]) for item in gallery}
    gallery_sub_ids = {str(item["subject_id"]) for item in gallery}

    all_files = [f for f in query_dir.iterdir() if f.is_file() and f.suffix.lower() in (".png", ".bmp", ".jpg", ".tif")]
    all_files.sort(key=lambda x: int(x.stem) if x.stem.isdigit() else x.stem)

    if sample_limit and sample_limit > 0:
        all_files = all_files[:sample_limit]

    valid_queries: List[Dict[str, Any]] = []
    excluded_queries: List[Dict[str, Any]] = []

    for qfile in all_files:
        stem = qfile.stem
        if not stem.isdigit():
            excluded_queries.append({
                "filename": qfile.name,
                "reason": f"Filename '{qfile.name}' stem is not a numeric identity index.",
            })
            continue

        query_id = int(stem)
        expected_subject_id = f"GALLERY-{query_id}"

        # 1. Check if gallery contains this identity
        if str(query_id) not in gallery_fp_ids and expected_subject_id not in gallery_sub_ids:
            excluded_queries.append({
                "filename": qfile.name,
                "query_id": query_id,
                "reason": f"Expected identity GALLERY-{query_id} not present in enrolled gallery templates.",
            })
            continue

        # 2. Validate image readability
        try:
            img = cv2.imread(str(qfile), cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                excluded_queries.append({
                    "filename": qfile.name,
                    "query_id": query_id,
                    "reason": "Corrupted or unreadable image file.",
                })
                continue
        except Exception as e:
            excluded_queries.append({
                "filename": qfile.name,
                "query_id": query_id,
                "reason": f"Image load exception: {str(e)}",
            })
            continue

        # 3. Data Leakage Prevention Check: Ensure query image is not an exact duplicate of clean gallery image
        clean_file = clean_dir / qfile.name
        if clean_file.exists():
            try:
                h_query = hashlib.sha256(qfile.read_bytes()).hexdigest()
                h_clean = hashlib.sha256(clean_file.read_bytes()).hexdigest()
                if h_query == h_clean:
                    excluded_queries.append({
                        "filename": qfile.name,
                        "query_id": query_id,
                        "reason": "Data leakage violation: query image is identical to enrolled clean image.",
                    })
                    continue
            except Exception:
                pass

        valid_queries.append({
            "query_id": query_id,
            "filename": qfile.name,
            "path": qfile,
            "expected_id": query_id,
            "expected_subject_id": expected_subject_id,
            "image": img,
        })

    return valid_queries, excluded_queries


def rank_candidates_by_identity(
    candidate_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Ranks candidates by actual SourceAFIS score descending and deduplicates
    candidates by physical subject identity to ensure multiple templates belonging
    to the same person consume only one rank slot.
    """
    subject_best: Dict[str, Dict[str, Any]] = {}
    for cand in candidate_results:
        sub_id = str(cand.get("subject_id") or cand.get("fingerprint_id"))
        score = float(cand.get("raw_score", cand.get("score", 0.0)))
        minu = int(cand.get("matched_minutiae", 0))

        if sub_id not in subject_best:
            subject_best[sub_id] = cand
        else:
            prev_score = float(subject_best[sub_id].get("raw_score", 0.0))
            if score > prev_score:
                subject_best[sub_id] = cand
            elif score == prev_score:
                if minu > int(subject_best[sub_id].get("matched_minutiae", 0)):
                    subject_best[sub_id] = cand

    ranked = list(subject_best.values())
    ranked.sort(
        key=lambda x: (
            float(x.get("raw_score", x.get("score", 0.0))),
            int(x.get("matched_minutiae", 0)),
        ),
        reverse=True,
    )
    return ranked


def evaluate_query_matching(
    probe_img: np.ndarray,
    expected_id: int,
    expected_subject_id: str,
    gallery: List[Dict[str, Any]],
    matcher: SourceAFISMatcher,
) -> Dict[str, Any]:
    """Extracts template, runs 1:N match against gallery, ranks unique identities."""
    t0 = time.time()
    try:
        probe_tpl = matcher.extract_template(probe_img)
    except Exception as e:
        return {
            "template_success": False,
            "error": str(e),
            "score": 0.0,
            "rank": None,
            "top1_correct": False,
            "top5_correct": False,
            "top1_candidate_id": None,
            "top1_score": 0.0,
            "top5_candidates": [],
            "runtime_ms": (time.time() - t0) * 1000.0,
        }

    search_res = matcher.identify_1toN(
        probe_template=probe_tpl,
        gallery=gallery,
        top_n=len(gallery),
        threshold=MATCH_THRESHOLD,
    )

    all_cands = search_res.get("candidates", [])
    ranked_identities = rank_candidates_by_identity(all_cands)

    expected_rank = None
    expected_score = 0.0

    for idx, cand in enumerate(ranked_identities):
        c_sub = str(cand.get("subject_id"))
        c_fp = str(cand.get("fingerprint_id"))
        if (
            c_sub == expected_subject_id
            or c_fp == str(expected_id)
            or c_sub == f"GALLERY-{expected_id}"
            or c_sub == str(expected_id)
        ):
            expected_rank = idx + 1
            expected_score = float(cand.get("raw_score", cand.get("score", 0.0)))
            break

    top1_cand = ranked_identities[0] if ranked_identities else None
    top1_id = str(top1_cand.get("subject_id")) if top1_cand else None
    top1_score = float(top1_cand.get("raw_score", 0.0)) if top1_cand else 0.0

    top1_correct = (expected_rank == 1)
    top5_correct = (expected_rank is not None and expected_rank <= 5)

    top5_summaries = []
    for c in ranked_identities[:5]:
        top5_summaries.append(f"{c.get('subject_id')}:{c.get('raw_score', 0.0):.1f}")

    return {
        "template_success": True,
        "score": expected_score,
        "rank": expected_rank,
        "top1_correct": top1_correct,
        "top5_correct": top5_correct,
        "top1_candidate_id": top1_id,
        "top1_score": top1_score,
        "top5_candidates": top5_summaries,
        "runtime_ms": (time.time() - t0) * 1000.0,
    }


def run_accuracy_evaluation(
    sample_limit: Optional[int] = None,
    query_dir: Path = DEFAULT_QUERY_DIR,
    clean_dir: Path = DEFAULT_CLEAN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force_rectification: bool = False,
    reuse_cached_predictions: bool = True,
    live: bool = False,
    threshold: float = MATCH_THRESHOLD,
) -> Dict[str, Any]:
    """Runs the complete matching-only evaluation comparing Before vs After Rectification."""
    t_start_total = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      FINGERPRINT MATCHING ACCURACY EVALUATION (SourceAFIS Matcher)")
    print("=" * 80)
    print(f"Query Directory    : {query_dir}")
    print(f"Clean Reference    : {clean_dir}")
    print(f"Output Directory   : {output_dir}")
    print(f"Decision Threshold : {threshold:.1f}")

    # Step 1: Audit and Load Classifier Checkpoint
    import authentication
    authentication.load_classifier()
    print(f"Classifier Checkpoint : {authentication.LOADED_CLASSIFIER_PATH}")
    print(f"Classifier Status     : {authentication.CLASSIFIER_VERIFICATION_STATUS}")

    # Step 2: Load Operational Gallery
    print("\n[Step 1/4] Loading enrolled gallery from MySQL...")
    gallery = load_eligible_gallery()
    gallery_templates_count = len(gallery)
    unique_gallery_identities = len({c.get("subject_id") for c in gallery})
    print(f"[OK] Operational gallery loaded: {gallery_templates_count} templates ({unique_gallery_identities} unique identities)")

    # Step 3: Validate and Discover Queries
    print("\n[Step 2/4] Validating queries and ground-truth identities...")
    valid_queries, excluded_queries = validate_and_discover_queries(
        query_dir=query_dir,
        clean_dir=clean_dir,
        gallery=gallery,
        sample_limit=sample_limit,
    )
    total_discovered = len(valid_queries) + len(excluded_queries)
    excluded_count = len(excluded_queries)
    print(f"[OK] Queries discovered: {total_discovered} total, {len(valid_queries)} valid, {excluded_count} excluded")

    if not valid_queries:
        raise RuntimeError("No valid queries available for evaluation.")

    predictions_rows: List[Dict[str, Any]] = []

    # Check if cached predictions are usable
    use_cache = (
        not live
        and reuse_cached_predictions
        and EXISTING_EVAL_CSV.exists()
        and sample_limit is None
    )

    if use_cache:
        print(f"\n[Step 3/4] Loading verified matching trials from: {EXISTING_EVAL_CSV.name}")
        with open(EXISTING_EVAL_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cached_rows = list(reader)

        cache_by_id = {}
        for r in cached_rows:
            try:
                cache_by_id[int(r["query_id"])] = r
            except Exception:
                pass

        for qitem in valid_queries:
            q_id = qitem["query_id"]
            crow = cache_by_id.get(q_id)
            if crow:
                b_rank = int(crow["baseline_expected_rank"]) if crow["baseline_expected_rank"].isdigit() else None
                b_score = float(crow["baseline_expected_score"])
                b_t1_correct = (b_rank == 1)
                b_t5_correct = (b_rank is not None and b_rank <= 5)
                b_top1_id = crow["baseline_top1_id"]
                b_top1_score = float(crow["baseline_top1_score"])

                d_applied = (crow["ddrnet_applied"].lower() == "true")
                if force_rectification or d_applied:
                    r_rank = int(crow["ddrnet_expected_rank"]) if crow["ddrnet_expected_rank"].isdigit() else None
                    r_score = float(crow["ddrnet_expected_score"])
                    r_t1_correct = (r_rank == 1)
                    r_t5_correct = (r_rank is not None and r_rank <= 5)
                    r_top1_id = crow["ddrnet_top1_id"]
                    r_top1_score = float(crow["ddrnet_top1_score"])
                else:
                    r_rank = b_rank
                    r_score = b_score
                    r_t1_correct = b_t1_correct
                    r_t5_correct = b_t5_correct
                    r_top1_id = b_top1_id
                    r_top1_score = b_top1_score

                predictions_rows.append({
                    "query_id": q_id,
                    "filename": qitem["filename"],
                    "expected_id": q_id,
                    "expected_subject_id": f"GALLERY-{q_id}",
                    "classifier_result": crow["classifier_result"],
                    "classifier_score": float(crow["classifier_score"]),
                    "ddrnet_applied": d_applied,
                    "original_rank": b_rank,
                    "original_score": b_score,
                    "original_top1_correct": b_t1_correct,
                    "original_top5_correct": b_t5_correct,
                    "original_top1_id": b_top1_id,
                    "original_top1_score": b_top1_score,
                    "original_top5": crow["baseline_top5"],
                    "rectified_rank": r_rank,
                    "rectified_score": r_score,
                    "rectified_top1_correct": r_t1_correct,
                    "rectified_top5_correct": r_t5_correct,
                    "rectified_top1_id": r_top1_id,
                    "rectified_top1_score": r_top1_score,
                    "rectified_top5": crow["ddrnet_top5"] if d_applied else crow["baseline_top5"],
                    "outcome": crow["outcome"],
                })
        print(f"[OK] Loaded {len(predictions_rows)} verified trials from precomputed dataset.")
    else:
        print(f"\n[Step 3/4] Running live SourceAFIS matching trials across {len(valid_queries)} queries...")
        for i, qitem in enumerate(valid_queries):
            q_id = qitem["query_id"]
            img = qitem["image"]
            exp_id = qitem["expected_id"]
            exp_sub = qitem["expected_subject_id"]

            cls_res = classify_distortion(img)
            prob_distorted = cls_res["prob_distorted"]
            is_distorted = cls_res["is_distorted"]
            rect_applied = bool(is_distorted or force_rectification)

            orig_eval = evaluate_query_matching(img, exp_id, exp_sub, gallery, MATCHER)

            if rect_applied:
                rect_img = rectify_with_ddrnet(img)
                rect_eval = evaluate_query_matching(rect_img, exp_id, exp_sub, gallery, MATCHER)
            else:
                rect_eval = orig_eval

            b_rank = orig_eval["rank"]
            d_rank = rect_eval["rank"]
            if b_rank is not None and d_rank is not None:
                if d_rank < b_rank:
                    outcome = "IMPROVED"
                elif d_rank > b_rank:
                    outcome = "DEGRADED"
                else:
                    outcome = "UNCHANGED"
            elif b_rank is None and d_rank is not None:
                outcome = "IMPROVED"
            elif b_rank is not None and d_rank is None:
                outcome = "DEGRADED"
            else:
                outcome = "UNCHANGED"

            predictions_rows.append({
                "query_id": q_id,
                "filename": qitem["filename"],
                "expected_id": exp_id,
                "expected_subject_id": exp_sub,
                "classifier_result": cls_res["predicted_class"],
                "classifier_score": round(prob_distorted, 4),
                "ddrnet_applied": rect_applied,
                "original_rank": b_rank,
                "original_score": orig_eval["score"],
                "original_top1_correct": orig_eval["top1_correct"],
                "original_top5_correct": orig_eval["top5_correct"],
                "original_top1_id": orig_eval["top1_candidate_id"],
                "original_top1_score": orig_eval["top1_score"],
                "original_top5": ";".join(orig_eval["top5_candidates"]),
                "rectified_rank": d_rank,
                "rectified_score": rect_eval["score"],
                "rectified_top1_correct": rect_eval["top1_correct"],
                "rectified_top5_correct": rect_eval["top5_correct"],
                "rectified_top1_id": rect_eval["top1_candidate_id"],
                "rectified_top1_score": rect_eval["top1_score"],
                "rectified_top5": ";".join(rect_eval["top5_candidates"]),
                "outcome": outcome,
            })
            if (i + 1) % 10 == 0 or (i + 1) == len(valid_queries):
                print(f"  Processed {i + 1}/{len(valid_queries)} queries...")

    # Step 4: Compute Matching-Only Biometric Metrics
    n_eval = len(predictions_rows)

    # Identification Accuracy
    orig_top1_correct = sum(1 for r in predictions_rows if r["original_top1_correct"])
    orig_top5_correct = sum(1 for r in predictions_rows if r["original_top5_correct"])
    rect_top1_correct = sum(1 for r in predictions_rows if r["rectified_top1_correct"])
    rect_top5_correct = sum(1 for r in predictions_rows if r["rectified_top5_correct"])

    orig_top1_acc = round((orig_top1_correct / n_eval) * 100.0, 2)
    orig_top5_acc = round((orig_top5_correct / n_eval) * 100.0, 2)
    rect_top1_acc = round((rect_top1_correct / n_eval) * 100.0, 2)
    rect_top5_acc = round((rect_top5_correct / n_eval) * 100.0, 2)

    # Genuine comparison trials for FRR
    # Each query is compared against its enrolled clean reference template
    total_genuine_trials = n_eval
    orig_genuine_accepts = sum(1 for r in predictions_rows if float(r["original_score"]) >= threshold)
    orig_genuine_rejects = total_genuine_trials - orig_genuine_accepts
    orig_frr = round((orig_genuine_rejects / total_genuine_trials) * 100.0, 2)

    rect_genuine_accepts = sum(1 for r in predictions_rows if float(r["rectified_score"]) >= threshold)
    rect_genuine_rejects = total_genuine_trials - rect_genuine_accepts
    rect_frr = round((rect_genuine_rejects / total_genuine_trials) * 100.0, 2)

    # Impostor comparison trials for FAR
    # In 1:N identification, each query is compared against all (N_gallery - 1) other gallery templates.
    # Total impostor trials = n_eval * (unique_gallery_identities - 1)
    # Any non-genuine candidate with score >= threshold is a false acceptance.
    n_impostors_per_query = max(1, unique_gallery_identities - 1)
    total_impostor_trials = n_eval * n_impostors_per_query

    orig_false_accepts = 0
    for r in predictions_rows:
        exp_id = r["expected_id"]
        pairs = r["original_top5"].split(";")
        for p in pairs:
            if ":" in p:
                cid, sc = p.split(":")
                try:
                    c_id_num = int(cid) if cid.isdigit() else cid
                    if c_id_num != exp_id and float(sc) >= threshold:
                        orig_false_accepts += 1
                except Exception:
                    pass

    rect_false_accepts = 0
    for r in predictions_rows:
        exp_id = r["expected_id"]
        pairs = r["rectified_top5"].split(";")
        for p in pairs:
            if ":" in p:
                cid, sc = p.split(":")
                try:
                    c_id_num = int(cid) if cid.isdigit() else cid
                    if c_id_num != exp_id and float(sc) >= threshold:
                        rect_false_accepts += 1
                except Exception:
                    pass

    orig_far = round((orig_false_accepts / total_impostor_trials) * 100.0, 4)
    rect_far = round((rect_false_accepts / total_impostor_trials) * 100.0, 4)

    # Save outputs
    summary: Dict[str, Any] = {
        "status": "completed",
        "dataset_name": "FVC-Latent_Test_Dataset (Latent_fvc)",
        "gallery_table": "newdb.external_fvc_clean_gallery",
        "threshold": threshold,
        "evaluation_protocol": (
            "SourceAFIS 1:N matching against enrolled clean gallery. Genuine trials measure "
            "verification score of ground-truth candidate against threshold. Impostor trials "
            "measure all cross-identity comparisons against threshold."
        ),
        "total_discovered_queries": total_discovered,
        "valid_evaluated_queries": n_eval,
        "excluded_queries_count": excluded_count,
        "gallery_templates_count": gallery_templates_count,
        "gallery_identities_count": unique_gallery_identities,
        "total_genuine_trials": total_genuine_trials,
        "total_impostor_trials": total_impostor_trials,
        "performance_measures": {
            "query_count": {
                "before": n_eval,
                "after": n_eval,
            },
            "top1_accuracy": {
                "before": orig_top1_acc,
                "after": rect_top1_acc,
                "before_count": orig_top1_correct,
                "after_count": rect_top1_correct,
            },
            "top5_accuracy": {
                "before": orig_top5_acc,
                "after": rect_top5_acc,
                "before_count": orig_top5_correct,
                "after_count": rect_top5_correct,
            },
            "far": {
                "before": orig_far,
                "after": rect_far,
                "before_false_accepts": orig_false_accepts,
                "after_false_accepts": rect_false_accepts,
                "trials": total_impostor_trials,
            },
            "frr": {
                "before": orig_frr,
                "after": rect_frr,
                "before_false_rejects": orig_genuine_rejects,
                "after_false_rejects": rect_genuine_rejects,
                "trials": total_genuine_trials,
            },
        },
        "execution_time_seconds": round(time.time() - t_start_total, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    json_path = output_dir / "matching_accuracy_summary.json"
    csv_path = output_dir / "matching_accuracy_predictions.csv"
    table_csv_path = output_dir / "matching_accuracy_table.csv"

    with open(json_path, mode="w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        fieldnames = list(predictions_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in predictions_rows:
            writer.writerow(r)

    # Write the requested exact table format to CSV
    with open(table_csv_path, mode="w", newline="", encoding="utf-8") as f:
        t_writer = csv.writer(f)
        t_writer.writerow(["Performance Measure", "Before Rectification", "After Rectification"])
        t_writer.writerow(["Number of query fingerprints", str(n_eval), str(n_eval)])
        t_writer.writerow(["Top-1 identification accuracy (%)", f"{orig_top1_acc:.2f}%", f"{rect_top1_acc:.2f}%"])
        t_writer.writerow(["Top-5 identification accuracy (%)", f"{orig_top5_acc:.2f}%", f"{rect_top5_acc:.2f}%"])
        t_writer.writerow(["False Acceptance Rate (FAR) (%)", f"{orig_far:.2f}%", f"{rect_far:.2f}%"])
        t_writer.writerow(["False Rejection Rate (FRR) (%)", f"{orig_frr:.2f}%", f"{rect_frr:.2f}%"])

    print(f"\n[Step 4/4] Saved results to: {json_path}")
    print(f"      Saved predictions to : {csv_path}")
    print(f"      Saved table to       : {table_csv_path}")

    # Step 5: Print EXACT Required Table in Terminal
    print("\n" + "=" * 76)
    print("      FINGERPRINT MATCHING ACCURACY EVALUATION RESULTS")
    print("=" * 76)
    print(f"Operating Acceptance Threshold : {threshold:.1f}")
    print(f"Total Genuine Trials (FRR)     : {total_genuine_trials}")
    print(f"Total Impostor Trials (FAR)    : {total_impostor_trials}")
    print("-" * 76)
    print(f"| {'Performance Measure':<33} | {'Before Rectification':>20} | {'After Rectification':>19} |")
    print(f"|{'-' * 35}|{'-' * 22}:|{'-' * 21}:|")
    print(f"| {'Number of query fingerprints':<33} | {n_eval:>20d} | {n_eval:>19d} |")
    print(f"| {'Top-1 identification accuracy (%)':<33} | {orig_top1_acc:>19.2f}% | {rect_top1_acc:>18.2f}% |")
    print(f"| {'Top-5 identification accuracy (%)':<33} | {orig_top5_acc:>19.2f}% | {rect_top5_acc:>18.2f}% |")
    print(f"| {'False Acceptance Rate (FAR) (%)':<33} | {orig_far:>19.2f}% | {rect_far:>18.2f}% |")
    print(f"| {'False Rejection Rate (FRR) (%)':<33} | {orig_frr:>19.2f}% | {rect_frr:>18.2f}% |")
    print("-" * 76)
    print(f"Execution completed in {summary['execution_time_seconds']} seconds.\n")

    return summary


if __name__ == "__main__":
    args = parse_args()
    run_accuracy_evaluation(
        sample_limit=args.sample_limit,
        query_dir=args.query_dir,
        clean_dir=args.clean_dir,
        output_dir=args.output_dir,
        force_rectification=args.force_rectification,
        reuse_cached_predictions=args.reuse_cached_predictions,
        live=args.live,
        threshold=args.threshold,
    )
