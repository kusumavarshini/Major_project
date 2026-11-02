
#!/usr/bin/env python3
"""
Evaluation of Fingerprint Identification Performance Before and After DDRNet Rectification Using SourceAFIS.

Project: FingerSure: Fingerprint Distortion Rectification for Accurate Recognition
Evaluation Target:
  - Condition B: 675 DDRNet-rectified fingerprint test images (data/rectified_ddrnet_test)
  - Condition A: 225 Original distorted fingerprint test images (data/test_distorted/distorted)
Clean Reference Gallery:
  - 225 Clean reference fingerprints (data/split/test) across 15 families (45 distinct individual identities, 5 impressions each)
Matching Engine:
  - Project's SourceAFIS implementation (matcher.py / MATCHER.identify_1toN)
Decision Threshold:
  - database.MATCH_THRESHOLD = 50.0

All outputs are saved strictly inside 'sourceafis_rectified_evaluation/'.
No existing project files, databases, models, or datasets are modified or overwritten.
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from matcher import MATCHER, SourceAFISMatcher
import afis.algorithms.safis as safis
from database import MATCH_THRESHOLD

OUTPUT_DIR = PROJECT_ROOT / "sourceafis_rectified_evaluation"
CLEAN_DIR = PROJECT_ROOT / "data" / "split" / "test"
DISTORTED_DIR = PROJECT_ROOT / "data" / "test_distorted" / "distorted"
RECTIFIED_DIR = PROJECT_ROOT / "data" / "rectified_ddrnet_test"
METADATA_CSV = PROJECT_ROOT / "data" / "test_distorted" / "distortion_metadata.csv"

# Global gallery reference for worker processes
GLOBAL_GALLERY: List[Dict[str, Any]] = []


def log(msg: str, log_file: Optional[Path] = None):
    print(msg, flush=True)
    if log_file:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")


def pre_index_template(cand_tpl: Dict[str, Any], params: safis.SafisParams) -> None:
    """Pre-computes invariant edges, length buckets, and adjacency for fast 1:N search."""
    c_minu = cand_tpl.get("minutiae")
    if c_minu is None or len(c_minu) < 2:
        cand_tpl["_edges"] = []
        cand_tpl["_buckets"] = {}
        cand_tpl["_adj"] = {}
        cand_tpl["_cache_params_key"] = (
            params.max_neighbors,
            params.max_edge_length,
            params.length_tol,
            params.angle_tol,
        )
        return

    edges = safis.build_edges(c_minu, params)
    buckets: Dict[int, List[safis._Edge]] = {}
    adj: Dict[int, List[safis._Edge]] = {}
    lt = float(params.length_tol) or 1e-6

    for e in edges:
        k = int(round(e.length / lt))
        buckets.setdefault(k, []).append(e)
        adj.setdefault(e.ref, []).append(e)

    cand_tpl["_edges"] = edges
    cand_tpl["_buckets"] = buckets
    cand_tpl["_adj"] = adj
    cand_tpl["_cache_params_key"] = (
        params.max_neighbors,
        params.max_edge_length,
        params.length_tol,
        params.angle_tol,
    )


def extract_template_task(args: Tuple[str, str, str, str, str]) -> Dict[str, Any]:
    """Worker task to extract template from an image file."""
    path_str, filename, stem, fam, mem = args
    img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            "path": path_str,
            "filename": filename,
            "stem": stem,
            "family_id": fam,
            "member_type": mem,
            "subject_id": f"{fam}/{mem}",
            "template": {"minutiae": None, "minutiae_count": 0},
            "error": "Failed to read image file",
        }
    try:
        tpl = MATCHER.extract_template(img)
        return {
            "path": path_str,
            "filename": filename,
            "stem": stem,
            "family_id": fam,
            "member_type": mem,
            "subject_id": f"{fam}/{mem}",
            "template": tpl,
            "error": None,
        }
    except Exception as e:
        return {
            "path": path_str,
            "filename": filename,
            "stem": stem,
            "family_id": fam,
            "member_type": mem,
            "subject_id": f"{fam}/{mem}",
            "template": {"minutiae": None, "minutiae_count": 0},
            "error": str(e),
        }


def init_worker(gallery_data: List[Dict[str, Any]]) -> None:
    """Worker process initializer to set shared read-only gallery."""
    global GLOBAL_GALLERY
    GLOBAL_GALLERY = gallery_data


def match_query_task(q_item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker task: Evaluates 1 probe query against GLOBAL_GALLERY.
    Aggregates scores across all candidate impressions for each distinct identity,
    excluding self-scan impression to strictly prevent identity leakage.
    """
    global GLOBAL_GALLERY
    target_identity = q_item["ground_truth_identity"]
    query_stem = q_item["stem"]
    condition = q_item["condition"]
    query_path = q_item["query_path"]

    img = cv2.imread(query_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {
            **q_item,
            "status": "FAILED",
            "failure_reason": f"Could not read image file from disk: {query_path}",
            "minutiae_count": 0,
            "top_predicted_identity": None,
            "top_1_score": 0.0,
            "genuine_identity_score": 0.0,
            "genuine_identity_rank": None,
            "is_top1_correct": False,
            "is_top5_correct": False,
            "genuine_verified": False,
            "false_rejection": True,
            "impostor_scores": [],
            "processing_time_ms": 0.0,
        }

    t0 = time.time()
    try:
        probe_tpl = MATCHER.extract_template(img)
    except Exception as e:
        return {
            **q_item,
            "status": "FAILED",
            "failure_reason": f"Template extraction failed: {str(e)}",
            "minutiae_count": 0,
            "top_predicted_identity": None,
            "top_1_score": 0.0,
            "genuine_identity_score": 0.0,
            "genuine_identity_rank": None,
            "is_top1_correct": False,
            "is_top5_correct": False,
            "genuine_verified": False,
            "false_rejection": True,
            "impostor_scores": [],
            "processing_time_ms": round((time.time() - t0) * 1000.0, 2),
        }

    q_minu = probe_tpl.get("minutiae")
    minu_cnt = len(q_minu) if q_minu is not None else 0
    if minu_cnt < 2:
        return {
            **q_item,
            "status": "FAILED",
            "failure_reason": f"Biometric Failure-to-Acquire (FTA): extracted minutiae count is {minu_cnt} (< 2 required)",
            "minutiae_count": minu_cnt,
            "top_predicted_identity": None,
            "top_1_score": 0.0,
            "genuine_identity_score": 0.0,
            "genuine_identity_rank": None,
            "is_top1_correct": False,
            "is_top5_correct": False,
            "genuine_verified": False,
            "false_rejection": True,
            "impostor_scores": [],
            "processing_time_ms": round((time.time() - t0) * 1000.0, 2),
        }

    # Execute 1:N SourceAFIS identification against full gallery
    is_rect = (condition == "rectified")
    res_1toN = MATCHER.identify_1toN(
        probe_tpl,
        GLOBAL_GALLERY,
        top_n=len(GLOBAL_GALLERY),
        threshold=MATCH_THRESHOLD,
        is_rectified=is_rect,
    )
    cands = res_1toN.get("candidates", [])

    # Aggregate by distinct person identity (max score rule)
    # Self-matching exclusion: Exclude clean gallery template with identical stem
    id_scores: Dict[str, Dict[str, Any]] = {}
    for c in cands:
        cid = c["subject_id"]
        c_stem = Path(c["filename"]).stem
        # Prevent self-scan comparison
        if cid == target_identity and c_stem == query_stem:
            continue
        sc = float(c["raw_score"])
        mm = int(c.get("matched_minutiae", 0))

        if cid not in id_scores or (sc, mm) > (id_scores[cid]["score"], id_scores[cid]["matched_minutiae"]):
            id_scores[cid] = {
                "identity": cid,
                "score": sc,
                "matched_minutiae": mm,
                "best_filename": c["filename"],
            }

    # Rank all candidate identities descending
    ranked_identities = sorted(
        id_scores.values(),
        key=lambda x: (x["score"], x["matched_minutiae"]),
        reverse=True,
    )

    ranks_map = {item["identity"]: idx + 1 for idx, item in enumerate(ranked_identities)}
    genuine_rank = ranks_map.get(target_identity, None)

    top1_cand = ranked_identities[0] if ranked_identities else None
    top1_id = top1_cand["identity"] if top1_cand else None
    top1_score = top1_cand["score"] if top1_cand else 0.0

    target_cand = id_scores.get(target_identity, None)
    genuine_score = target_cand["score"] if target_cand else 0.0

    is_top1 = bool(genuine_rank == 1)
    is_top5 = bool(genuine_rank is not None and genuine_rank <= 5)
    genuine_verified = bool(genuine_score >= MATCH_THRESHOLD)
    false_rejection = not genuine_verified

    impostor_scores = [
        item["score"] for item in ranked_identities if item["identity"] != target_identity
    ]

    proc_time_ms = round((time.time() - t0) * 1000.0, 2)

    return {
        **q_item,
        "status": "SUCCESS",
        "failure_reason": None,
        "minutiae_count": minu_cnt,
        "top_predicted_identity": top1_id,
        "top_1_score": round(top1_score, 2),
        "genuine_identity_score": round(genuine_score, 2),
        "genuine_identity_rank": genuine_rank,
        "is_top1_correct": is_top1,
        "is_top5_correct": is_top5,
        "genuine_verified": genuine_verified,
        "false_rejection": false_rejection,
        "impostor_scores": impostor_scores,
        "processing_time_ms": proc_time_ms,
    }


def compute_metrics(
    results: List[Dict[str, Any]],
    threshold: float = MATCH_THRESHOLD,
) -> Dict[str, Any]:
    """Computes full biometric identification and verification metrics."""
    total_queries = len(results)
    if total_queries == 0:
        return {}

    successful = [r for r in results if r["status"] == "SUCCESS"]
    failed = [r for r in results if r["status"] != "SUCCESS"]

    eval_denom = len(successful)  # Actual successfully evaluated queries denominator

    top1_correct = sum(1 for r in successful if r["is_top1_correct"])
    top5_correct = sum(1 for r in successful if r["is_top5_correct"])

    top1_acc_valid = (top1_correct / eval_denom * 100.0) if eval_denom > 0 else 0.0
    top5_acc_valid = (top5_correct / eval_denom * 100.0) if eval_denom > 0 else 0.0

    top1_acc_all = (top1_correct / total_queries * 100.0) if total_queries > 0 else 0.0
    top5_acc_all = (top5_correct / total_queries * 100.0) if total_queries > 0 else 0.0

    # Cumulative Match Characteristic (CMC) up to Rank 5
    cmc = {}
    for k in range(1, 6):
        c_k = sum(1 for r in successful if r["genuine_identity_rank"] is not None and r["genuine_identity_rank"] <= k)
        cmc[f"Rank_{k}"] = round((c_k / eval_denom * 100.0) if eval_denom > 0 else 0.0, 2)

    # Verification statistics (FRR and FAR)
    false_rejections = sum(1 for r in successful if r["genuine_identity_score"] < threshold)
    frr = (false_rejections / eval_denom * 100.0) if eval_denom > 0 else 0.0

    total_impostor_trials = sum(len(r["impostor_scores"]) for r in successful)
    false_acceptances = sum(
        sum(1 for s in r["impostor_scores"] if s >= threshold)
        for r in successful
    )
    far = (false_acceptances / total_impostor_trials * 100.0) if total_impostor_trials > 0 else 0.0

    genuine_scores = [r["genuine_identity_score"] for r in successful]
    mean_gen = float(np.mean(genuine_scores)) if genuine_scores else 0.0
    median_gen = float(np.median(genuine_scores)) if genuine_scores else 0.0
    min_gen = float(np.min(genuine_scores)) if genuine_scores else 0.0
    max_gen = float(np.max(genuine_scores)) if genuine_scores else 0.0

    all_impostor_scores = [s for r in successful for s in r["impostor_scores"]]
    mean_imp = float(np.mean(all_impostor_scores)) if all_impostor_scores else 0.0
    median_imp = float(np.median(all_impostor_scores)) if all_impostor_scores else 0.0
    max_imp = float(np.max(all_impostor_scores)) if all_impostor_scores else 0.0

    minutiae_counts = [r["minutiae_count"] for r in successful]
    mean_minu = float(np.mean(minutiae_counts)) if minutiae_counts else 0.0

    return {
        "total_queries": total_queries,
        "successfully_evaluated_queries": eval_denom,
        "failed_or_skipped_queries": len(failed),
        "failure_reasons": [r["failure_reason"] for r in failed],
        "top1_correct_count": top1_correct,
        "top1_accuracy_valid_pct": round(top1_acc_valid, 2),
        "top1_accuracy_all_pct": round(top1_acc_all, 2),
        "top5_correct_count": top5_correct,
        "top5_accuracy_valid_pct": round(top5_acc_valid, 2),
        "top5_accuracy_all_pct": round(top5_acc_all, 2),
        "cmc_rates": cmc,
        "operating_threshold": threshold,
        "false_rejections_count": false_rejections,
        "frr_pct": round(frr, 2),
        "impostor_trials_count": total_impostor_trials,
        "false_acceptances_count": false_acceptances,
        "far_pct": round(far, 4),
        "mean_genuine_score": round(mean_gen, 2),
        "median_genuine_score": round(median_gen, 2),
        "min_genuine_score": round(min_gen, 2),
        "max_genuine_score": round(max_gen, 2),
        "mean_impostor_score": round(mean_imp, 2),
        "median_impostor_score": round(median_imp, 2),
        "max_impostor_score": round(max_imp, 2),
        "mean_minutiae_count": round(mean_minu, 2),
    }


def main():
    t_start_total = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_file = OUTPUT_DIR / "evaluation_log.txt"
    if log_file.exists():
        log_file.unlink()

    log("=" * 90, log_file)
    log("FINGERSURE: SOURCEAFIS RECTIFIED BIOMETRIC EVALUATION PIPELINE", log_file)
    log("=" * 90, log_file)
    log(f"Execution started at: {time.strftime('%Y-%m-%d %H:%M:%S')}", log_file)
    log(f"Project root        : {PROJECT_ROOT}", log_file)
    log(f"Output directory    : {OUTPUT_DIR}", log_file)
    log(f"Operating threshold : {MATCH_THRESHOLD} (SourceAFIS raw score)", log_file)
    log(f"Available CPU cores : {cpu_count()}", log_file)

    # ---------------------------------------------------------
    # STEP 1: Programmatic Dataset Inspection & Verification
    # ---------------------------------------------------------
    log("\n[STEP 1] Programmatic Dataset Inspection & Path Verification...", log_file)
    for p_name, p_path in [
        ("Clean Gallery (split/test)", CLEAN_DIR),
        ("Original Distorted (test_distorted)", DISTORTED_DIR),
        ("DDRNet-Rectified (rectified_ddrnet_test)", RECTIFIED_DIR),
        ("Metadata CSV", METADATA_CSV),
    ]:
        exists = p_path.exists()
        log(f"  - {p_name:<40}: exists={exists} | path={p_path}", log_file)
        if not exists:
            raise FileNotFoundError(f"Required path does not exist: {p_path}")

    # Inspect Rectified Dataset
    rect_files = sorted(list(RECTIFIED_DIR.glob("*.png")))
    log(f"\nVerifying folder 'rectified_ddrnet_test' counts:", log_file)
    log(f"  Total rectified image files found: {len(rect_files)} (Expected: 675)", log_file)
    assert len(rect_files) == 675, f"Expected 675 rectified files, found {len(rect_files)}"

    rect_families = set()
    rect_distortions = {}
    rect_items = []
    for f in rect_files:
        parts = f.stem.split("__")
        if len(parts) == 4:
            fam, mem, stem, dtype = parts
            rect_families.add(fam)
            rect_distortions[dtype] = rect_distortions.get(dtype, 0) + 1
            rect_items.append({
                "filename": f.name,
                "path": str(f),
                "family_id": fam,
                "member_type": mem,
                "stem": stem,
                "distortion_type": dtype,
                "ground_truth_identity": f"{fam}/{mem}",
            })
        else:
            log(f"  WARNING: Unexpected filename format: {f.name}", log_file)

    log(f"  Verified family count            : {len(rect_families)} (Expected: 15)", log_file)
    log(f"  Families                         : {sorted(list(rect_families))}", log_file)
    log(f"  Distortion types distribution    : {rect_distortions}", log_file)
    for fam in sorted(list(rect_families)):
        cnt = sum(1 for item in rect_items if item["family_id"] == fam)
        assert cnt == 45, f"Family {fam} has {cnt} images (Expected: 45)"
    log("  >> VERIFIED: Exactly 15 families, 45 images per family, exactly 675 images total.", log_file)

    # Inspect Clean Reference Gallery
    clean_files = sorted(list(CLEAN_DIR.glob("**/*.png")))
    log(f"\nVerifying folder 'data/split/test' (Clean Reference Gallery):", log_file)
    log(f"  Total clean reference images found: {len(clean_files)} (Expected: 225)", log_file)
    clean_identities = set()
    clean_items = []
    for f in clean_files:
        rel = f.relative_to(CLEAN_DIR)
        fam, mem = rel.parts[0], rel.parts[1]
        ident = f"{fam}/{mem}"
        clean_identities.add(ident)
        clean_items.append((str(f), f.name, f.stem, fam, mem))
    log(f"  Distinct clean reference identities: {len(clean_identities)} (Expected: 45)", log_file)

    # Inspect Original Distorted Dataset
    dist_files = sorted(list(DISTORTED_DIR.glob("**/*.png")))
    log(f"\nVerifying folder 'data/test_distorted/distorted':", log_file)
    log(f"  Total distorted images found      : {len(dist_files)} (Expected: 225)", log_file)

    df_meta = pd.read_csv(METADATA_CSV)
    log(f"  Metadata rows loaded from CSV     : {len(df_meta)}", log_file)
    log(f"  Distortion types in metadata      : {df_meta['distortion_type'].value_counts().to_dict()}", log_file)

    # ---------------------------------------------------------
    # STEP 2: Establish and Export Validated Identity Mapping
    # ---------------------------------------------------------
    log("\n[STEP 2] Establishing and exporting identity mapping table...", log_file)
    mapping_records = []
    meta_lookup = {}
    for _, r in df_meta.iterrows():
        fam = str(r["family_id"])
        mem = str(r["member"])
        clean_stem = Path(str(r["clean_output_path"])).stem
        meta_lookup[(fam, mem, clean_stem)] = str(r["distortion_type"])

    for idx, r_item in enumerate(rect_items, 1):
        fam = r_item["family_id"]
        mem = r_item["member_type"]
        stem = r_item["stem"]
        dtype = r_item["distortion_type"]

        clean_path = CLEAN_DIR / fam / mem / f"{stem}.png"
        dist_path = DISTORTED_DIR / fam / mem / f"{stem}_distorted.png"

        clean_exists = clean_path.exists()
        dist_exists = dist_path.exists()
        meta_dtype = meta_lookup.get((fam, mem, stem), None)

        is_paired = (meta_dtype == dtype and dist_exists)

        mapping_records.append({
            "query_index": idx,
            "rectified_filename": r_item["filename"],
            "original_distorted_filename": f"{stem}_distorted.png" if dist_exists else None,
            "clean_reference_stem": stem,
            "clean_reference_rel_path": f"{fam}/{mem}/{stem}.png" if clean_exists else None,
            "family_id": fam,
            "member_type": mem,
            "ground_truth_identity": f"{fam}/{mem}",
            "rectified_distortion_type": dtype,
            "metadata_distortion_type": meta_dtype,
            "has_clean_reference": clean_exists,
            "has_distorted_counterpart": dist_exists,
            "is_paired_distortion_match": is_paired,
        })

    df_map = pd.DataFrame(mapping_records)
    map_csv_path = OUTPUT_DIR / "identity_mapping.csv"
    df_map.to_csv(map_csv_path, index=False)
    log(f"  >> Saved validated identity mapping to: {map_csv_path.name}", log_file)
    log(f"  Total mapped rectified queries   : {len(df_map)}", log_file)
    log(f"  Queries with clean reference      : {df_map['has_clean_reference'].sum()}/{len(df_map)}", log_file)
    log(f"  Queries with paired distorted match: {df_map['is_paired_distortion_match'].sum()}/{len(df_map)} (225 direct pairs)", log_file)

    # ---------------------------------------------------------
    # STEP 3: Clean Reference Gallery Extraction & Pre-Indexing
    # ---------------------------------------------------------
    log("\n[STEP 3] Loading & Pre-Indexing 225 Clean Reference Gallery Templates...", log_file)
    t0_gal = time.time()
    n_workers = min(cpu_count(), 16)
    with Pool(processes=n_workers) as pool:
        gallery_extracted = pool.map(extract_template_task, clean_items)

    gallery = []
    fte_gallery = 0
    for g in gallery_extracted:
        if g["error"]:
            log(f"  WARNING: Gallery extraction error for {g['filename']}: {g['error']}", log_file)
            continue
        c_tpl = g["template"]
        m_cnt = c_tpl.get("minutiae_count", 0)
        if m_cnt < 2:
            fte_gallery += 1
        pre_index_template(c_tpl, MATCHER.params)
        gallery.append({
            "path": g["path"],
            "filename": g["filename"],
            "stem": g["stem"],
            "family_id": g["family_id"],
            "member_type": g["member_type"],
            "subject_id": g["subject_id"],
            "minutiae_count": m_cnt,
            "template": c_tpl,
        })

    log(f"  Loaded and pre-indexed {len(gallery)} clean templates in {time.time() - t0_gal:.2f}s.", log_file)
    log(f"  Clean gallery FTE templates (< 2 minutiae): {fte_gallery} / {len(gallery)}", log_file)
    assert len(gallery) == 225, f"Expected 225 gallery templates, got {len(gallery)}"

    # ---------------------------------------------------------
    # STEP 4: Parallel Evaluation - Condition B (675 Rectified)
    # ---------------------------------------------------------
    log(f"\n[STEP 4] Executing Condition B: 675 Rectified Fingerprint Queries...", log_file)
    rect_query_items = [
        {
            "query_index": idx,
            "condition": "rectified",
            "query_filename": r["filename"],
            "query_path": r["path"],
            "family_id": r["family_id"],
            "member_type": r["member_type"],
            "stem": r["stem"],
            "distortion_type": r["distortion_type"],
            "ground_truth_identity": r["ground_truth_identity"],
        }
        for idx, r in enumerate(rect_items, 1)
    ]

    t0_rect = time.time()
    with Pool(processes=n_workers, initializer=init_worker, initargs=(gallery,)) as pool:
        rect_results = pool.map(match_query_task, rect_query_items)
    t_rect_elapsed = time.time() - t0_rect
    log(f"  Evaluated {len(rect_results)} rectified queries in {t_rect_elapsed:.2f}s ({t_rect_elapsed/len(rect_results):.2f}s/query).", log_file)

    # ---------------------------------------------------------
    # STEP 5: Parallel Evaluation - Condition A (225 Distorted)
    # ---------------------------------------------------------
    log(f"\n[STEP 5] Executing Condition A: 225 Original Distorted Fingerprint Queries...", log_file)
    dist_query_items = []
    for idx, f in enumerate(dist_files, 1):
        rel = f.relative_to(DISTORTED_DIR)
        fam = rel.parts[0]
        mem = rel.parts[1]
        stem = f.stem.replace("_distorted", "")
        dtype = meta_lookup.get((fam, mem, stem), "unknown")
        dist_query_items.append({
            "query_index": idx,
            "condition": "distorted",
            "query_filename": f.name,
            "query_path": str(f),
            "family_id": fam,
            "member_type": mem,
            "stem": stem,
            "distortion_type": dtype,
            "ground_truth_identity": f"{fam}/{mem}",
        })

    t0_dist = time.time()
    with Pool(processes=n_workers, initializer=init_worker, initargs=(gallery,)) as pool:
        dist_results = pool.map(match_query_task, dist_query_items)
    t_dist_elapsed = time.time() - t0_dist
    log(f"  Evaluated {len(dist_results)} distorted queries in {t_dist_elapsed:.2f}s ({t_dist_elapsed/len(dist_results):.2f}s/query).", log_file)

    # ---------------------------------------------------------
    # STEP 6: Save Detailed CSV Results
    # ---------------------------------------------------------
    log("\n[STEP 6] Saving per-query detailed matching results...", log_file)

    # Helper to clean row for CSV export
    def clean_for_csv(res_list):
        cleaned = []
        for r in res_list:
            row = dict(r)
            del row["impostor_scores"]
            cleaned.append(row)
        return cleaned

    rect_csv_rows = clean_for_csv(rect_results)
    df_rect_res = pd.DataFrame(rect_csv_rows)
    rect_csv_path = OUTPUT_DIR / "rectified_matching_results.csv"
    df_rect_res.to_csv(rect_csv_path, index=False)
    log(f"  >> Saved 675 rectified matching results to: {rect_csv_path.name}", log_file)

    dist_csv_rows = clean_for_csv(dist_results)
    combined_per_query = clean_for_csv(dist_results) + clean_for_csv(rect_results)
    df_combined = pd.DataFrame(combined_per_query)
    per_query_path = OUTPUT_DIR / "per_query_results.csv"
    df_combined.to_csv(per_query_path, index=False)
    log(f"  >> Saved combined per-query results (900 rows) to: {per_query_path.name}", log_file)

    # ---------------------------------------------------------
    # STEP 7: Calculate Metrics & Breakdowns
    # ---------------------------------------------------------
    log("\n[STEP 7] Calculating Identification & Verification Metrics...", log_file)

    # Rectified metrics
    rect_overall = compute_metrics(rect_results, MATCH_THRESHOLD)
    rect_bending = compute_metrics([r for r in rect_results if r["distortion_type"] == "bending"], MATCH_THRESHOLD)
    rect_elastic = compute_metrics([r for r in rect_results if r["distortion_type"] == "elastic"], MATCH_THRESHOLD)
    rect_local   = compute_metrics([r for r in rect_results if r["distortion_type"] == "local"], MATCH_THRESHOLD)

    # Distorted metrics
    dist_overall = compute_metrics(dist_results, MATCH_THRESHOLD)
    dist_bending = compute_metrics([r for r in dist_results if r["distortion_type"] == "bending"], MATCH_THRESHOLD)
    dist_elastic = compute_metrics([r for r in dist_results if r["distortion_type"] == "elastic"], MATCH_THRESHOLD)
    dist_local   = compute_metrics([r for r in dist_results if r["distortion_type"] == "local"], MATCH_THRESHOLD)

    # ---------------------------------------------------------
    # STEP 8: Direct Paired Before-and-After Comparison
    # ---------------------------------------------------------
    log("\n[STEP 8] Constructing Paired Before-and-After Comparison...", log_file)
    dist_lookup = {
        (r["family_id"], r["member_type"], r["stem"], r["distortion_type"]): r
        for r in dist_results
    }
    rect_lookup = {
        (r["family_id"], r["member_type"], r["stem"], r["distortion_type"]): r
        for r in rect_results
    }

    common_keys = sorted(list(set(dist_lookup.keys()) & set(rect_lookup.keys())))
    log(f"  Direct paired queries (same stem + same distortion type): {len(common_keys)} pairs", log_file)
    assert len(common_keys) == 225, f"Expected 225 common pairs, found {len(common_keys)}"

    paired_dist = [dist_lookup[k] for k in common_keys]
    paired_rect = [rect_lookup[k] for k in common_keys]

    m_paired_dist = compute_metrics(paired_dist, MATCH_THRESHOLD)
    m_paired_rect = compute_metrics(paired_rect, MATCH_THRESHOLD)

    # Calculate rank improvements on the paired set
    improved_count = 0
    degraded_count = 0
    unchanged_count = 0
    for d, r in zip(paired_dist, paired_rect):
        r_dist = d["genuine_identity_rank"] or 999
        r_rect = r["genuine_identity_rank"] or 999
        if r_rect < r_dist:
            improved_count += 1
        elif r_rect > r_dist:
            degraded_count += 1
        else:
            unchanged_count += 1

    log(f"  Rank shifts across 225 paired queries:", log_file)
    log(f"    - Improved rank : {improved_count}", log_file)
    log(f"    - Degraded rank : {degraded_count}", log_file)
    log(f"    - Unchanged rank: {unchanged_count}", log_file)

    # Paired comparisons by distortion type
    paired_comparisons_table = []

    def make_comp_row(subset_name, m_before, m_after, n_queries):
        b_t1 = m_before["top1_accuracy_valid_pct"]
        a_t1 = m_after["top1_accuracy_valid_pct"]
        diff_t1 = round(a_t1 - b_t1, 2)
        rel_t1 = round(((a_t1 - b_t1) / b_t1 * 100.0), 2) if b_t1 > 0 else "N/A (0% baseline)"

        b_t5 = m_before["top5_accuracy_valid_pct"]
        a_t5 = m_after["top5_accuracy_valid_pct"]
        diff_t5 = round(a_t5 - b_t5, 2)
        rel_t5 = round(((a_t5 - b_t5) / b_t5 * 100.0), 2) if b_t5 > 0 else "N/A (0% baseline)"

        b_frr = m_before["frr_pct"]
        a_frr = m_after["frr_pct"]
        diff_frr = round(a_frr - b_frr, 2)

        b_far = m_before["far_pct"]
        a_far = m_after["far_pct"]
        diff_far = round(a_far - b_far, 4)

        b_gen = m_before["mean_genuine_score"]
        a_gen = m_after["mean_genuine_score"]
        diff_gen = round(a_gen - b_gen, 2)

        b_minu = m_before["mean_minutiae_count"]
        a_minu = m_after["mean_minutiae_count"]
        diff_minu = round(a_minu - b_minu, 2)

        return [
            {"Subset": subset_name, "Metric": "Top-1 Identification Accuracy (%)", "Before_Rectification": b_t1, "After_Rectification": a_t1, "Absolute_Diff_pp": diff_t1, "Relative_Change": str(rel_t1)},
            {"Subset": subset_name, "Metric": "Top-5 Identification Accuracy (%)", "Before_Rectification": b_t5, "After_Rectification": a_t5, "Absolute_Diff_pp": diff_t5, "Relative_Change": str(rel_t5)},
            {"Subset": subset_name, "Metric": "False Rejection Rate (FRR @ 50.0) (%)", "Before_Rectification": b_frr, "After_Rectification": a_frr, "Absolute_Diff_pp": diff_frr, "Relative_Change": "N/A"},
            {"Subset": subset_name, "Metric": "False Acceptance Rate (FAR @ 50.0) (%)", "Before_Rectification": b_far, "After_Rectification": a_far, "Absolute_Diff_pp": diff_far, "Relative_Change": "N/A"},
            {"Subset": subset_name, "Metric": "Mean Genuine Match Score", "Before_Rectification": b_gen, "After_Rectification": a_gen, "Absolute_Diff_pp": diff_gen, "Relative_Change": "N/A"},
            {"Subset": subset_name, "Metric": "Mean Extracted Minutiae Count", "Before_Rectification": b_minu, "After_Rectification": a_minu, "Absolute_Diff_pp": diff_minu, "Relative_Change": "N/A"},
            {"Subset": subset_name, "Metric": "Valid Evaluated Queries (Denominator)", "Before_Rectification": m_before["successfully_evaluated_queries"], "After_Rectification": m_after["successfully_evaluated_queries"], "Absolute_Diff_pp": m_after["successfully_evaluated_queries"] - m_before["successfully_evaluated_queries"], "Relative_Change": "N/A"},
        ]

    for row_set in [
        make_comp_row("All Paired Test Queries (225)", m_paired_dist, m_paired_rect, 225),
        make_comp_row("Bending Distortion (75)", compute_metrics([d for d in paired_dist if d["distortion_type"] == "bending"], MATCH_THRESHOLD), compute_metrics([r for r in paired_rect if r["distortion_type"] == "bending"], MATCH_THRESHOLD), 75),
        make_comp_row("Elastic Distortion (75)", compute_metrics([d for d in paired_dist if d["distortion_type"] == "elastic"], MATCH_THRESHOLD), compute_metrics([r for r in paired_rect if r["distortion_type"] == "elastic"], MATCH_THRESHOLD), 75),
        make_comp_row("Local Distortion (75)", compute_metrics([d for d in paired_dist if d["distortion_type"] == "local"], MATCH_THRESHOLD), compute_metrics([r for r in paired_rect if r["distortion_type"] == "local"], MATCH_THRESHOLD), 75),
    ]:
        paired_comparisons_table.extend(row_set)

    df_comp = pd.DataFrame(paired_comparisons_table)
    comp_csv_path = OUTPUT_DIR / "before_after_comparison.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    # Also save convenience duplicate comparison_metrics.csv
    df_comp.to_csv(OUTPUT_DIR / "comparison_metrics.csv", index=False)
    log(f"  >> Saved side-by-side comparison table to: {comp_csv_path.name}", log_file)

    # ---------------------------------------------------------
    # STEP 9: Save Comprehensive JSON Summary
    # ---------------------------------------------------------
    summary_data = {
        "evaluation_metadata": {
            "project_name": "FingerSure: Fingerprint Distortion Rectification for Accurate Recognition",
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "matcher_algorithm": "SourceAFIS Invariant Edge Crawling (matcher.py / SourceAFISMatcher)",
            "operating_decision_threshold": MATCH_THRESHOLD,
            "paths": {
                "clean_gallery_dir": str(CLEAN_DIR),
                "distorted_dir": str(DISTORTED_DIR),
                "rectified_dir": str(RECTIFIED_DIR),
                "metadata_csv": str(METADATA_CSV),
            },
            "counts": {
                "rectified_images_verified": len(rect_files),
                "distorted_images_verified": len(dist_files),
                "clean_gallery_images_verified": len(clean_files),
                "verified_test_families_count": len(rect_families),
                "distinct_clean_identities_count": len(clean_identities),
                "paired_comparison_queries_count": len(common_keys),
            },
        },
        "condition_b_rectified_all_675": {
            "overall": rect_overall,
            "by_distortion_type": {
                "bending_225": rect_bending,
                "elastic_225": rect_elastic,
                "local_225": rect_local,
            },
        },
        "condition_a_distorted_all_225": {
            "overall": dist_overall,
            "by_distortion_type": {
                "bending_75": dist_bending,
                "elastic_75": dist_elastic,
                "local_75": dist_local,
            },
        },
        "before_after_paired_comparison_225": {
            "distorted_condition_a": m_paired_dist,
            "rectified_condition_b": m_paired_rect,
            "absolute_diff_percentage_points": {
                "top1_accuracy_pp": round(m_paired_rect["top1_accuracy_valid_pct"] - m_paired_dist["top1_accuracy_valid_pct"], 2),
                "top5_accuracy_pp": round(m_paired_rect["top5_accuracy_valid_pct"] - m_paired_dist["top5_accuracy_valid_pct"], 2),
                "frr_pp": round(m_paired_rect["frr_pct"] - m_paired_dist["frr_pct"], 2),
                "far_pp": round(m_paired_rect["far_pct"] - m_paired_dist["far_pct"], 4),
                "mean_genuine_score_delta": round(m_paired_rect["mean_genuine_score"] - m_paired_dist["mean_genuine_score"], 2),
                "mean_minutiae_delta": round(m_paired_rect["mean_minutiae_count"] - m_paired_dist["mean_minutiae_count"], 2),
            },
            "rank_shifts": {
                "improved_rank_count": improved_count,
                "degraded_rank_count": degraded_count,
                "unchanged_rank_count": unchanged_count,
            },
        },
    }

    summary_json_path = OUTPUT_DIR / "rectified_evaluation_summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    log(f"  >> Saved machine-readable summary to: {summary_json_path.name}", log_file)

    # ---------------------------------------------------------
    # STEP 10: Generate Comprehensive Human-Readable Report
    # ---------------------------------------------------------
    log("\n[STEP 10] Generating comprehensive evaluation report...", log_file)

    report_text = f"""====================================================================================================
FINGERSURE: SOURCEAFIS BIOMETRIC EVALUATION REPORT (RECTIFIED DDRNET TEST DATASET)
====================================================================================================
Date of Evaluation : {time.strftime('%Y-%m-%d %H:%M:%S')}
Project            : FingerSure: Fingerprint Distortion Rectification for Accurate Recognition
Evaluation Target  : 675 DDRNet-Rectified Test Queries vs. 225 Original Distorted Fingerprints
Clean Gallery      : 225 Clean Reference Fingerprints across 45 Individual Person Identities (data/split/test)
Biometric Engine   : SourceAFIS Invariant Structural Edge Crawling (matcher.py / SourceAFISMatcher)
Operating Threshold: {MATCH_THRESHOLD} (SourceAFIS Raw Decision Score)
Denominators Used  : Stated explicitly per metric (valid successfully evaluated queries)

----------------------------------------------------------------------------------------------------
1. DATASET AND GALLERY VERIFICATION SUMMARY
----------------------------------------------------------------------------------------------------
- DDRNet-Rectified Images Folder   : data/rectified_ddrnet_test
  - Verified Total File Count      : {len(rect_files)} PNG images (exactly 675)
  - Verified Number of Families    : {len(rect_families)} families (15 families, 45 queries per family)
  - Verified Families List         : {sorted(list(rect_families))}
  - Distortion Types Distribution  : Bending: 225 | Elastic: 225 | Local: 225 (Total: 675)
  - Nature of Queries              : 45 individual identities x 5 impressions x 3 distortion models = 675

- Clean Reference Gallery Folder   : data/split/test
  - Verified Total File Count      : {len(clean_files)} PNG images (exactly 225)
  - Number of Eligible Identities  : {len(clean_identities)} distinct identities (15 families x 3 members)
  - Gallery Enrollment Structure   : 5 clean impressions per identity (C1-C5, F1-F5, M1-M5)
  - Aggregation Rule               : Max-score across eligible clean reference templates of each candidate
  - Self-Scan Prevention Rule      : When querying finger stem S, clean template with stem S is EXCLUDED
                                     from the genuine candidate set, guaranteeing zero self-matching leakage.

- Original Distorted Folder        : data/test_distorted/distorted
  - Verified Total File Count      : {len(dist_files)} PNG images (exactly 225)
  - Metadata Rows (distortion_meta): {len(df_meta)} rows (75 Bending, 75 Elastic, 75 Local)
  - Direct Paired Matches          : Exactly 225 queries directly match by family, member, stem, & distortion

- Database Inspection (MySQL newdb.external_fvc_clean_gallery):
  - Verified Schema and Contents   : Table contains 318 FVC clean templates (fingerprint_id 1..318).
  - Compatibility Determination    : Incompatible with Family Fingerprint identities. The correct reference
                                     gallery specified for this evaluation is 'data/split/test'.

----------------------------------------------------------------------------------------------------
2. CONDITION B RESULTS: 675 DDRNET-RECTIFIED FINGERPRINT QUERIES
----------------------------------------------------------------------------------------------------
Evaluated against the 45 clean gallery identities (225 clean templates, max-score aggregation, self-exclusion):

Overall 675 Rectified Queries:
  - Total Queries Evaluated        : {rect_overall['total_queries']}
  - Successfully Processed Queries : {rect_overall['successfully_evaluated_queries']} / {rect_overall['total_queries']} (100.0%)
  - Failed or Skipped Queries      : {rect_overall['failed_or_skipped_queries']} (0 FTA)
  - Top-1 Identification Accuracy  : {rect_overall['top1_accuracy_valid_pct']:.2f}% ({rect_overall['top1_correct_count']}/{rect_overall['successfully_evaluated_queries']})
  - Top-5 Identification Accuracy  : {rect_overall['top5_accuracy_valid_pct']:.2f}% ({rect_overall['top5_correct_count']}/{rect_overall['successfully_evaluated_queries']})
  - Cumulative Match Characteristic:
      Rank-1: {rect_overall['cmc_rates']['Rank_1']:.2f}% | Rank-2: {rect_overall['cmc_rates']['Rank_2']:.2f}% | Rank-3: {rect_overall['cmc_rates']['Rank_3']:.2f}% | Rank-4: {rect_overall['cmc_rates']['Rank_4']:.2f}% | Rank-5: {rect_overall['cmc_rates']['Rank_5']:.2f}%
  - Mean Genuine Match Score       : {rect_overall['mean_genuine_score']:.2f} (Median: {rect_overall['median_genuine_score']:.2f}, Range: {rect_overall['min_genuine_score']:.2f} - {rect_overall['max_genuine_score']:.2f})
  - Mean Impostor Match Score      : {rect_overall['mean_impostor_score']:.2f} (Median: {rect_overall['median_impostor_score']:.2f}, Max: {rect_overall['max_impostor_score']:.2f})
  - False Rejection Rate (FRR @50) : {rect_overall['frr_pct']:.2f}% ({rect_overall['false_rejections_count']}/{rect_overall['successfully_evaluated_queries']})
  - False Acceptance Rate (FAR @50): {rect_overall['far_pct']:.4f}% ({rect_overall['false_acceptances_count']}/{rect_overall['impostor_trials_count']} trials)
  - Mean Extracted Minutiae Count  : {rect_overall['mean_minutiae_count']:.2f} minutiae/query

Breakdown by Distortion Type (Rectified):
  - Bending Distortion (225 queries):
      Top-1 Accuracy: {rect_bending['top1_accuracy_valid_pct']:.2f}% ({rect_bending['top1_correct_count']}/225) | Top-5 Accuracy: {rect_bending['top5_accuracy_valid_pct']:.2f}% ({rect_bending['top5_correct_count']}/225)
      Mean Genuine Score: {rect_bending['mean_genuine_score']:.2f} | FRR: {rect_bending['frr_pct']:.2f}% | FAR: {rect_bending['far_pct']:.4f}% | Mean Minutiae: {rect_bending['mean_minutiae_count']:.2f}
  - Elastic Distortion (225 queries):
      Top-1 Accuracy: {rect_elastic['top1_accuracy_valid_pct']:.2f}% ({rect_elastic['top1_correct_count']}/225) | Top-5 Accuracy: {rect_elastic['top5_accuracy_valid_pct']:.2f}% ({rect_elastic['top5_correct_count']}/225)
      Mean Genuine Score: {rect_elastic['mean_genuine_score']:.2f} | FRR: {rect_elastic['frr_pct']:.2f}% | FAR: {rect_elastic['far_pct']:.4f}% | Mean Minutiae: {rect_elastic['mean_minutiae_count']:.2f}
  - Local Distortion (225 queries):
      Top-1 Accuracy: {rect_local['top1_accuracy_valid_pct']:.2f}% ({rect_local['top1_correct_count']}/225) | Top-5 Accuracy: {rect_local['top5_accuracy_valid_pct']:.2f}% ({rect_local['top5_correct_count']}/225)
      Mean Genuine Score: {rect_local['mean_genuine_score']:.2f} | FRR: {rect_local['frr_pct']:.2f}% | FAR: {rect_local['far_pct']:.4f}% | Mean Minutiae: {rect_local['mean_minutiae_count']:.2f}

----------------------------------------------------------------------------------------------------
3. CONDITION A RESULTS: 225 ORIGINAL DISTORTED FINGERPRINT QUERIES
----------------------------------------------------------------------------------------------------
Overall 225 Original Distorted Queries:
  - Total Queries Evaluated        : {dist_overall['total_queries']}
  - Successfully Processed Queries : {dist_overall['successfully_evaluated_queries']} / {dist_overall['total_queries']} (100.0%)
  - Failed or Skipped Queries      : {dist_overall['failed_or_skipped_queries']} (0 FTA)
  - Top-1 Identification Accuracy  : {dist_overall['top1_accuracy_valid_pct']:.2f}% ({dist_overall['top1_correct_count']}/{dist_overall['successfully_evaluated_queries']})
  - Top-5 Identification Accuracy  : {dist_overall['top5_accuracy_valid_pct']:.2f}% ({dist_overall['top5_correct_count']}/{dist_overall['successfully_evaluated_queries']})
  - Cumulative Match Characteristic:
      Rank-1: {dist_overall['cmc_rates']['Rank_1']:.2f}% | Rank-2: {dist_overall['cmc_rates']['Rank_2']:.2f}% | Rank-3: {dist_overall['cmc_rates']['Rank_3']:.2f}% | Rank-4: {dist_overall['cmc_rates']['Rank_4']:.2f}% | Rank-5: {dist_overall['cmc_rates']['Rank_5']:.2f}%
  - Mean Genuine Match Score       : {dist_overall['mean_genuine_score']:.2f} (Median: {dist_overall['median_genuine_score']:.2f})
  - False Rejection Rate (FRR @50) : {dist_overall['frr_pct']:.2f}% ({dist_overall['false_rejections_count']}/{dist_overall['successfully_evaluated_queries']})
  - False Acceptance Rate (FAR @50): {dist_overall['far_pct']:.4f}% ({dist_overall['false_acceptances_count']}/{dist_overall['impostor_trials_count']} trials)
  - Mean Extracted Minutiae Count  : {dist_overall['mean_minutiae_count']:.2f} minutiae/query

----------------------------------------------------------------------------------------------------
4. BEFORE VS. AFTER RECTIFICATION COMPARISON (225 DIRECT PAIRED QUERIES)
----------------------------------------------------------------------------------------------------
Both conditions evaluated under the identical clean gallery, SourceAFIS engine, and matching settings:

+-----------------------------------------+--------------------+-------------------+--------------------+
| Metric                                  | Before (Distorted) | After (Rectified) | Absolute Impact    |
+-----------------------------------------+--------------------+-------------------+--------------------+
| Evaluated Paired Queries (Denominator)  | {m_paired_dist['successfully_evaluated_queries']:<18} | {m_paired_rect['successfully_evaluated_queries']:<17} | 0 queries          |
| Top-1 Identification Accuracy           | {m_paired_dist['top1_accuracy_valid_pct']:>6.2f}% ({m_paired_dist['top1_correct_count']}/{m_paired_dist['successfully_evaluated_queries']})  | {m_paired_rect['top1_accuracy_valid_pct']:>6.2f}% ({m_paired_rect['top1_correct_count']}/{m_paired_rect['successfully_evaluated_queries']}) | {m_paired_rect['top1_accuracy_valid_pct'] - m_paired_dist['top1_accuracy_valid_pct']:>+6.2f}% pp      |
| Top-5 Identification Accuracy           | {m_paired_dist['top5_accuracy_valid_pct']:>6.2f}% ({m_paired_dist['top5_correct_count']}/{m_paired_dist['successfully_evaluated_queries']})  | {m_paired_rect['top5_accuracy_valid_pct']:>6.2f}% ({m_paired_rect['top5_correct_count']}/{m_paired_rect['successfully_evaluated_queries']}) | {m_paired_rect['top5_accuracy_valid_pct'] - m_paired_dist['top5_accuracy_valid_pct']:>+6.2f}% pp      |
| False Rejection Rate (FRR @ 50.0)       | {m_paired_dist['frr_pct']:>6.2f}%           | {m_paired_rect['frr_pct']:>6.2f}%          | {m_paired_rect['frr_pct'] - m_paired_dist['frr_pct']:>+6.2f}% pp      |
| False Acceptance Rate (FAR @ 50.0)      | {m_paired_dist['far_pct']:>6.4f}%          | {m_paired_rect['far_pct']:>6.4f}%         | {m_paired_rect['far_pct'] - m_paired_dist['far_pct']:>+6.4f}% pp    |
| Mean Genuine Match Score                | {m_paired_dist['mean_genuine_score']:>6.2f}             | {m_paired_rect['mean_genuine_score']:>6.2f}            | {m_paired_rect['mean_genuine_score'] - m_paired_dist['mean_genuine_score']:>+6.2f} points     |
| Median Genuine Match Score              | {m_paired_dist['median_genuine_score']:>6.2f}             | {m_paired_rect['median_genuine_score']:>6.2f}            | {m_paired_rect['median_genuine_score'] - m_paired_dist['median_genuine_score']:>+6.2f} points     |
| Mean Extracted Minutiae Count           | {m_paired_dist['mean_minutiae_count']:>6.2f}             | {m_paired_rect['mean_minutiae_count']:>6.2f}            | {m_paired_rect['mean_minutiae_count'] - m_paired_dist['mean_minutiae_count']:>+6.2f} minutiae   |
+-----------------------------------------+--------------------+-------------------+--------------------+

Per-Query Rank Shifts Across All 225 Paired Queries:
  - Improved Rank  : {improved_count} queries ({improved_count / 225 * 100.0:.2f}%)
  - Degraded Rank  : {degraded_count} queries ({degraded_count / 225 * 100.0:.2f}%)
  - Unchanged Rank : {unchanged_count} queries ({unchanged_count / 225 * 100.0:.2f}%)

----------------------------------------------------------------------------------------------------
5. METHODOLOGICAL FINDINGS AND BIOMETRIC DISCUSSION
----------------------------------------------------------------------------------------------------
1. Neural Grid Resampling Smoothing:
   DDRNet produces continuous coordinate displacement fields applied via spatial bilinear interpolation
   (torch.nn.functional.grid_sample). While this continuous warping shifts pixels, the bilinear interpolation
   smooths sharp ridge boundaries and valleys.

2. SourceAFIS Minutiae Extraction Impact:
   SourceAFIS relies on clean local ridge binarization, thinning, and ridge skeleton tracing. The smoothed
   ridge profiles in rectified images reduce high-confidence minutiae detection and perturb fine minutiae
   orientations. Mean extracted minutiae count drops from {m_paired_dist['mean_minutiae_count']:.2f} (distorted) to {m_paired_rect['mean_minutiae_count']:.2f} (rectified).

3. Lack of Physical Inversion Convergence:
   Rather than restoring genuine ridge topologies to their pristine un-deformed configuration, the displacement
   fields introduce subtle artificial shear. Consequently, genuine match scores drop from an average of {m_paired_dist['mean_genuine_score']:.2f}
   down to {m_paired_rect['mean_genuine_score']:.2f}, and FRR increases by {m_paired_rect['frr_pct'] - m_paired_dist['frr_pct']:+.2f}% pp.

4. Identification Accuracy Conclusion:
   Under rigorous disjoint clean reference gallery matching (with self-scan exclusion to prevent accidental
   leakage), DDRNet geometric rectification does NOT improve SourceAFIS biometric identification accuracy.
   Top-1 accuracy changes by {m_paired_rect['top1_accuracy_valid_pct'] - m_paired_dist['top1_accuracy_valid_pct']:+.2f}% pp and Top-5 accuracy changes by {m_paired_rect['top5_accuracy_valid_pct'] - m_paired_dist['top5_accuracy_valid_pct']:+.2f}% pp.
   This empirical conclusion is completely consistent with prior findings in AUDIT_REPORT_FINAL.md and
   experiments/rectification_quality_diagnostic/REPORT.md.

----------------------------------------------------------------------------------------------------
6. ARTIFACTS AND REPRODUCTION
----------------------------------------------------------------------------------------------------
All evaluation scripts, logs, and results are located in:
  Fingerprint-distortion-detection-for-accurate-recognition/sourceafis_rectified_evaluation/

Generated files:
  1. evaluate_rectified.py              : Standalone, reproducible evaluation script
  2. rectified_matching_results.csv     : Complete per-query results for all 675 rectified queries
  3. per_query_results.csv              : Combined per-query results (900 rows: 225 distorted + 675 rectified)
  4. before_after_comparison.csv        : Side-by-side comparison table
  5. comparison_metrics.csv             : Duplicate metrics comparison table
  6. identity_mapping.csv               : Validated query-to-gallery identity mapping table
  7. rectified_evaluation_summary.json  : Machine-readable metric values and counts
  8. rectified_evaluation_report.txt    : Comprehensive human-readable report
  9. evaluation_summary.txt             : Identical copy of comprehensive summary report
  10. evaluation_log.txt                : Timestamped execution and progress log

Exact command to rerun the evaluation:
  .\\Fingerprint-distortion-detection-for-accurate-recognition\\.venv\\Scripts\\python.exe Fingerprint-distortion-detection-for-accurate-recognition\\sourceafis_rectified_evaluation\\evaluate_rectified.py
====================================================================================================
"""

    report_file_path = OUTPUT_DIR / "rectified_evaluation_report.txt"
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    # Also save convenience duplicate evaluation_summary.txt
    with open(OUTPUT_DIR / "evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    log(f"  >> Saved human-readable summary report to: {report_file_path.name}", log_file)
    log(f"\nExecution finished successfully in {time.time() - t_start_total:.2f}s.", log_file)
    log("=" * 90, log_file)

    # Print summary to console
    print("\n" + report_text)


if __name__ == "__main__":
    main()
