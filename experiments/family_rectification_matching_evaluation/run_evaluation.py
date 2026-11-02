#!/usr/bin/env python3
"""
Evaluation of DDRNet Fingerprint Distortion Rectification on Identification Accuracy.
Dataset: Family Fingerprint Dataset (180 eligible test queries from Impressions 2-5).
Gallery: 300 enrolled clean individual identities from MySQL 'fingerprint_authentication' (Impression 1).
Matching Engine: Unmodified SourceAFIS (matcher.py / MATCHER.identify_1toN).
Threshold: database.MATCH_THRESHOLD (50.0).

Compares:
  Condition 1 (BEFORE): Original Distorted Fingerprint Queries
  Condition 2 (AFTER):  DDRNet-Rectified Fingerprint Queries
Across:
  - Combined (180 queries)
  - Bending (60 queries)
  - Elastic (59 queries)
  - Local (61 queries)
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
import pickle
import re
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

import mysql.connector
from matcher import MATCHER, SourceAFISMatcher
import afis.algorithms.safis as safis
from database import MATCH_THRESHOLD

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "family_rectification_matching_evaluation"

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
DB_NAME = "fingerprint_authentication"

# Global gallery reference for worker processes
GLOBAL_GALLERY: List[Dict[str, Any]] = []


def get_mysql_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=DB_NAME,
    )


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


def load_300_clean_gallery() -> List[Dict[str, Any]]:
    """Loads all 300 clean Impression 1 enrolled templates from fingerprint_authentication."""
    print("Loading 300 clean enrolled templates (Impression 1) from MySQL database...", flush=True)
    t0 = time.time()
    conn = get_mysql_conn()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT 
            f.fingerprint_id,
            f.person_id,
            p.identity_code,
            p.family_id,
            p.member_type,
            f.impression_id,
            f.image_path,
            f.minutiae_count,
            f.minutiae_template
        FROM fingerprints f
        JOIN persons p ON f.person_id = p.person_id
        WHERE f.impression_id = 1
        ORDER BY p.person_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    gallery = []
    fte_count = 0

    for r in rows:
        raw_blob = r["minutiae_template"]
        try:
            t_obj = pickle.loads(raw_blob)
            minu = getattr(t_obj, "minutiae", None)
            if minu is None and isinstance(t_obj, dict):
                minu = t_obj.get("minutiae")
        except Exception:
            minu = None

        m_len = len(minu) if minu is not None else 0
        if m_len < 2:
            fte_count += 1

        tpl = {
            "minutiae": minu,
            "minutiae_count": m_len,
        }
        pre_index_template(tpl, MATCHER.params)

        gallery.append({
            "fingerprint_id": int(r["fingerprint_id"]),
            "person_id": int(r["person_id"]),
            "subject_id": str(r["identity_code"]),
            "family_id": str(r["family_id"]),
            "member_type": str(r["member_type"]),
            "impression_id": int(r["impression_id"]),
            "image_path": str(r["image_path"]),
            "actual_minutiae_count": m_len,
            "template": tpl,
        })

    elapsed = time.time() - t0
    print(f"Loaded {len(gallery)} gallery templates in {elapsed:.2f}s.", flush=True)
    print(f"  - Clean Gallery FTE templates (< 2 minutiae): {fte_count} / {len(gallery)}", flush=True)
    return gallery


def load_180_eligible_queries() -> List[Dict[str, Any]]:
    """Loads and verifies the 180 eligible test queries (Impressions 2-5)."""
    csv_path = PROJECT_ROOT / "data" / "test_distorted" / "distortion_metadata.csv"
    df = pd.read_csv(csv_path)

    p_dist_base = PROJECT_ROOT / "data" / "test_distorted" / "distorted"
    p_rect_base = PROJECT_ROOT / "data" / "rectified_ddrnet_test"

    def get_imp(path_str):
        m = re.search(r'_([CFM])(\d)', str(path_str))
        return int(m.group(2)) if m else None

    df["impression"] = df["clean_output_path"].apply(get_imp)
    eligible = df[df["impression"] >= 2].copy()

    queries = []
    for idx, r in eligible.iterrows():
        fam = str(r["family_id"])
        mem = str(r["member"])
        dist_type = str(r["distortion_type"])
        stem = Path(r["clean_output_path"]).stem
        imp_id = int(r["impression"])
        identity_code = f"{fam}/{mem}"

        dist_path = p_dist_base / fam / mem / f"{stem}_distorted.png"
        rect_path = p_rect_base / f"{fam}__{mem}__{stem}__{dist_type}.png"

        if not dist_path.exists():
            raise FileNotFoundError(f"Distorted image missing: {dist_path}")
        if not rect_path.exists():
            raise FileNotFoundError(f"Rectified image missing: {rect_path}")

        queries.append({
            "query_index": len(queries) + 1,
            "family_id": fam,
            "member_type": mem,
            "identity_code": identity_code,
            "impression_id": imp_id,
            "stem": stem,
            "distortion_type": dist_type,
            "distorted_image_path": str(dist_path),
            "rectified_image_path": str(rect_path),
        })

    print(f"Loaded {len(queries)} eligible queries from metadata (Impressions 2-5):", flush=True)
    counts = pd.Series([q["distortion_type"] for q in queries]).value_counts()
    for d, c in counts.items():
        print(f"  - {d}: {c}", flush=True)

    return queries


def init_worker(gallery_data: List[Dict[str, Any]]) -> None:
    """Worker process initializer to set shared read-only gallery."""
    global GLOBAL_GALLERY
    GLOBAL_GALLERY = gallery_data


def process_query_trial(query_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker task: Evaluates 1 query under BOTH Before (Distorted) and After (Rectified) conditions.
    """
    global GLOBAL_GALLERY
    target_identity = query_info["identity_code"]

    # ----------------------------------------------------
    # CONDITION 1: BEFORE (Distorted Query)
    # ----------------------------------------------------
    img_before = cv2.imread(query_info["distorted_image_path"], cv2.IMREAD_GRAYSCALE)
    if img_before is None:
        before_tpl = {"minutiae": None, "minutiae_count": 0}
        before_fta = True
    else:
        before_tpl = MATCHER.extract_template(img_before)
        b_minu = before_tpl.get("minutiae")
        before_fta = (b_minu is None or len(b_minu) < 2)

    t0_b = time.time()
    res_before = MATCHER.identify_1toN(before_tpl, GLOBAL_GALLERY, top_n=len(GLOBAL_GALLERY), threshold=MATCH_THRESHOLD)
    time_b_ms = (time.time() - t0_b) * 1000.0

    cands_before = res_before.get("candidates", [])
    ranks_before = {c["subject_id"]: i + 1 for i, c in enumerate(cands_before)}
    before_genuine_rank = ranks_before.get(target_identity, None)

    before_top1_cand = cands_before[0] if cands_before else None
    before_top1_id = before_top1_cand["subject_id"] if before_top1_cand else None
    before_top1_score = float(before_top1_cand["raw_score"]) if before_top1_cand else 0.0

    before_target_cand = next((c for c in cands_before if c["subject_id"] == target_identity), None)
    before_genuine_score = float(before_target_cand["raw_score"]) if before_target_cand else 0.0
    before_genuine_matched = int(before_target_cand.get("matched_minutiae", 0)) if before_target_cand else 0

    before_is_top1 = (before_genuine_rank == 1)
    before_is_top5 = (before_genuine_rank is not None and before_genuine_rank <= 5)
    before_genuine_accepted = (before_genuine_score >= MATCH_THRESHOLD)

    # Collect all 299 impostor scores for FAR
    before_impostor_scores = [
        float(c["raw_score"]) for c in cands_before if c["subject_id"] != target_identity
    ]

    # ----------------------------------------------------
    # CONDITION 2: AFTER (Rectified Query)
    # ----------------------------------------------------
    img_after = cv2.imread(query_info["rectified_image_path"], cv2.IMREAD_GRAYSCALE)
    if img_after is None:
        after_tpl = {"minutiae": None, "minutiae_count": 0}
        after_fta = True
    else:
        after_tpl = MATCHER.extract_template(img_after)
        a_minu = after_tpl.get("minutiae")
        after_fta = (a_minu is None or len(a_minu) < 2)

    t0_a = time.time()
    res_after = MATCHER.identify_1toN(after_tpl, GLOBAL_GALLERY, top_n=len(GLOBAL_GALLERY), threshold=MATCH_THRESHOLD)
    time_a_ms = (time.time() - t0_a) * 1000.0

    cands_after = res_after.get("candidates", [])
    ranks_after = {c["subject_id"]: i + 1 for i, c in enumerate(cands_after)}
    after_genuine_rank = ranks_after.get(target_identity, None)

    after_top1_cand = cands_after[0] if cands_after else None
    after_top1_id = after_top1_cand["subject_id"] if after_top1_cand else None
    after_top1_score = float(after_top1_cand["raw_score"]) if after_top1_cand else 0.0

    after_target_cand = next((c for c in cands_after if c["subject_id"] == target_identity), None)
    after_genuine_score = float(after_target_cand["raw_score"]) if after_target_cand else 0.0
    after_genuine_matched = int(after_target_cand.get("matched_minutiae", 0)) if after_target_cand else 0

    after_is_top1 = (after_genuine_rank == 1)
    after_is_top5 = (after_genuine_rank is not None and after_genuine_rank <= 5)
    after_genuine_accepted = (after_genuine_score >= MATCH_THRESHOLD)

    after_impostor_scores = [
        float(c["raw_score"]) for c in cands_after if c["subject_id"] != target_identity
    ]

    # Rank comparison
    if before_genuine_rank is None and after_genuine_rank is not None:
        rank_change = "IMPROVED"
    elif before_genuine_rank is not None and after_genuine_rank is None:
        rank_change = "DEGRADED"
    elif before_genuine_rank is None and after_genuine_rank is None:
        rank_change = "UNCHANGED_UNMATCHED"
    elif after_genuine_rank < before_genuine_rank:
        rank_change = "IMPROVED"
    elif after_genuine_rank > before_genuine_rank:
        rank_change = "DEGRADED"
    else:
        rank_change = "UNCHANGED"

    score_diff = round(after_genuine_score - before_genuine_score, 2)

    return {
        "query_index": query_info["query_index"],
        "family_id": query_info["family_id"],
        "member_type": query_info["member_type"],
        "identity_code": target_identity,
        "impression_id": query_info["impression_id"],
        "stem": query_info["stem"],
        "distortion_type": query_info["distortion_type"],
        "distorted_image_path": query_info["distorted_image_path"],
        "rectified_image_path": query_info["rectified_image_path"],
        # Before
        "before_minutiae_count": before_tpl.get("minutiae_count", 0),
        "before_is_fta": before_fta,
        "before_predicted_top1": before_top1_id,
        "before_predicted_top1_score": before_top1_score,
        "before_genuine_rank": before_genuine_rank,
        "before_genuine_score": before_genuine_score,
        "before_genuine_matched_minutiae": before_genuine_matched,
        "before_top1_correct": before_is_top1,
        "before_top5_correct": before_is_top5,
        "before_genuine_accepted": before_genuine_accepted,
        "before_time_ms": round(time_b_ms, 2),
        "before_impostor_scores": before_impostor_scores,
        # After
        "after_minutiae_count": after_tpl.get("minutiae_count", 0),
        "after_is_fta": after_fta,
        "after_predicted_top1": after_top1_id,
        "after_predicted_top1_score": after_top1_score,
        "after_genuine_rank": after_genuine_rank,
        "after_genuine_score": after_genuine_score,
        "after_genuine_matched_minutiae": after_genuine_matched,
        "after_top1_correct": after_is_top1,
        "after_top5_correct": after_is_top5,
        "after_genuine_accepted": after_genuine_accepted,
        "after_time_ms": round(time_a_ms, 2),
        "after_impostor_scores": after_impostor_scores,
        # Comparison
        "rank_change": rank_change,
        "score_diff": score_diff,
    }


def compute_metrics_for_subset(
    subset_results: List[Dict[str, Any]],
    subset_name: str,
    gallery_fte_identities: set,
) -> Dict[str, Any]:
    """
    Computes rigorous Top-1, Top-5, FAR, and FRR metrics for a query subset.
    """
    n_queries = len(subset_results)
    if n_queries == 0:
        return {}

    # Verification trial counts
    # Each query represents 1 genuine trial against the clean enrolled template of the same identity
    n_genuine_trials = n_queries
    # Each query tests against 299 impostor gallery templates
    n_impostors_per_query = len(subset_results[0]["before_impostor_scores"])
    total_impostor_trials = n_queries * n_impostors_per_query

    # Valid query criteria:
    # Probe is valid if minutiae >= 2 (not FTA) AND its corresponding gallery enrolled template has minutiae >= 2 (not FTE)
    # BEFORE
    before_valid_results = [
        r for r in subset_results 
        if (not r["before_is_fta"]) and (r["identity_code"] not in gallery_fte_identities)
    ]
    before_n_valid = len(before_valid_results)
    before_n_fta = sum(1 for r in subset_results if r["before_is_fta"])

    before_top1_all = sum(1 for r in subset_results if r["before_top1_correct"])
    before_top5_all = sum(1 for r in subset_results if r["before_top5_correct"])
    before_top1_val = sum(1 for r in before_valid_results if r["before_top1_correct"])
    before_top5_val = sum(1 for r in before_valid_results if r["before_top5_correct"])

    # FRR: Genuine trials where score < MATCH_THRESHOLD
    before_fr_count = sum(1 for r in subset_results if not r["before_genuine_accepted"])
    before_frr_pct = (before_fr_count / n_genuine_trials) * 100.0

    # FAR: Impostor comparisons where impostor_score >= MATCH_THRESHOLD
    before_fa_count = sum(
        sum(1 for s in r["before_impostor_scores"] if s >= MATCH_THRESHOLD)
        for r in subset_results
    )
    before_far_pct = (before_fa_count / total_impostor_trials) * 100.0

    before_gen_scores = [r["before_genuine_score"] for r in subset_results]
    before_mean_gen = float(np.mean(before_gen_scores))
    before_median_gen = float(np.median(before_gen_scores))

    # AFTER
    after_valid_results = [
        r for r in subset_results 
        if (not r["after_is_fta"]) and (r["identity_code"] not in gallery_fte_identities)
    ]
    after_n_valid = len(after_valid_results)
    after_n_fta = sum(1 for r in subset_results if r["after_is_fta"])

    after_top1_all = sum(1 for r in subset_results if r["after_top1_correct"])
    after_top5_all = sum(1 for r in subset_results if r["after_top5_correct"])
    after_top1_val = sum(1 for r in after_valid_results if r["after_top1_correct"])
    after_top5_val = sum(1 for r in after_valid_results if r["after_top5_correct"])

    after_fr_count = sum(1 for r in subset_results if not r["after_genuine_accepted"])
    after_frr_pct = (after_fr_count / n_genuine_trials) * 100.0

    after_fa_count = sum(
        sum(1 for s in r["after_impostor_scores"] if s >= MATCH_THRESHOLD)
        for r in subset_results
    )
    after_far_pct = (after_fa_count / total_impostor_trials) * 100.0

    after_gen_scores = [r["after_genuine_score"] for r in subset_results]
    after_mean_gen = float(np.mean(after_gen_scores))
    after_median_gen = float(np.median(after_gen_scores))

    # Outcomes
    improved = sum(1 for r in subset_results if r["rank_change"] == "IMPROVED")
    degraded = sum(1 for r in subset_results if r["rank_change"] == "DEGRADED")
    unchanged = sum(1 for r in subset_results if r["rank_change"] in ("UNCHANGED", "UNCHANGED_UNMATCHED"))

    return {
        "subset_name": subset_name,
        "query_count": n_queries,
        "gallery_size": len(GLOBAL_GALLERY),
        "threshold": MATCH_THRESHOLD,
        "genuine_trials_count": n_genuine_trials,
        "impostor_trials_count": total_impostor_trials,
        # Before
        "before_valid_queries": before_n_valid,
        "before_failed_queries_fta": before_n_fta,
        "before_top1_correct_count": before_top1_all,
        "before_top1_accuracy_all_pct": round((before_top1_all / n_queries) * 100.0, 2),
        "before_top1_accuracy_valid_pct": round((before_top1_val / before_n_valid * 100.0), 2) if before_n_valid > 0 else 0.0,
        "before_top5_correct_count": before_top5_all,
        "before_top5_accuracy_all_pct": round((before_top5_all / n_queries) * 100.0, 2),
        "before_top5_accuracy_valid_pct": round((before_top5_val / before_n_valid * 100.0), 2) if before_n_valid > 0 else 0.0,
        "before_false_rejections_count": before_fr_count,
        "before_frr_pct": round(before_frr_pct, 2),
        "before_false_acceptances_count": before_fa_count,
        "before_far_pct": round(before_far_pct, 4),
        "before_mean_genuine_score": round(before_mean_gen, 2),
        "before_median_genuine_score": round(before_median_gen, 2),
        # After
        "after_valid_queries": after_n_valid,
        "after_failed_queries_fta": after_n_fta,
        "after_top1_correct_count": after_top1_all,
        "after_top1_accuracy_all_pct": round((after_top1_all / n_queries) * 100.0, 2),
        "after_top1_accuracy_valid_pct": round((after_top1_val / after_n_valid * 100.0), 2) if after_n_valid > 0 else 0.0,
        "after_top5_correct_count": after_top5_all,
        "after_top5_accuracy_all_pct": round((after_top5_all / n_queries) * 100.0, 2),
        "after_top5_accuracy_valid_pct": round((after_top5_val / after_n_valid * 100.0), 2) if after_n_valid > 0 else 0.0,
        "after_false_rejections_count": after_fr_count,
        "after_frr_pct": round(after_frr_pct, 2),
        "after_false_acceptances_count": after_fa_count,
        "after_far_pct": round(after_far_pct, 4),
        "after_mean_genuine_score": round(after_mean_gen, 2),
        "after_median_genuine_score": round(after_median_gen, 2),
        # Comparison
        "improved_rank_count": improved,
        "degraded_rank_count": degraded,
        "unchanged_rank_count": unchanged,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    n_workers = min(cpu_count(), 14)

    print("=" * 90, flush=True)
    print("   DDRNET RECTIFICATION BIOMETRIC IDENTIFICATION EVALUATION (FAMILY DATASET)", flush=True)
    print("   Clean Gallery: 300 Enrolled Identities (MySQL fingerprint_authentication)", flush=True)
    print(f"   Workers: {n_workers} CPU cores | Decision Threshold: {MATCH_THRESHOLD}", flush=True)
    print("=" * 90, flush=True)

    # 1. Load clean gallery (300 templates)
    gallery = load_300_clean_gallery()
    gallery_fte_identities = {g["subject_id"] for g in gallery if g["actual_minutiae_count"] < 2}
    print(f"Gallery FTE identities (< 2 minutiae): {len(gallery_fte_identities)} identities", flush=True)

    # 2. Load 180 eligible queries (impressions 2-5)
    queries = load_180_eligible_queries()

    # 3. Execute parallel matching trials (Before and After on every query)
    print(f"\nLaunching parallel evaluation across {n_workers} worker processes...", flush=True)
    t_start = time.time()
    with Pool(processes=n_workers, initializer=init_worker, initargs=(gallery,)) as pool:
        trial_results = pool.map(process_query_trial, queries)
    elapsed_total = time.time() - t_start

    print(f"\nAll {len(trial_results)} queries evaluated (Before + After = {len(trial_results)*2} searches) in {elapsed_total:.2f}s ({elapsed_total/len(trial_results):.2f}s per query).", flush=True)

    # 4. Save per-query detailed CSV
    # Strip heavy impostor scores list from CSV export
    csv_rows = []
    for r in trial_results:
        row = dict(r)
        del row["before_impostor_scores"]
        del row["after_impostor_scores"]
        csv_rows.append(row)

    df_per_query = pd.DataFrame(csv_rows)
    per_query_csv_path = OUTPUT_DIR / "per_query_results.csv"
    df_per_query.to_csv(per_query_csv_path, index=False)
    print(f"[Saved] Per-query results written to: {per_query_csv_path.name}", flush=True)

    # 5. Compute metrics by subset: Combined, Bending, Elastic, Local
    subsets = {
        "Combined": trial_results,
        "Bending": [r for r in trial_results if r["distortion_type"] == "bending"],
        "Elastic": [r for r in trial_results if r["distortion_type"] == "elastic"],
        "Local": [r for r in trial_results if r["distortion_type"] == "local"],
    }

    all_metrics = {}
    summary_rows = []

    for name, subset in subsets.items():
        m = compute_metrics_for_subset(subset, name, gallery_fte_identities)
        all_metrics[name] = m
        summary_rows.append({
            "Distortion_Type": name,
            "Total_Queries": m["query_count"],
            "Before_Valid_Queries": m["before_valid_queries"],
            "Before_Top1_Accuracy_Pct": m["before_top1_accuracy_all_pct"],
            "Before_Top1_Count": f"{m['before_top1_correct_count']}/{m['query_count']}",
            "Before_Top5_Accuracy_Pct": m["before_top5_accuracy_all_pct"],
            "Before_Top5_Count": f"{m['before_top5_correct_count']}/{m['query_count']}",
            "Before_FAR_Pct": m["before_far_pct"],
            "Before_FRR_Pct": m["before_frr_pct"],
            "After_Valid_Queries": m["after_valid_queries"],
            "After_Top1_Accuracy_Pct": m["after_top1_accuracy_all_pct"],
            "After_Top1_Count": f"{m['after_top1_correct_count']}/{m['query_count']}",
            "After_Top5_Accuracy_Pct": m["after_top5_accuracy_all_pct"],
            "After_Top5_Count": f"{m['after_top5_correct_count']}/{m['query_count']}",
            "After_FAR_Pct": m["after_far_pct"],
            "After_FRR_Pct": m["after_frr_pct"],
            "Improved_Rank": m["improved_rank_count"],
            "Degraded_Rank": m["degraded_rank_count"],
            "Unchanged_Rank": m["unchanged_rank_count"],
        })

    # Save summary.csv
    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = OUTPUT_DIR / "summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    print(f"[Saved] Summary table written to: {summary_csv_path.name}", flush=True)

    # Save metrics.json
    metrics_json_path = OUTPUT_DIR / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"[Saved] Machine-readable metrics written to: {metrics_json_path.name}", flush=True)

    # Print comparative table to console
    print("\n" + "=" * 105, flush=True)
    print("                      BEFORE VS AFTER RECTIFICATION MATCHING ACCURACY", flush=True)
    print("=" * 105, flush=True)
    header = f"{'Distortion':<10} | {'Queries':<7} | {'Before Top-1':<12} | {'After Top-1':<12} | {'Before Top-5':<12} | {'After Top-5':<12} | {'Before FRR':<10} | {'After FRR':<10}"
    print(header, flush=True)
    print("-" * 105, flush=True)
    for r in summary_rows:
        line = (
            f"{r['Distortion_Type']:<10} | "
            f"{r['Total_Queries']:<7} | "
            f"{r['Before_Top1_Accuracy_Pct']:>5.2f}% ({r['Before_Top1_Count']}) | "
            f"{r['After_Top1_Accuracy_Pct']:>5.2f}% ({r['After_Top1_Count']}) | "
            f"{r['Before_Top5_Accuracy_Pct']:>5.2f}% ({r['Before_Top5_Count']}) | "
            f"{r['After_Top5_Accuracy_Pct']:>5.2f}% ({r['After_Top5_Count']}) | "
            f"{r['Before_FRR_Pct']:>8.2f}% | "
            f"{r['After_FRR_Pct']:>8.2f}%"
        )
        print(line, flush=True)
    print("=" * 105, flush=True)

    # Write REPORT.md
    report_path = OUTPUT_DIR / "REPORT.md"
    comb = all_metrics["Combined"]
    bnd = all_metrics["Bending"]
    ela = all_metrics["Elastic"]
    loc = all_metrics["Local"]

    report_content = f"""# DDRNet Fingerprint Rectification: Matching Accuracy Evaluation Report

**Project:** Fingerprint Distortion Rectification for Accurate Recognition  
**Evaluation Target:** 180 Eligible Test Queries (Impressions 2–5) from the Family Fingerprint Dataset  
**Clean Gallery:** 300 Enrolled Individual Identities (Impression 1, MySQL `fingerprint_authentication`)  
**Matching Algorithm:** SourceAFIS Invariant Edge Crawling (`matcher.py` / `MATCHER.identify_1toN`)  
**Operating Decision Threshold:** `{MATCH_THRESHOLD}` (`database.MATCH_THRESHOLD`)  
**Date:** {time.strftime('%Y-%m-%d')}  

---

## 1. Executive Summary

This evaluation measures whether DDRNet non-rigid geometric rectification improves SourceAFIS biometric identification of distorted family fingerprints.

- **Query Set:** Exactly **180 test queries** from impressions 2–5, corresponding to the 45 test persons across 15 test families (`FAMILY-13`, `FAMILY-14`, `FAMILY-15`, `FAMILY-20`, `FAMILY-23`, `FAMILY-36`, `FAMILY-37`, `FAMILY-42`, `FAMILY-48`, `FAMILY-54`, `FAMILY-65`, `FAMILY-66`, `FAMILY-67`, `FAMILY-78`, `FAMILY-9`).
- **Gallery Set:** Exactly **300 enrolled clean individual templates** (clean Impression 1 of all 300 persons in the database).
- **Zero Self-Matching:** Every query is physically disjoint from its clean enrolled gallery template (queries are impressions 2–5; gallery is impression 1).
- **Ground Truth:** Biometric identity is evaluated strictly at the **individual person level** (`identity_code = family_id/member_type`). Family and member categories are not treated as identities.

---

## 2. Before vs. After Rectification Performance Comparison

| Metric | Condition 1: BEFORE (Original Distorted) | Condition 2: AFTER (DDRNet Rectified) | Absolute Impact |
| :--- | :---: | :---: | :---: |
| **Total Test Queries** | **180** | **180** | — |
| **Clean Gallery Size (Identities)** | **300** | **300** | — |
| **Combined Top-1 Accuracy (All Queries)** | **{comb['before_top1_accuracy_all_pct']}%** ({comb['before_top1_correct_count']}/180) | **{comb['after_top1_accuracy_all_pct']}%** ({comb['after_top1_correct_count']}/180) | **{comb['after_top1_accuracy_all_pct'] - comb['before_top1_accuracy_all_pct']:+.2f}%** |
| **Combined Top-1 Accuracy (Valid Queries)** | **{comb['before_top1_accuracy_valid_pct']}%** ({comb['before_top1_correct_count']}/{comb['before_valid_queries']}) | **{comb['after_top1_accuracy_valid_pct']}%** ({comb['after_top1_correct_count']}/{comb['after_valid_queries']}) | **{comb['after_top1_accuracy_valid_pct'] - comb['before_top1_accuracy_valid_pct']:+.2f}%** |
| **Combined Top-5 Accuracy (All Queries)** | **{comb['before_top5_accuracy_all_pct']}%** ({comb['before_top5_correct_count']}/180) | **{comb['after_top5_accuracy_all_pct']}%** ({comb['after_top5_correct_count']}/180) | **{comb['after_top5_accuracy_all_pct'] - comb['before_top5_accuracy_all_pct']:+.2f}%** |
| **Combined Top-5 Accuracy (Valid Queries)** | **{comb['before_top5_accuracy_valid_pct']}%** ({comb['before_top5_correct_count']}/{comb['before_valid_queries']}) | **{comb['after_top5_accuracy_valid_pct']}%** ({comb['after_top5_correct_count']}/{comb['after_valid_queries']}) | **{comb['after_top5_accuracy_valid_pct'] - comb['before_top5_accuracy_valid_pct']:+.2f}%** |
| **False Rejection Rate (FRR @ Thresh {MATCH_THRESHOLD})** | **{comb['before_frr_pct']}%** ({comb['before_false_rejections_count']}/180) | **{comb['after_frr_pct']}%** ({comb['after_false_rejections_count']}/180) | **{comb['after_frr_pct'] - comb['before_frr_pct']:+.2f}%** |
| **False Acceptance Rate (FAR @ Thresh {MATCH_THRESHOLD})** | **{comb['before_far_pct']}%** ({comb['before_false_acceptances_count']}/{comb['impostor_trials_count']}) | **{comb['after_far_pct']}%** ({comb['after_false_acceptances_count']}/{comb['impostor_trials_count']}) | **{comb['after_far_pct'] - comb['before_far_pct']:+.4f}%** |
| **Mean Genuine Match Score** | **{comb['before_mean_genuine_score']}** | **{comb['after_mean_genuine_score']}** | **{comb['after_mean_genuine_score'] - comb['before_mean_genuine_score']:+.2f}** |
| **Median Genuine Match Score** | **{comb['before_median_genuine_score']}** | **{comb['after_median_genuine_score']}** | **{comb['after_median_genuine_score'] - comb['before_median_genuine_score']:+.2f}** |
| **Valid Query Coverage (Minutiae $\ge 2$)** | **{comb['before_valid_queries']}/180** ({comb['before_failed_queries_fta']} FTA) | **{comb['after_valid_queries']}/180** ({comb['after_failed_queries_fta']} FTA) | **{comb['after_valid_queries'] - comb['before_valid_queries']:+d} queries** |
| **Per-Query Rank Improvement** | — | **{comb['improved_rank_count']} improved, {comb['degraded_rank_count']} degraded, {comb['unchanged_rank_count']} unchanged** | — |

---

## 3. Results by Distortion Type

### 3.1 Bending Distortion (60 Queries)
- **Before Rectification:**
  - Top-1 Accuracy: **{bnd['before_top1_accuracy_all_pct']}%** ({bnd['before_top1_correct_count']}/60 all, {bnd['before_top1_accuracy_valid_pct']}% valid)
  - Top-5 Accuracy: **{bnd['before_top5_accuracy_all_pct']}%** ({bnd['before_top5_correct_count']}/60 all, {bnd['before_top5_accuracy_valid_pct']}% valid)
  - FRR: **{bnd['before_frr_pct']}%** | FAR: **{bnd['before_far_pct']}%** (Trials: 60 genuine, {bnd['impostor_trials_count']} impostor)
- **After Rectification:**
  - Top-1 Accuracy: **{bnd['after_top1_accuracy_all_pct']}%** ({bnd['after_top1_correct_count']}/60 all, {bnd['after_top1_accuracy_valid_pct']}% valid)
  - Top-5 Accuracy: **{bnd['after_top5_accuracy_all_pct']}%** ({bnd['after_top5_correct_count']}/60 all, {bnd['after_top5_accuracy_valid_pct']}% valid)
  - FRR: **{bnd['after_frr_pct']}%** | FAR: **{bnd['after_far_pct']}%** (Trials: 60 genuine, {bnd['impostor_trials_count']} impostor)
  - Rank Shifts: **{bnd['improved_rank_count']} improved, {bnd['degraded_rank_count']} degraded, {bnd['unchanged_rank_count']} unchanged**

### 3.2 Elastic Distortion (59 Queries)
- **Before Rectification:**
  - Top-1 Accuracy: **{ela['before_top1_accuracy_all_pct']}%** ({ela['before_top1_correct_count']}/59 all, {ela['before_top1_accuracy_valid_pct']}% valid)
  - Top-5 Accuracy: **{ela['before_top5_accuracy_all_pct']}%** ({ela['before_top5_correct_count']}/59 all, {ela['before_top5_accuracy_valid_pct']}% valid)
  - FRR: **{ela['before_frr_pct']}%** | FAR: **{ela['before_far_pct']}%** (Trials: 59 genuine, {ela['impostor_trials_count']} impostor)
- **After Rectification:**
  - Top-1 Accuracy: **{ela['after_top1_accuracy_all_pct']}%** ({ela['after_top1_correct_count']}/59 all, {ela['after_top1_accuracy_valid_pct']}% valid)
  - Top-5 Accuracy: **{ela['after_top5_accuracy_all_pct']}%** ({ela['after_top5_correct_count']}/59 all, {ela['after_top5_accuracy_valid_pct']}% valid)
  - FRR: **{ela['after_frr_pct']}%** | FAR: **{ela['after_far_pct']}%** (Trials: 59 genuine, {ela['impostor_trials_count']} impostor)
  - Rank Shifts: **{ela['improved_rank_count']} improved, {ela['degraded_rank_count']} degraded, {ela['unchanged_rank_count']} unchanged**

### 3.3 Local Distortion (61 Queries)
- **Before Rectification:**
  - Top-1 Accuracy: **{loc['before_top1_accuracy_all_pct']}%** ({loc['before_top1_correct_count']}/61 all, {loc['before_top1_accuracy_valid_pct']}% valid)
  - Top-5 Accuracy: **{loc['before_top5_accuracy_all_pct']}%** ({loc['before_top5_correct_count']}/61 all, {loc['before_top5_accuracy_valid_pct']}% valid)
  - FRR: **{loc['before_frr_pct']}%** | FAR: **{loc['before_far_pct']}%** (Trials: 61 genuine, {loc['impostor_trials_count']} impostor)
- **After Rectification:**
  - Top-1 Accuracy: **{loc['after_top1_accuracy_all_pct']}%** ({loc['after_top1_correct_count']}/61 all, {loc['after_top1_accuracy_valid_pct']}% valid)
  - Top-5 Accuracy: **{loc['after_top5_accuracy_all_pct']}%** ({loc['after_top5_correct_count']}/61 all, {loc['after_top5_accuracy_valid_pct']}% valid)
  - FRR: **{loc['after_frr_pct']}%** | FAR: **{loc['after_far_pct']}%** (Trials: 61 genuine, {loc['impostor_trials_count']} impostor)
  - Rank Shifts: **{loc['improved_rank_count']} improved, {loc['degraded_rank_count']} degraded, {loc['unchanged_rank_count']} unchanged**

---

## 4. Verification Protocol and Metric Formulas

### 4.1 Identification Accuracy (Top-1 and Top-5)
- **Top-1 Accuracy:**
  $$\\text{{Top-1}} = \\frac{{\\sum_{{i=1}}^{{N}} \\mathbb{{I}}(\\text{{genuine\\_rank}}_i = 1)}}{{N}} \\times 100\\%$$
- **Top-5 Accuracy:**
  $$\\text{{Top-5}} = \\frac{{\\sum_{{i=1}}^{{N}} \\mathbb{{I}}(\\text{{genuine\\_rank}}_i \\le 5)}}{{N}} \\times 100\\%$$

### 4.2 Verification Rates (FAR and FRR)
The verification protocol evaluates each query against the decision threshold $T = {MATCH_THRESHOLD}$:
- **Genuine Trials ($N_{{g}} = 180$):**
  Each probe is compared to the clean enrolled template of the same person.
  $$\\text{{FRR}}(T) = \\frac{{\\sum_{{i=1}}^{{N_g}} \\mathbb{{I}}(\\text{{genuine\\_score}}_i < T)}}{{N_g}} \\times 100\\%$$
- **Impostor Trials ($N_{{imp}} = 180 \\times 299 = 53,820$ comparisons):**
  Each probe is compared against the clean enrolled templates of all other 299 persons.
  $$\\text{{FAR}}(T) = \\frac{{\\sum_{{i=1}}^{{N_g}} \\sum_{{j \\ne \\text{{target}}}} \\mathbb{{I}}(\\text{{impostor\\_score}}_{{i,j}} \\ge T)}}{{N_{{imp}}}} \\times 100\\%$$

---

## 5. Artifacts and Reproduction Command

The evaluation was executed using the standalone script:
```powershell
.venv\\Scripts\\python experiments/family_rectification_matching_evaluation/run_evaluation.py
```

All generated files are strictly preserved in:
- [`experiments/family_rectification_matching_evaluation/REPORT.md`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/family_rectification_matching_evaluation/REPORT.md)
- [`experiments/family_rectification_matching_evaluation/summary.csv`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/family_rectification_matching_evaluation/summary.csv)
- [`experiments/family_rectification_matching_evaluation/per_query_results.csv`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/family_rectification_matching_evaluation/per_query_results.csv)
- [`experiments/family_rectification_matching_evaluation/metrics.json`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/family_rectification_matching_evaluation/metrics.json)
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[Saved] Detailed report written to: {report_path.name}", flush=True)


if __name__ == "__main__":
    main()
