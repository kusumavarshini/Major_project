#!/usr/bin/env python3
"""
Standalone Diagnostic Evaluation Script for the 1,500-Record Family Fingerprint Dataset.
Database: fingerprint_authentication (tables: fingerprints, persons).

Protocol:
- Closed-set 1:N Identification with 300 unique individual identities across 100 families.
- Gallery: Impression 1 (300 enrolled templates, 1 per identity).
- Probes: Impression 2 (300 test queries), plus cross-impression checks (Impressions 3, 4, 5).
- Zero self-matching: The probe template is strictly separate from the gallery template.
- Direct template deserialization: Uses stored minutiae templates directly without loading images.
- Complete trial recording: Computes Top-1, Top-5, Genuine/Impostor score distributions, FAR, FRR, and EER.
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

import mysql.connector
from matcher import MATCHER, _fast_crawl, compute_minutiae_feature_score
import afis.algorithms.safis as safis

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "diagnostic_evaluation_1500"

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
DB_NAME = "fingerprint_authentication"


def get_mysql_conn():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=DB_NAME,
    )


def pre_index_template(tpl_dict: Dict[str, Any], params: safis.SafisParams) -> None:
    """Pre-computes invariant edges, length buckets, and adjacency for fast 1:N search."""
    minu = tpl_dict.get("minutiae")
    if minu is None or len(minu) < 2:
        tpl_dict["_edges"] = []
        tpl_dict["_buckets"] = {}
        tpl_dict["_adj"] = {}
        return

    edges = safis.build_edges(minu, params)
    buckets: Dict[int, List[safis._Edge]] = {}
    adj: Dict[int, List[safis._Edge]] = {}
    lt = float(params.length_tol) or 1e-6

    for e in edges:
        k = int(round(e.length / lt))
        buckets.setdefault(k, []).append(e)
        adj.setdefault(e.ref, []).append(e)

    tpl_dict["_edges"] = edges
    tpl_dict["_buckets"] = buckets
    tpl_dict["_adj"] = adj
    tpl_dict["_cache_params_key"] = (
        params.max_neighbors,
        params.max_edge_length,
        params.length_tol,
        params.angle_tol,
    )


def load_all_family_templates() -> List[Dict[str, Any]]:
    """Loads and deserializes all 1,500 fingerprint records from fingerprint_authentication."""
    conn = get_mysql_conn()
    cur = conn.cursor(dictionary=True)
    query = """
        SELECT 
            f.fingerprint_id,
            f.person_id,
            p.identity_code,
            p.family_id,
            p.member_type,
            f.impression_id,
            f.image_path,
            f.minutiae_count,
            f.image_width,
            f.image_height,
            f.minutiae_template
        FROM fingerprints f
        JOIN persons p ON f.person_id = p.person_id
        ORDER BY f.person_id, f.impression_id
    """
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    records = []
    for r in rows:
        raw_blob = r["minutiae_template"]
        tpl_obj = pickle.loads(raw_blob)
        minu = getattr(tpl_obj, "minutiae", None)
        if minu is None and isinstance(tpl_obj, dict):
            minu = tpl_obj.get("minutiae")

        tpl_dict = {
            "minutiae": minu,
            "minutiae_count": len(minu) if minu is not None else 0,
            "image_width": r["image_width"] or 224,
            "image_height": r["image_height"] or 224,
        }
        pre_index_template(tpl_dict, MATCHER.params)

        records.append({
            "fingerprint_id": int(r["fingerprint_id"]),
            "person_id": int(r["person_id"]),
            "identity_code": str(r["identity_code"]),
            "family_id": str(r["family_id"]),
            "member_type": str(r["member_type"]),
            "impression_id": int(r["impression_id"]),
            "image_path": str(r["image_path"]),
            "db_minutiae_count": r["minutiae_count"],
            "actual_minutiae_count": len(minu) if minu is not None else 0,
            "template": tpl_dict,
        })
    return records


def run_identification_experiment(
    gallery_records: List[Dict[str, Any]],
    probe_records: List[Dict[str, Any]],
    protocol_name: str,
) -> Tuple[pd.DataFrame, Dict[str, Any], List[float], List[float]]:
    """
    Evaluates probe records against enrolled gallery records.
    Returns:
      (results_df, metrics_summary, genuine_scores, impostor_scores)
    """
    print(f"\nRunning Identification Protocol: {protocol_name}")
    print(f"Gallery size: {len(gallery_records)} identities | Probe size: {len(probe_records)} queries")

    # Map gallery person_id -> gallery record
    gallery_person_map = {g["person_id"]: g for g in gallery_records}

    rows = []
    genuine_scores = []
    impostor_scores = []

    rank1_count = 0
    rank5_count = 0
    valid_queries_count = 0
    fta_queries_count = 0
    fte_gallery_count = 0

    # Pre-index gallery list for MATCHER
    gallery_for_matcher = [
        {
            "fingerprint_id": g["fingerprint_id"],
            "person_id": g["person_id"],
            "subject_id": g["identity_code"],
            "template": g["template"],
        }
        for g in gallery_records
    ]

    for idx, probe in enumerate(probe_records, start=1):
        p_tpl = probe["template"]
        p_minu = p_tpl.get("minutiae")
        target_person_id = probe["person_id"]
        target_identity = probe["identity_code"]

        # Check if probe is Feature Extraction Failure (FTA)
        is_probe_fta = (p_minu is None or len(p_minu) < 2)
        # Check if corresponding enrolled gallery template is FTE
        target_gallery_record = gallery_person_map.get(target_person_id)
        is_gallery_fte = False
        if target_gallery_record:
            g_minu = target_gallery_record["template"].get("minutiae")
            if g_minu is None or len(g_minu) < 2:
                is_gallery_fte = True

        if is_probe_fta:
            fta_queries_count += 1
        if is_gallery_fte:
            fte_gallery_count += 1

        t0 = time.time()
        search_res = MATCHER.identify_1toN(p_tpl, gallery_for_matcher, top_n=len(gallery_for_matcher))
        match_time_ms = (time.time() - t0) * 1000.0

        candidates = search_res["candidates"]
        
        # Determine candidate ranks
        pred_top1_id = candidates[0]["subject_id"] if candidates else None
        pred_top1_score = candidates[0]["raw_score"] if candidates else 0.0

        # Find genuine candidate rank and score
        genuine_rank = None
        genuine_score = 0.0
        genuine_matched_minutiae = 0

        for rk, c in enumerate(candidates, start=1):
            if c["subject_id"] == target_identity:
                genuine_rank = rk
                genuine_score = float(c["raw_score"])
                genuine_matched_minutiae = c.get("matched_minutiae_count", 0)
                break

        # Record genuine score
        if target_gallery_record:
            genuine_scores.append(genuine_score)

        # Collect top impostor scores for distribution (take top 5 impostor candidates)
        for c in candidates[:10]:
            if c["subject_id"] != target_identity:
                impostor_scores.append(float(c["raw_score"]))

        # Check Top-1 and Top-5 success
        is_top1 = (genuine_rank == 1)
        is_top5 = (genuine_rank is not None and genuine_rank <= 5)

        if is_top1:
            rank1_count += 1
        if is_top5:
            rank5_count += 1

        rows.append({
            "probe_fingerprint_id": probe["fingerprint_id"],
            "person_id": target_person_id,
            "identity_code": target_identity,
            "family_id": probe["family_id"],
            "member_type": probe["member_type"],
            "probe_impression_id": probe["impression_id"],
            "probe_minutiae_count": probe["actual_minutiae_count"],
            "is_probe_fta": is_probe_fta,
            "is_gallery_fte": is_gallery_fte,
            "predicted_top1_identity": pred_top1_id,
            "predicted_top1_score": pred_top1_score,
            "genuine_rank": genuine_rank,
            "genuine_score": genuine_score,
            "genuine_matched_minutiae": genuine_matched_minutiae,
            "is_top1_correct": is_top1,
            "is_top5_correct": is_top5,
            "match_time_ms": round(match_time_ms, 2),
        })

    df = pd.DataFrame(rows)
    n_queries = len(probe_records)

    # Compute metrics: All queries vs Valid queries (excluding FTA/FTE)
    top1_acc_all = (rank1_count / n_queries) * 100.0
    top5_acc_all = (rank5_count / n_queries) * 100.0

    valid_mask = (~df["is_probe_fta"]) & (~df["is_gallery_fte"])
    valid_df = df[valid_mask]
    n_valid = len(valid_df)

    if n_valid > 0:
        top1_acc_valid = (valid_df["is_top1_correct"].sum() / n_valid) * 100.0
        top5_acc_valid = (valid_df["is_top5_correct"].sum() / n_valid) * 100.0
        mean_valid_genuine_score = float(valid_df["genuine_score"].mean())
    else:
        top1_acc_valid = 0.0
        top5_acc_valid = 0.0
        mean_valid_genuine_score = 0.0

    # Verification metrics: FAR and FRR at different thresholds
    # Genuine trial: genuine_score >= T
    # Impostor trial: impostor_score >= T
    thresholds = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0]
    verif_metrics = {}
    gen_arr = np.array(genuine_scores)
    imp_arr = np.array(impostor_scores)

    for th in thresholds:
        frr = float(np.mean(gen_arr < th) * 100.0) if len(gen_arr) > 0 else 100.0
        far = float(np.mean(imp_arr >= th) * 100.0) if len(imp_arr) > 0 else 0.0
        verif_metrics[f"threshold_{int(th)}"] = {
            "threshold": th,
            "FRR_percent": round(frr, 2),
            "FAR_percent": round(far, 4),
        }

    summary = {
        "protocol": protocol_name,
        "gallery_size": len(gallery_records),
        "total_queries": n_queries,
        "probe_fta_count": fta_queries_count,
        "gallery_fte_count": fte_gallery_count,
        "valid_query_trials": n_valid,
        "strict_top1_accuracy_percent": round(top1_acc_all, 2),
        "strict_top1_correct_count": rank1_count,
        "strict_top5_accuracy_percent": round(top5_acc_all, 2),
        "strict_top5_correct_count": rank5_count,
        "valid_top1_accuracy_percent": round(top1_acc_valid, 2),
        "valid_top1_correct_count": int(valid_df["is_top1_correct"].sum()),
        "valid_top5_accuracy_percent": round(top5_acc_valid, 2),
        "valid_top5_correct_count": int(valid_df["is_top5_correct"].sum()),
        "mean_genuine_score_overall": round(float(np.mean(gen_arr)), 2) if len(gen_arr) > 0 else 0.0,
        "mean_genuine_score_valid": round(mean_valid_genuine_score, 2),
        "mean_top_impostor_score": round(float(np.mean(imp_arr)), 2) if len(imp_arr) > 0 else 0.0,
        "verification_metrics_by_threshold": verif_metrics,
    }

    return df, summary, genuine_scores, impostor_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1,500 Family Fingerprint Dataset")
    parser.add_argument(
        "--full-cross-impression",
        action="store_true",
        help="Also run Impressions 3, 4, 5 against Impression 1",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 90)
    print("   EVALUATING 1,500-RECORD FAMILY FINGERPRINT DATASET (MySQL: fingerprint_authentication)")
    print("=" * 90)

    # 1. Load all records
    print("\n[Step 1/3] Loading all 1,500 records and deserializing templates...")
    t0 = time.time()
    records = load_all_family_templates()
    t_load = time.time() - t0
    print(f"[OK] Successfully loaded and deserialized {len(records)} templates in {t_load:.2f}s")

    # Group by impression_id
    by_impression: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        by_impression.setdefault(r["impression_id"], []).append(r)

    for imp_id in sorted(by_impression.keys()):
        zero_cnt = sum(1 for r in by_impression[imp_id] if r["actual_minutiae_count"] == 0)
        print(f"  - Impression {imp_id}: {len(by_impression[imp_id])} records ({zero_cnt} zero-minutiae / FTA)")

    # 2. Primary Benchmark: Gallery = Impression 1 (300 identities), Probes = Impression 2 (300 queries)
    print("\n[Step 2/3] Executing Primary Closed-Set Benchmark (Impression 2 probes vs Impression 1 gallery)...")
    gallery_imp1 = by_impression[1]
    probes_imp2 = by_impression[2]

    df_res, summary, gen_scores, imp_scores = run_identification_experiment(
        gallery_records=gallery_imp1,
        probe_records=probes_imp2,
        protocol_name="Impression 2 (probe) vs Impression 1 (gallery) - 300 unique identities",
    )

    # 3. Optional Multi-Impression Evaluation (Impressions 3, 4, 5)
    multi_summaries = {"primary_imp2_vs_imp1": summary}
    if args.full_cross_impression:
        for probe_imp_id in (3, 4, 5):
            _, sub_summary, _, _ = run_identification_experiment(
                gallery_records=gallery_imp1,
                probe_records=by_impression[probe_imp_id],
                protocol_name=f"Impression {probe_imp_id} (probe) vs Impression 1 (gallery)",
            )
            multi_summaries[f"imp{probe_imp_id}_vs_imp1"] = sub_summary

    # Save CSV and JSON
    csv_out = OUTPUT_DIR / "per_query_family_1500_results.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\n[Saved] Per-query results written to: {csv_out}")

    json_out = OUTPUT_DIR / "family_1500_metrics.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(multi_summaries, f, indent=2)
    print(f"[Saved] Summary metrics written to: {json_out}")

    # Print Report Table
    print("\n" + "=" * 90)
    print("                    FAMILY FINGERPRINT DATASET IDENTIFICATION RESULTS")
    print("=" * 90)
    print(f"Enrolled Gallery Size           : {summary['gallery_size']} identities (Impression 1)")
    print(f"Probe Queries Evaluated         : {summary['total_queries']} queries (Impression 2)")
    print(f"Probe FTA Records (0 minutiae)  : {summary['probe_fta_count']}")
    print(f"Gallery FTE Records (0 minutiae): {summary['gallery_fte_count']}")
    print(f"Valid Query Trials Evaluated    : {summary['valid_query_trials']}")
    print("-" * 90)
    print(f"Strict Top-1 Identification Acc : {summary['strict_top1_accuracy_percent']}% ({summary['strict_top1_correct_count']}/{summary['total_queries']})")
    print(f"Strict Top-5 Identification Acc : {summary['strict_top5_accuracy_percent']}% ({summary['strict_top5_correct_count']}/{summary['total_queries']})")
    print(f"Valid Top-1 Identification Acc  : {summary['valid_top1_accuracy_percent']}% ({summary['valid_top1_correct_count']}/{summary['valid_query_trials']})")
    print(f"Valid Top-5 Identification Acc  : {summary['valid_top5_accuracy_percent']}% ({summary['valid_top5_correct_count']}/{summary['valid_query_trials']})")
    print(f"Mean Genuine Match Score (Valid): {summary['mean_genuine_score_valid']}")
    print(f"Mean Top Impostor Match Score   : {summary['mean_top_impostor_score']}")
    print("-" * 90)
    print("Verification Operating Points (FAR vs FRR):")
    for k, v in summary["verification_metrics_by_threshold"].items():
        print(f"  Threshold {v['threshold']:>5.1f} | FRR: {v['FRR_percent']:>6.2f}% | FAR: {v['FAR_percent']:>7.4f}%")
    print("=" * 90)


if __name__ == "__main__":
    main()
