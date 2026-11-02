#!/usr/bin/env python3
"""
Standalone Fingerprint Matching Evaluation on the 1,500-Record Family Dataset.
Database: fingerprint_authentication (MySQL)
Tables: fingerprints, persons

Strict Constraints:
- Uses existing SourceAFIS matching implementation (matcher.py / MATCHER.identify_1toN).
- Zero modifications to matcher.py, backend_api.py, database, .env, or model checkpoints.
- Uses stored minutiae templates directly from fingerprints.minutiae_template without loading images.
- Valid closed-set identification: Probe impression vs Enrolled gallery impression across 300 identities.
- Probe template is strictly excluded from gallery (zero self-match).
- Evaluates Top-1, Top-5, valid/failed queries, genuine/impostor score distributions, and FAR/FRR.
- Evaluates Impression 2 vs Impression 1 (primary), plus cross-impression checks (Impressions 3, 4, 5 vs 1, and 1 vs 2).
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
from multiprocessing import Pool, cpu_count
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

import mysql.connector
from matcher import MATCHER, SourceAFISMatcher
import afis.algorithms.safis as safis
from database import MATCH_THRESHOLD

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "family_1500_evaluation"

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


def load_all_records() -> List[Dict[str, Any]]:
    """Loads all 1,500 records from fingerprint_authentication and deserializes stored templates."""
    print("Loading all 1,500 fingerprint records from MySQL database 'fingerprint_authentication'...", flush=True)
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
        ORDER BY p.person_id, f.impression_id
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    records = []
    failed_deserialization = 0
    zero_minutiae_count = 0

    for r in rows:
        raw_blob = r["minutiae_template"]
        try:
            t_obj = pickle.loads(raw_blob)
            minu = getattr(t_obj, "minutiae", None)
            if minu is None and isinstance(t_obj, dict):
                minu = t_obj.get("minutiae")
        except Exception as e:
            failed_deserialization += 1
            minu = None

        m_len = len(minu) if minu is not None else 0
        if m_len == 0:
            zero_minutiae_count += 1

        tpl = {
            "minutiae": minu,
            "minutiae_count": m_len,
        }

        records.append({
            "fingerprint_id": int(r["fingerprint_id"]),
            "person_id": int(r["person_id"]),
            "identity_code": str(r["identity_code"]),
            "family_id": str(r["family_id"]),
            "member_type": str(r["member_type"]),
            "impression_id": int(r["impression_id"]),
            "image_path": str(r["image_path"]),
            "db_minutiae_count": r["minutiae_count"],
            "actual_minutiae_count": m_len,
            "template": tpl,
        })

    elapsed = time.time() - t0
    print(f"Loaded {len(records)} records in {elapsed:.2f}s.", flush=True)
    print(f"  - Deserialization failures: {failed_deserialization}", flush=True)
    print(f"  - Zero-minutiae records: {zero_minutiae_count} / {len(records)}", flush=True)
    return records


def init_worker(gallery_data: List[Dict[str, Any]]) -> None:
    """Worker process initializer to set the shared read-only gallery."""
    global GLOBAL_GALLERY
    GLOBAL_GALLERY = gallery_data


def process_single_probe(probe: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker task: Evaluates 1 probe against the global gallery using MATCHER.identify_1toN.
    """
    global GLOBAL_GALLERY
    target_identity = probe["identity_code"]
    p_tpl = probe["template"]
    p_minu = p_tpl.get("minutiae")
    is_probe_fta = (p_minu is None or len(p_minu) < 2)

    t0 = time.time()
    # Call existing unmodified MATCHER.identify_1toN
    res = MATCHER.identify_1toN(p_tpl, GLOBAL_GALLERY, top_n=len(GLOBAL_GALLERY), threshold=MATCH_THRESHOLD)
    search_ms = (time.time() - t0) * 1000.0

    candidates = res.get("candidates", [])

    ranks = {c["subject_id"]: idx + 1 for idx, c in enumerate(candidates)}
    genuine_rank = ranks.get(target_identity, None)

    top1_cand = candidates[0] if candidates else None
    pred_top1_id = top1_cand["subject_id"] if top1_cand else None
    pred_top1_score = float(top1_cand["raw_score"]) if top1_cand else 0.0

    target_cand = next((c for c in candidates if c["subject_id"] == target_identity), None)
    genuine_score = float(target_cand["raw_score"]) if target_cand else 0.0
    genuine_minutiae_matched = int(target_cand.get("matched_minutiae", 0)) if target_cand else 0

    is_top1 = (genuine_rank == 1)
    is_top5 = (genuine_rank is not None and genuine_rank <= 5)

    # Top impostor score (best candidate from a different person)
    top_impostor_cand = next((c for c in candidates if c["subject_id"] != target_identity), None)
    top_impostor_score = float(top_impostor_cand["raw_score"]) if top_impostor_cand else 0.0

    return {
        "probe_fingerprint_id": probe["fingerprint_id"],
        "person_id": probe["person_id"],
        "identity_code": target_identity,
        "family_id": probe["family_id"],
        "member_type": probe["member_type"],
        "probe_impression_id": probe["impression_id"],
        "probe_minutiae_count": probe["actual_minutiae_count"],
        "is_probe_fta": is_probe_fta,
        "predicted_top1_identity": pred_top1_id,
        "predicted_top1_score": pred_top1_score,
        "genuine_rank": genuine_rank,
        "genuine_score": genuine_score,
        "genuine_matched_minutiae": genuine_minutiae_matched,
        "top_impostor_score": top_impostor_score,
        "is_top1_correct": is_top1,
        "is_top5_correct": is_top5,
        "search_time_ms": round(search_ms, 2),
    }


def evaluate_protocol(
    gallery_records: List[Dict[str, Any]],
    probe_records: List[Dict[str, Any]],
    protocol_name: str,
    num_workers: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes a closed-set 1:N identification protocol over the provided gallery and probe sets.
    """
    print(f"\n================================================================================", flush=True)
    print(f"PROTOCOL: {protocol_name}", flush=True)
    print(f"Gallery size: {len(gallery_records)} templates | Probe queries: {len(probe_records)} queries", flush=True)
    print(f"Parallel worker processes: {num_workers}", flush=True)
    print(f"================================================================================", flush=True)

    # Pre-index gallery templates
    print("Pre-indexing gallery templates with SourceAFIS invariant edges...", flush=True)
    t_idx0 = time.time()
    gallery_for_matcher = []
    gallery_fte_count = 0
    gallery_person_ids = set()

    for g in gallery_records:
        pre_index_template(g["template"], MATCHER.params)
        m_len = g["actual_minutiae_count"]
        if m_len < 2:
            gallery_fte_count += 1
        gallery_person_ids.add(g["person_id"])
        gallery_for_matcher.append({
            "fingerprint_id": g["fingerprint_id"],
            "person_id": g["person_id"],
            "subject_id": g["identity_code"],
            "family_id": g["family_id"],
            "member_type": g["member_type"],
            "template": g["template"],
        })
    print(f"Pre-indexing completed in {time.time() - t_idx0:.2f}s. Gallery FTE records (0 minutiae): {gallery_fte_count}", flush=True)

    # Count probe FTA
    probe_fta_count = sum(1 for p in probe_records if p["actual_minutiae_count"] < 2)
    print(f"Probe FTA records (0 minutiae): {probe_fta_count}", flush=True)

    # Run parallel evaluation
    t_start = time.time()
    with Pool(processes=num_workers, initializer=init_worker, initargs=(gallery_for_matcher,)) as pool:
        trial_results = pool.map(process_single_probe, probe_records)
    total_elapsed = time.time() - t_start

    print(f"All {len(trial_results)} queries evaluated in {total_elapsed:.2f}s (Avg {total_elapsed/len(trial_results):.2f}s per query).", flush=True)

    df = pd.DataFrame(trial_results)

    # Calculations
    n_queries = len(probe_records)
    top1_correct = int(df["is_top1_correct"].sum())
    top5_correct = int(df["is_top5_correct"].sum())

    top1_accuracy = (top1_correct / n_queries) * 100.0
    top5_accuracy = (top5_correct / n_queries) * 100.0

    # Valid queries (excluding probe FTA and gallery FTE)
    # A trial is valid if probe has >=2 minutiae AND its enrolled gallery template has >=2 minutiae
    valid_mask = (~df["is_probe_fta"])
    # Check if gallery has template with >= 2 minutiae for that person
    fte_person_ids = {g["person_id"] for g in gallery_records if g["actual_minutiae_count"] < 2}
    valid_mask = valid_mask & (~df["person_id"].isin(fte_person_ids))
    valid_df = df[valid_mask]
    n_valid = len(valid_df)

    if n_valid > 0:
        valid_top1_correct = int(valid_df["is_top1_correct"].sum())
        valid_top5_correct = int(valid_df["is_top5_correct"].sum())
        valid_top1_accuracy = (valid_top1_correct / n_valid) * 100.0
        valid_top5_accuracy = (valid_top5_correct / n_valid) * 100.0
        mean_valid_genuine_score = float(valid_df["genuine_score"].mean())
    else:
        valid_top1_correct = 0
        valid_top5_correct = 0
        valid_top1_accuracy = 0.0
        valid_top5_accuracy = 0.0
        mean_valid_genuine_score = 0.0

    # Score distributions
    genuine_scores = [r["genuine_score"] for r in trial_results]
    impostor_scores = [r["top_impostor_score"] for r in trial_results if r["top_impostor_score"] > 0]

    mean_genuine = float(np.mean(genuine_scores)) if genuine_scores else 0.0
    median_genuine = float(np.median(genuine_scores)) if genuine_scores else 0.0
    mean_impostor = float(np.mean(impostor_scores)) if impostor_scores else 0.0
    median_impostor = float(np.median(impostor_scores)) if impostor_scores else 0.0

    # Verification Operating Points (FAR and FRR)
    # Genuine trials: probe vs enrolled template of same subject (n_queries trials)
    # Impostor trials: probe vs top candidate of different subject
    gen_arr = np.array(genuine_scores)
    imp_arr = np.array(impostor_scores)

    verif_metrics = {}
    thresholds_to_test = [30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0]
    for th in thresholds_to_test:
        frr = float(np.mean(gen_arr < th) * 100.0) if len(gen_arr) > 0 else 100.0
        far = float(np.mean(imp_arr >= th) * 100.0) if len(imp_arr) > 0 else 0.0
        verif_metrics[f"threshold_{int(th)}"] = {
            "threshold": th,
            "FRR_percent": round(frr, 2),
            "FAR_percent": round(far, 4),
        }

    summary = {
        "protocol_name": protocol_name,
        "gallery_size": len(gallery_records),
        "total_queries_evaluated": n_queries,
        "probe_fta_failed_templates": probe_fta_count,
        "gallery_fte_failed_templates": gallery_fte_count,
        "skipped_queries": 0,
        "valid_query_trials": n_valid,
        "strict_overall_top1_accuracy_percent": round(top1_accuracy, 2),
        "strict_overall_top1_correct_count": top1_correct,
        "strict_overall_top5_accuracy_percent": round(top5_accuracy, 2),
        "strict_overall_top5_correct_count": top5_correct,
        "valid_query_top1_accuracy_percent": round(valid_top1_accuracy, 2),
        "valid_query_top1_correct_count": valid_top1_correct,
        "valid_query_top5_accuracy_percent": round(valid_top5_accuracy, 2),
        "valid_query_top5_correct_count": valid_top5_correct,
        "mean_genuine_score_overall": round(mean_genuine, 2),
        "median_genuine_score_overall": round(median_genuine, 2),
        "mean_genuine_score_valid": round(mean_valid_genuine_score, 2),
        "mean_top_impostor_score": round(mean_impostor, 2),
        "median_top_impostor_score": round(median_impostor, 2),
        "operating_threshold": MATCH_THRESHOLD,
        "frr_at_operating_threshold_50": verif_metrics.get("threshold_50", {}).get("FRR_percent"),
        "far_at_operating_threshold_50": verif_metrics.get("threshold_50", {}).get("FAR_percent"),
        "verification_metrics_by_threshold": verif_metrics,
        "total_runtime_seconds": round(total_elapsed, 2),
    }

    print(f"Results Summary for {protocol_name}:", flush=True)
    print(f"  - Strict Top-1 Accuracy: {summary['strict_overall_top1_accuracy_percent']}% ({top1_correct}/{n_queries})", flush=True)
    print(f"  - Strict Top-5 Accuracy: {summary['strict_overall_top5_accuracy_percent']}% ({top5_correct}/{n_queries})", flush=True)
    print(f"  - Valid Top-1 Accuracy : {summary['valid_query_top1_accuracy_percent']}% ({valid_top1_correct}/{n_valid})", flush=True)
    print(f"  - Valid Top-5 Accuracy : {summary['valid_query_top5_accuracy_percent']}% ({valid_top5_correct}/{n_valid})", flush=True)
    print(f"  - Mean Genuine Score   : {summary['mean_genuine_score_overall']}", flush=True)
    print(f"  - Mean Top Impostor    : {summary['mean_top_impostor_score']}", flush=True)
    print(f"  - FRR @ Threshold 50.0 : {summary['frr_at_operating_threshold_50']}%", flush=True)
    print(f"  - FAR @ Threshold 50.0 : {summary['far_at_operating_threshold_50']}%", flush=True)

    return df, summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate 1,500 Family Fingerprint Dataset (Unmodified SourceAFIS)")
    parser.add_argument("--workers", type=int, default=12, help="Number of worker processes (default: 12)")
    parser.add_argument("--evaluate-all-impressions", action="store_true", default=True, help="Evaluate all 5 impression combinations (default: True)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 90, flush=True)
    print("   STANDALONE EVALUATION OF CURRENT SOURCEAFIS MATCHER ON 1,500-RECORD FAMILY DATASET", flush=True)
    print("   Database: fingerprint_authentication | Tables: fingerprints, persons", flush=True)
    print("=" * 90, flush=True)

    # 1. Load all records
    records = load_all_records()

    # Group records by impression_id
    by_impression: Dict[int, List[Dict[str, Any]]] = {}
    for r in records:
        by_impression.setdefault(r["impression_id"], []).append(r)

    print("\nImpression Summary:", flush=True)
    for imp_id in sorted(by_impression.keys()):
        imp_recs = by_impression[imp_id]
        zero_cnt = sum(1 for r in imp_recs if r["actual_minutiae_count"] == 0)
        print(f"  Impression {imp_id}: {len(imp_recs)} records | Zero minutiae (FTA): {zero_cnt}", flush=True)

    # 2. Primary Evaluation Protocol: Impression 2 probes vs Impression 1 gallery
    # Enrolled Gallery: Impression 1 (300 identities)
    # Probes: Impression 2 (300 queries)
    gallery_imp1 = by_impression[1]
    probes_imp2 = by_impression[2]

    df_imp2, summary_imp2 = evaluate_protocol(
        gallery_records=gallery_imp1,
        probe_records=probes_imp2,
        protocol_name="Primary: Impression 2 (Probes) vs Impression 1 (Gallery) - 300 Identities",
        num_workers=args.workers,
    )

    # Save Primary CSV
    csv_imp2_path = OUTPUT_DIR / "per_query_predictions_imp2_vs_imp1.csv"
    df_imp2.to_csv(csv_imp2_path, index=False)
    print(f"\n[Saved] Primary per-query results saved to: {csv_imp2_path.name}", flush=True)

    all_summaries = {"primary_impression_2_vs_1": summary_imp2}
    all_dfs = [df_imp2]

    # 3. Optional/Complete evaluation across other impressions:
    if args.evaluate_all_impressions:
        # Impression 3 vs 1
        df_imp3, summary_imp3 = evaluate_protocol(
            gallery_records=gallery_imp1,
            probe_records=by_impression[3],
            protocol_name="Impression 3 (Probes) vs Impression 1 (Gallery) - 300 Identities",
            num_workers=args.workers,
        )
        all_summaries["impression_3_vs_1"] = summary_imp3
        all_dfs.append(df_imp3)

        # Impression 4 vs 1
        df_imp4, summary_imp4 = evaluate_protocol(
            gallery_records=gallery_imp1,
            probe_records=by_impression[4],
            protocol_name="Impression 4 (Probes) vs Impression 1 (Gallery) - 300 Identities",
            num_workers=args.workers,
        )
        all_summaries["impression_4_vs_1"] = summary_imp4
        all_dfs.append(df_imp4)

        # Impression 5 vs 1
        df_imp5, summary_imp5 = evaluate_protocol(
            gallery_records=gallery_imp1,
            probe_records=by_impression[5],
            protocol_name="Impression 5 (Probes) vs Impression 1 (Gallery) - 300 Identities",
            num_workers=args.workers,
        )
        all_summaries["impression_5_vs_1"] = summary_imp5
        all_dfs.append(df_imp5)

        # Impression 1 vs 2 (Reversed gallery/probe)
        df_imp1_vs_2, summary_imp1_vs_2 = evaluate_protocol(
            gallery_records=by_impression[2],
            probe_records=gallery_imp1,
            protocol_name="Impression 1 (Probes) vs Impression 2 (Gallery) - 300 Identities",
            num_workers=args.workers,
        )
        all_summaries["impression_1_vs_2_reversed"] = summary_imp1_vs_2

    # Save combined per-query predictions
    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv_path = OUTPUT_DIR / "per_query_predictions_all_impressions.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    print(f"[Saved] Combined per-query results ({len(combined_df)} trials) saved to: {combined_csv_path.name}", flush=True)

    # Save JSON summary metrics
    json_path = OUTPUT_DIR / "family_1500_evaluation_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"[Saved] Evaluation summary metrics saved to: {json_path.name}", flush=True)

    # Overall Summary across all 4 non-gallery impressions (Impressions 2, 3, 4, 5 vs Impression 1)
    total_non_gallery_trials = len(combined_df)
    total_non_gallery_top1 = int(combined_df["is_top1_correct"].sum())
    total_non_gallery_top5 = int(combined_df["is_top5_correct"].sum())
    overall_top1_pct = (total_non_gallery_top1 / total_non_gallery_trials) * 100.0
    overall_top5_pct = (total_non_gallery_top5 / total_non_gallery_trials) * 100.0

    print("\n" + "=" * 90, flush=True)
    print("                    FINAL EVALUATION SUMMARY RESULTS", flush=True)
    print("=" * 90, flush=True)
    print(f"Primary Benchmark (Impression 2 Probes vs Impression 1 Gallery, 300 Identities):", flush=True)
    print(f"  - Strict Top-1 Accuracy: {summary_imp2['strict_overall_top1_accuracy_percent']}% ({summary_imp2['strict_overall_top1_correct_count']}/300)", flush=True)
    print(f"  - Strict Top-5 Accuracy: {summary_imp2['strict_overall_top5_accuracy_percent']}% ({summary_imp2['strict_overall_top5_correct_count']}/300)", flush=True)
    print(f"  - Valid Top-1 Accuracy : {summary_imp2['valid_query_top1_accuracy_percent']}% ({summary_imp2['valid_query_top1_correct_count']}/{summary_imp2['valid_query_trials']})", flush=True)
    print(f"  - Valid Top-5 Accuracy : {summary_imp2['valid_query_top5_accuracy_percent']}% ({summary_imp2['valid_query_top5_correct_count']}/{summary_imp2['valid_query_trials']})", flush=True)
    print(f"  - FRR @ Threshold 50.0 : {summary_imp2['frr_at_operating_threshold_50']}%", flush=True)
    print(f"  - FAR @ Threshold 50.0 : {summary_imp2['far_at_operating_threshold_50']}%", flush=True)

    if args.evaluate_all_impressions:
        print(f"\nAll Cross-Impression Trials Combined (Impressions 2, 3, 4, 5 vs Impression 1 Gallery):", flush=True)
        print(f"  - Total Trials Evaluated: {total_non_gallery_trials}", flush=True)
        print(f"  - Overall Top-1 Accuracy: {overall_top1_pct:.2f}% ({total_non_gallery_top1}/{total_non_gallery_trials})", flush=True)
        print(f"  - Overall Top-5 Accuracy: {overall_top5_pct:.2f}% ({total_non_gallery_top5}/{total_non_gallery_trials})", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
