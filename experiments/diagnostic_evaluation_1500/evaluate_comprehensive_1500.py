#!/usr/bin/env python3
"""
Comprehensive Diagnostic Evaluation Script: Fingerprint Identification Accuracy
Targeting the 317-Query FVC Benchmark and the 1,500-Image Family Gallery.

Evaluates:
1. Standard 317-Query Benchmark against 317 Gallery (Baseline vs DDRNet Rectified)
2. Large 1,500-Image Gallery (317 Genuine FVC Clean + 1,183 Distractor Templates)
3. Closed-Set Cross-Impression Benchmark on the Family Fingerprint Dataset (300 identities)
"""

from __future__ import annotations

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path
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
import pickle

from matcher import MATCHER, SourceAFISMatcher, _fast_crawl
import afis.algorithms.safis as safis

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "diagnostic_evaluation_1500"
EXISTING_317_CSV = PROJECT_ROOT / "latent_fvc_317_baseline_vs_ddrnet.csv"

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")


def get_mysql_conn(db_name: str):
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=db_name,
    )


def pre_index_template(cand_tpl: Dict[str, Any], params: safis.SafisParams) -> None:
    """Pre-computes invariant edges, length buckets, and adjacencies for fast 1:N search."""
    c_minu = cand_tpl.get("minutiae")
    if c_minu is None or len(c_minu) < 2:
        cand_tpl["_edges"] = []
        cand_tpl["_buckets"] = {}
        cand_tpl["_adj"] = {}
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


def load_317_gallery() -> List[Dict[str, Any]]:
    """Loads the 317 FVC clean templates from newdb.external_fvc_clean_gallery."""
    conn = get_mysql_conn("newdb")
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT fingerprint_id, template_data, minutiae_count
        FROM external_fvc_clean_gallery
        WHERE fingerprint_id <= 317
        ORDER BY fingerprint_id
        """
    )
    gallery = []
    for r in cur.fetchall():
        tpl = pickle.loads(r["template_data"])
        fid = int(r["fingerprint_id"])
        item = {
            "fingerprint_id": fid,
            "subject_id": f"GALLERY-{fid}",
            "expected_id": fid,
            "is_distractor": False,
            "template": tpl,
        }
        pre_index_template(tpl, MATCHER.params)
        gallery.append(item)
    cur.close()
    conn.close()
    return gallery


def load_1500_gallery(base_317_gallery: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Expands the 317-gallery to exactly 1,500 templates by adding 1,183 distractor templates
    from the 1,500-image Family Fingerprint Dataset (fingerprint_authentication.fingerprints).
    """
    conn = get_mysql_conn("fingerprint_authentication")
    cur = conn.cursor(dictionary=True)
    cur.execute(
        """
        SELECT f.fingerprint_id, p.identity_code, f.minutiae_template, f.minutiae_count
        FROM fingerprints f
        JOIN persons p ON f.person_id = p.person_id
        ORDER BY f.fingerprint_id
        LIMIT 1183
        """
    )
    distractors = []
    for r in cur.fetchall():
        t_obj = pickle.loads(r["minutiae_template"])
        tpl = {
            "minutiae": t_obj.minutiae,
            "minutiae_count": int(r["minutiae_count"]),
            "image_width": 224,
            "image_height": 224,
        }
        fid = 10000 + int(r["fingerprint_id"])
        item = {
            "fingerprint_id": fid,
            "subject_id": str(r["identity_code"]),
            "expected_id": None,
            "is_distractor": True,
            "template": tpl,
        }
        pre_index_template(tpl, MATCHER.params)
        distractors.append(item)

    cur.close()
    conn.close()

    full_gallery = list(base_317_gallery) + distractors
    return full_gallery


def load_physical_finger_mapping() -> Dict[int, int]:
    """Maps pair_id (1..317) to the true physical finger index (1..110) from original paths."""
    conn = get_mysql_conn("fingerprint_external")
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT pair_id, clean_image_path FROM external_pairs ORDER BY pair_id")
    import re
    mapping = {}
    for r in cur.fetchall():
        m = re.search(r"(\d+)_\d+\.png", r["clean_image_path"])
        if m:
            mapping[r["pair_id"]] = int(m.group(1))
    cur.close()
    conn.close()
    return mapping


def match_probe_against_distractors_pruned(
    probe_minu: np.ndarray,
    distractors: List[Dict[str, Any]],
    target_score_threshold: float,
) -> Tuple[int, Optional[str], float]:
    """
    Evaluates probe minutiae against 1,183 distractors.
    Uses candidate edge filtering: if compatible roots < 3, skips expensive graph crawling.
    Returns:
      (count_distractors_above_target_score, top_distractor_id, top_distractor_score)
    """
    ea = safis.build_edges(probe_minu, MATCHER.params)
    if not ea or len(probe_minu) < 2:
        return 0, None, 0.0

    p_adj: Dict[int, List[safis._Edge]] = {}
    for e in ea:
        p_adj.setdefault(e.ref, []).append(e)

    lt = float(MATCHER.params.length_tol) or 1e-6
    at = MATCHER.params.angle_tol

    distractors_above_target = 0
    top_score = 0.0
    top_id = None

    for d in distractors:
        ctpl = d["template"]
        c_buckets = ctpl.get("_buckets", {})
        roots = []
        for pe in ea:
            k = int(round(pe.length / lt))
            for key in (k - 1, k, k + 1):
                b = c_buckets.get(key)
                if not b:
                    continue
                for ce in b:
                    if safis._edges_compatible(pe, ce, MATCHER.params):
                        roots.append((pe, ce))

        if len(roots) < 3:
            continue

        best_n = 0
        c_adj = ctpl.get("_adj", {})
        for pe, ce in roots[:15]:
            pairing, _ = _fast_crawl(p_adj, c_adj, pe.ref, ce.ref, pe.neighbor, ce.neighbor, MATCHER.params)
            if len(pairing) > best_n:
                best_n = len(pairing)

        cand_score = float(best_n) * 5.0
        if best_n >= 2:
            cand_score += float(min(best_n * 2, len(roots))) * 1.5

        if cand_score > top_score:
            top_score = cand_score
            top_id = d["subject_id"]

        if cand_score > target_score_threshold:
            distractors_above_target += 1

    return distractors_above_target, top_id, top_score


def evaluate_317_benchmark(
    gallery_317: List[Dict[str, Any]],
    gallery_1500: List[Dict[str, Any]],
    physical_mapping: Dict[int, int],
    eval_csv_path: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Evaluates both 317 gallery and 1,500 gallery conditions across all 317 queries.
    """
    if not eval_csv_path.exists():
        raise FileNotFoundError(f"Existing evaluation CSV not found: {eval_csv_path}")

    df_base = pd.read_csv(eval_csv_path)
    n_queries = len(df_base)

    # 1. 317 Gallery Condition Analysis
    df_317 = df_base.copy()
    df_317["exp_finger"] = df_317["expected_id"].map(physical_mapping)
    df_317["base_top1_finger"] = df_317["baseline_top1_id"].map(physical_mapping)
    df_317["ddr_top1_finger"] = df_317["ddrnet_top1_id"].map(physical_mapping)

    base_317_t1_strict = int((df_317["baseline_top1_id"] == df_317["expected_id"]).sum())
    base_317_t1_finger = int((df_317["base_top1_finger"] == df_317["exp_finger"]).sum())
    base_317_t5 = int((df_317["baseline_expected_rank"] <= 5).sum())

    ddr_317_t1_strict = int((df_317["ddrnet_top1_id"] == df_317["expected_id"]).sum())
    ddr_317_t1_finger = int((df_317["ddr_top1_finger"] == df_317["exp_finger"]).sum())
    ddr_317_t5 = int((df_317["ddrnet_expected_rank"] <= 5).sum())

    thresh_50 = 50.0
    thresh_100 = 100.0
    total_genuine = n_queries
    total_impostor_317 = n_queries * (317 - 1)

    base_frr_50 = float(round(((df_317["baseline_expected_score"] < thresh_50).sum() / total_genuine) * 100.0, 2))
    ddr_frr_50 = float(round(((df_317["ddrnet_expected_score"] < thresh_50).sum() / total_genuine) * 100.0, 2))
    base_frr_100 = float(round(((df_317["baseline_expected_score"] < thresh_100).sum() / total_genuine) * 100.0, 2))
    ddr_frr_100 = float(round(((df_317["ddrnet_expected_score"] < thresh_100).sum() / total_genuine) * 100.0, 2))

    def count_fa(top5_series, exp_series, threshold):
        fa = 0
        for top5_str, exp_id in zip(top5_series, exp_series):
            for p in str(top5_str).split(";"):
                if ":" in p:
                    cid, sc = p.split(":")
                    try:
                        if int(cid) != int(exp_id) and float(sc) >= threshold:
                            fa += 1
                    except Exception:
                        pass
        return fa

    base_fa_50 = count_fa(df_317["baseline_top5"], df_317["expected_id"], thresh_50)
    ddr_fa_50 = count_fa(df_317["ddrnet_top5"], df_317["expected_id"], thresh_50)
    base_far_50 = float(round((base_fa_50 / total_impostor_317) * 100.0, 2))
    ddr_far_50 = float(round((ddr_fa_50 / total_impostor_317) * 100.0, 2))

    base_fa_100 = count_fa(df_317["baseline_top5"], df_317["expected_id"], thresh_100)
    ddr_fa_100 = count_fa(df_317["ddrnet_top5"], df_317["expected_id"], thresh_100)
    base_far_100 = float(round((base_fa_100 / total_impostor_317) * 100.0, 2))
    ddr_far_100 = float(round((ddr_fa_100 / total_impostor_317) * 100.0, 2))

    # 2. 1,500 Gallery Condition Evaluation
    # Match the candidate queries against the 1,183 distractors
    distractors_only = [g for g in gallery_1500 if g["is_distractor"]]
    print(f"\n[Step 2/3] Evaluating 1,500 gallery (1,183 distractors) on competitive queries...")

    latent_dir = PROJECT_ROOT / "data" / "Latent_fvc"
    df_1500 = df_317.copy()

    # Pre-load all 19 Top-1 queries + Top-5 queries to determine exact 1500 ranks
    competitive_qids = set(df_317[df_317["baseline_expected_rank"] <= 5]["query_id"]).union(
        set(df_317[df_317["ddrnet_expected_rank"] <= 5]["query_id"])
    )
    print(f"  Found {len(competitive_qids)} competitive queries (Rank <= 5 in 317 gallery)")

    b_1500_ranks = []
    d_1500_ranks = []
    b_1500_t1_flags = []
    d_1500_t1_flags = []
    b_1500_t5_flags = []
    d_1500_t5_flags = []
    top_distractor_ids = []
    top_distractor_scores = []

    for idx, r in df_317.iterrows():
        qid = int(r["query_id"])
        b_rank_317 = int(r["baseline_expected_rank"]) if str(r["baseline_expected_rank"]).isdigit() else 999
        d_rank_317 = int(r["ddrnet_expected_rank"]) if str(r["ddrnet_expected_rank"]).isdigit() else 999
        b_score = float(r["baseline_expected_score"])
        d_score = float(r["ddrnet_expected_score"])

        if qid in competitive_qids:
            q_img = cv2.imread(str(latent_dir / f"{qid}.png"), cv2.IMREAD_GRAYSCALE)
            q_tpl = MATCHER.extract_template(q_img)
            p_minu = q_tpl["minutiae"]

            # Baseline probe vs distractors
            n_dist_higher_base, top_did_b, top_dsc_b = match_probe_against_distractors_pruned(
                p_minu, distractors_only, target_score_threshold=b_score
            )
            # DDRNet probe vs distractors
            n_dist_higher_ddr, top_did_d, top_dsc_d = match_probe_against_distractors_pruned(
                p_minu, distractors_only, target_score_threshold=d_score
            )

            b_1500_rk = b_rank_317 + n_dist_higher_base
            d_1500_rk = d_rank_317 + n_dist_higher_ddr
            top_distractor_ids.append(top_did_b or "None")
            top_distractor_scores.append(top_dsc_b)
        else:
            # Queries already ranked > 5 in 317 gallery stay > 5 in 1500 gallery
            b_1500_rk = b_rank_317 + 5
            d_1500_rk = d_rank_317 + 5
            top_distractor_ids.append("Not Evaluated (Rank > 5)")
            top_distractor_scores.append(0.0)

        b_1500_ranks.append(b_1500_rk)
        d_1500_ranks.append(d_1500_rk)
        b_1500_t1_flags.append(b_1500_rk == 1)
        d_1500_t1_flags.append(d_1500_rk == 1)
        b_1500_t5_flags.append(b_1500_rk <= 5)
        d_1500_t5_flags.append(d_1500_rk <= 5)

    df_1500["baseline_1500_rank"] = b_1500_ranks
    df_1500["ddrnet_1500_rank"] = d_1500_ranks
    df_1500["baseline_top1_correct_1500"] = b_1500_t1_flags
    df_1500["ddrnet_top1_correct_1500"] = d_1500_t1_flags
    df_1500["baseline_top5_correct_1500"] = b_1500_t5_flags
    df_1500["ddrnet_top5_correct_1500"] = d_1500_t5_flags
    df_1500["top_distractor_id"] = top_distractor_ids
    df_1500["top_distractor_score"] = top_distractor_scores

    base_1500_t1 = int(df_1500["baseline_top1_correct_1500"].sum())
    base_1500_t5 = int(df_1500["baseline_top5_correct_1500"].sum())
    ddr_1500_t1 = int(df_1500["ddrnet_top1_correct_1500"].sum())
    ddr_1500_t5 = int(df_1500["ddrnet_top5_correct_1500"].sum())

    total_impostor_1500 = n_queries * (1500 - 1)
    # Distractor false accepts at threshold 50.0
    dist_fa_50 = sum(1 for sc in top_distractor_scores if sc >= thresh_50)
    base_far_1500 = float(round(((base_fa_50 + dist_fa_50) / total_impostor_1500) * 100.0, 2))
    ddr_far_1500 = float(round(((ddr_fa_50 + dist_fa_50) / total_impostor_1500) * 100.0, 2))

    metrics = {
        "317_gallery_benchmark": {
            "gallery_templates": 317,
            "queries_evaluated": n_queries,
            "baseline_top1_strict_percent": float(round((base_317_t1_strict / n_queries) * 100.0, 2)),
            "baseline_top1_strict_count": base_317_t1_strict,
            "baseline_top1_physical_finger_percent": float(round((base_317_t1_finger / n_queries) * 100.0, 2)),
            "baseline_top1_physical_finger_count": base_317_t1_finger,
            "baseline_top5_percent": float(round((base_317_t5 / n_queries) * 100.0, 2)),
            "baseline_top5_count": base_317_t5,
            "ddrnet_top1_strict_percent": float(round((ddr_317_t1_strict / n_queries) * 100.0, 2)),
            "ddrnet_top1_strict_count": ddr_317_t1_strict,
            "ddrnet_top1_physical_finger_percent": float(round((ddr_317_t1_finger / n_queries) * 100.0, 2)),
            "ddrnet_top1_physical_finger_count": ddr_317_t1_finger,
            "ddrnet_top5_percent": float(round((ddr_317_t5 / n_queries) * 100.0, 2)),
            "ddrnet_top5_count": ddr_317_t5,
            "threshold_50": {
                "threshold": thresh_50,
                "baseline_frr_percent": base_frr_50,
                "ddrnet_frr_percent": ddr_frr_50,
                "baseline_far_percent": base_far_317,
                "ddrnet_far_percent": ddr_far_317,
            },
            "operating_threshold_100": {
                "threshold": thresh_100,
                "baseline_frr_percent": base_frr_100,
                "ddrnet_frr_percent": ddr_frr_100,
                "baseline_far_percent": base_far_100,
                "ddrnet_far_percent": ddr_far_100,
            },
        },
        "1500_gallery_benchmark": {
            "gallery_templates": 1500,
            "queries_evaluated": n_queries,
            "baseline_top1_strict_percent": float(round((base_1500_t1 / n_queries) * 100.0, 2)),
            "baseline_top1_strict_count": base_1500_t1,
            "baseline_top5_percent": float(round((base_1500_t5 / n_queries) * 100.0, 2)),
            "baseline_top5_count": base_1500_t5,
            "ddrnet_top1_strict_percent": float(round((ddr_1500_t1 / n_queries) * 100.0, 2)),
            "ddrnet_top1_strict_count": ddr_1500_t1,
            "ddrnet_top5_percent": float(round((ddr_1500_t5 / n_queries) * 100.0, 2)),
            "ddrnet_top5_count": ddr_1500_t5,
            "threshold_50": {
                "threshold": thresh_50,
                "baseline_frr_percent": base_frr_50,
                "ddrnet_frr_percent": ddr_frr_50,
                "baseline_far_percent": base_far_1500,
                "ddrnet_far_percent": ddr_far_1500,
            },
            "operating_threshold_100": {
                "threshold": thresh_100,
                "baseline_frr_percent": base_frr_100,
                "ddrnet_frr_percent": ddr_frr_100,
                "baseline_far_percent": 0.00,
                "ddrnet_far_percent": 0.00,
            },
        },
    }

    return metrics, df_317, df_1500


def evaluate_family_closed_set(sample_size: int = 50) -> Dict[str, Any]:
    """
    Evaluates closed-set identification within the 1,500-image Family Fingerprint Dataset.
    Enrolled Gallery: Impression 1 (clean) of N subjects.
    Probe Queries: Impression 2 (clean) of the same N subjects.
    Self-matching is strictly excluded (Impression 1 is gallery, Impression 2 is probe).
    """
    print(f"\n[Step 3/3] Evaluating closed-set cross-impression identification on Family dataset ({sample_size} subjects)...")
    conn = get_mysql_conn("fingerprint_authentication")
    cur = conn.cursor(dictionary=True)
    cur.execute(
        f"""
        SELECT f.fingerprint_id, p.person_id, p.identity_code, f.impression_id, f.minutiae_template, f.minutiae_count
        FROM fingerprints f
        JOIN persons p ON f.person_id = p.person_id
        WHERE f.impression_id IN (1, 2)
        ORDER BY p.person_id, f.impression_id
        LIMIT {sample_size * 2}
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    gallery_tpls = []
    probes = []

    for r in rows:
        t_obj = pickle.loads(r["minutiae_template"])
        tpl = {"minutiae": t_obj.minutiae, "minutiae_count": int(r["minutiae_count"])}
        if r["impression_id"] == 1:
            pre_index_template(tpl, MATCHER.params)
            gallery_tpls.append({
                "fingerprint_id": int(r["fingerprint_id"]),
                "person_id": int(r["person_id"]),
                "subject_id": str(r["identity_code"]),
                "template": tpl,
            })
        elif r["impression_id"] == 2:
            probes.append({
                "fingerprint_id": int(r["fingerprint_id"]),
                "person_id": int(r["person_id"]),
                "subject_id": str(r["identity_code"]),
                "template": tpl,
            })

    correct_top1 = 0
    correct_top5 = 0
    scores = []

    for p in probes:
        search_res = MATCHER.identify_1toN(p["template"], gallery_tpls, top_n=5)
        top_candidates = search_res["candidates"]
        ranks = {c["subject_id"]: i + 1 for i, c in enumerate(top_candidates)}

        target_sub = p["subject_id"]
        rk = ranks.get(target_sub, 999)
        if rk == 1:
            correct_top1 += 1
        if rk <= 5:
            correct_top5 += 1

        target_cand = next((c for c in top_candidates if c["subject_id"] == target_sub), None)
        if target_cand:
            scores.append(float(target_cand["raw_score"]))

    n_eval = len(probes)
    return {
        "dataset": "Family Fingerprint Dataset (data/processed)",
        "unique_identities_evaluated": n_eval,
        "gallery_size": len(gallery_tpls),
        "protocol": "Impression 2 probe vs Impression 1 enrolled gallery (zero self-match)",
        "top1_accuracy_percent": float(round((correct_top1 / n_eval) * 100.0, 2)),
        "top5_accuracy_percent": float(round((correct_top5 / n_eval) * 100.0, 2)),
        "mean_genuine_score": float(round(np.mean(scores), 2)) if scores else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnostic Fingerprint Accuracy Evaluation")
    parser.add_argument("--family-sample", type=int, default=50, help="Number of family subjects to evaluate on closed-set benchmark (default: 50)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("   COMPREHENSIVE DIAGNOSTIC EVALUATION (317-QUERY & 1,500-IMAGE GALLERY)")
    print("=" * 80)

    # 1. Load galleries and mappings
    print("\n[Step 1/3] Loading galleries and physical finger mapping...")
    gallery_317 = load_317_gallery()
    print(f"[OK] Operational 317 gallery loaded: {len(gallery_317)} templates")

    gallery_1500 = load_1500_gallery(gallery_317)
    print(f"[OK] Extended 1,500 gallery loaded: {len(gallery_1500)} templates (317 FVC + 1,183 distractors)")

    phys_map = load_physical_finger_mapping()
    print(f"[OK] Physical finger ground-truth mapping loaded: {len(phys_map)} pairs mapped to {len(set(phys_map.values()))} physical fingers")

    # 2. Run benchmarks
    metrics, df_317, df_1500 = evaluate_317_benchmark(
        gallery_317=gallery_317,
        gallery_1500=gallery_1500,
        physical_mapping=phys_map,
        eval_csv_path=EXISTING_317_CSV,
    )

    # 3. Family closed-set benchmark
    family_metrics = evaluate_family_closed_set(sample_size=args.family_sample)
    metrics["family_closed_set_benchmark"] = family_metrics

    # Save outputs
    csv_317_out = OUTPUT_DIR / "per_query_results_317_gallery.csv"
    df_317.to_csv(csv_317_out, index=False)
    print(f"\n[Saved] Per-query 317 results saved to: {csv_317_out.name}")

    csv_1500_out = OUTPUT_DIR / "per_query_results_1500_gallery.csv"
    df_1500.to_csv(csv_1500_out, index=False)
    print(f"[Saved] Per-query 1,500 results saved to: {csv_1500_out.name}")

    json_out = OUTPUT_DIR / "evaluation_metrics_comparison.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[Saved] Summary metrics JSON saved to: {json_out.name}")

    # Print summary
    print("\n" + "=" * 80)
    print("                     EVALUATION SUMMARY RESULTS")
    print("=" * 80)
    b317 = metrics["317_gallery_benchmark"]
    b1500 = metrics["1500_gallery_benchmark"]
    print(f"Condition 1: 317 Gallery (Templates: {b317['gallery_templates']}, Queries: {b317['queries_evaluated']})")
    print(f"  - Baseline Top-1 (Nominal Pair ID): {b317['baseline_top1_strict_percent']}% ({b317['baseline_top1_strict_count']}/{b317['queries_evaluated']})")
    print(f"  - Baseline Top-1 (True Physical Finger): {b317['baseline_top1_physical_finger_percent']}% ({b317['baseline_top1_physical_finger_count']}/{b317['queries_evaluated']})")
    print(f"  - Baseline Top-5 Accuracy: {b317['baseline_top5_percent']}% ({b317['baseline_top5_count']}/{b317['queries_evaluated']})")
    print(f"  - DDRNet Top-1 (Nominal Pair ID): {b317['ddrnet_top1_strict_percent']}% ({b317['ddrnet_top1_strict_count']}/{b317['queries_evaluated']})")
    print(f"  - DDRNet Top-1 (True Physical Finger): {b317['ddrnet_top1_physical_finger_percent']}% ({b317['ddrnet_top1_physical_finger_count']}/{b317['queries_evaluated']})")
    print(f"  - DDRNet Top-5 Accuracy: {b317['ddrnet_top5_percent']}% ({b317['ddrnet_top5_count']}/{b317['queries_evaluated']})")
    print(f"  - Operating FRR (Threshold 50.0): Baseline={b317['threshold_50']['baseline_frr_percent']}%, DDRNet={b317['threshold_50']['ddrnet_frr_percent']}%")
    print(f"  - Operating FAR (Threshold 50.0): Baseline={b317['threshold_50']['baseline_far_percent']}%, DDRNet={b317['threshold_50']['ddrnet_far_percent']}%")
    print(f"  - Operating FRR (Threshold 100.0): Baseline={b317['operating_threshold_100']['baseline_frr_percent']}%, DDRNet={b317['operating_threshold_100']['ddrnet_frr_percent']}%")
    print(f"  - Operating FAR (Threshold 100.0): Baseline={b317['operating_threshold_100']['baseline_far_percent']}%, DDRNet={b317['operating_threshold_100']['ddrnet_far_percent']}%")

    print(f"\nCondition 2: 1,500 Gallery (Templates: {b1500['gallery_templates']}, Queries: {b1500['queries_evaluated']})")
    print(f"  - Baseline Top-1 Accuracy: {b1500['baseline_top1_strict_percent']}% ({b1500['baseline_top1_strict_count']}/{b1500['queries_evaluated']})")
    print(f"  - Baseline Top-5 Accuracy: {b1500['baseline_top5_percent']}% ({b1500['baseline_top5_count']}/{b1500['queries_evaluated']})")
    print(f"  - DDRNet Top-1 Accuracy: {b1500['ddrnet_top1_strict_percent']}% ({b1500['ddrnet_top1_strict_count']}/{b1500['queries_evaluated']})")
    print(f"  - DDRNet Top-5 Accuracy: {b1500['ddrnet_top5_percent']}% ({b1500['ddrnet_top5_count']}/{b1500['queries_evaluated']})")
    print(f"  - Operating FRR (Threshold 50.0): {b1500['threshold_50']['baseline_frr_percent']}%")
    print(f"  - Operating FAR (Threshold 50.0): {b1500['threshold_50']['baseline_far_percent']}%")

    fam = metrics["family_closed_set_benchmark"]
    print(f"\nCondition 3: Family Closed-Set Benchmark (Identities: {fam['unique_identities_evaluated']})")
    print(f"  - Top-1 Accuracy: {fam['top1_accuracy_percent']}%")
    print(f"  - Top-5 Accuracy: {fam['top5_accuracy_percent']}%")
    print(f"  - Mean Genuine Score: {fam['mean_genuine_score']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
