"""
COMPLETE 317-IMAGE LATENT -> CLEAN FVC 1:N RECOGNITION EVALUATION
Evaluates Baseline vs Proposed DDRNet Pipeline against newDB.external_fvc_clean_gallery.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import json
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import mysql.connector
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

# Patch authentication checkpoint paths to the original fingerprint project
FINGERPRINT_PROJECT = Path(r"C:\Users\kusum\OneDrive\Desktop\fingerprint")
import authentication as _auth_module
_auth_module.CLASSIFIER_CHECKPOINT = FINGERPRINT_PROJECT / "experiments" / "classifier_hard_negative" / "best_model.pth"
_auth_module.DDRNET_CHECKPOINT     = FINGERPRINT_PROJECT / "experiments" / "ddrnet_baseline" / "best_model.pth"
_auth_module.DDRNET = None
_auth_module.CLASSIFIER = None

from matcher import SourceAFISMatcher
import afis.algorithms.safis as safis
from authentication import rectify_with_ddrnet, classify_distortion, load_ddrnet, load_classifier

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
TARGET_DB = "newdb"
TARGET_TABLE = "external_fvc_clean_gallery"

LATENT_DIR = PROJECT_ROOT / "data" / "Latent_fvc"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean_fvc_new"
CSV_PATH = PROJECT_ROOT / "latent_fvc_317_baseline_vs_ddrnet.csv"
JSON_PATH = PROJECT_ROOT / "scratch" / "latent_evaluation_317_summary.json"

TOTAL_QUERIES = 317

def get_db_connection():
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=TARGET_DB,
    )

def load_gallery(matcher: SourceAFISMatcher) -> Tuple[List[Dict[str, Any]], List[int]]:
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"""
        SELECT fingerprint_id, template_data, minutiae_count, generation_status
        FROM `{TARGET_TABLE}`
        ORDER BY fingerprint_id ASC;
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    gallery = []
    excluded_ids = []
    length_tol = float(matcher.params.length_tol) or 1e-6

    for r in rows:
        fid = r["fingerprint_id"]
        tdata = r["template_data"]
        status = r["generation_status"]

        if status != "generated" or tdata is None:
            excluded_ids.append(fid)
            continue

        try:
            tpl = pickle.loads(tdata)
            c_minu = tpl.get("minutiae")
            if c_minu is not None and len(c_minu) >= 2:
                cand_edges = safis.build_edges(c_minu, matcher.params)
                cand_buckets = {}
                for e in cand_edges:
                    k = int(round(e.length / length_tol))
                    cand_buckets.setdefault(k, []).append(e)
                tpl["_edges"] = cand_edges
                tpl["_buckets"] = cand_buckets
            else:
                tpl["_edges"] = []
                tpl["_buckets"] = {}

            gallery.append({"fingerprint_id": fid, "template": tpl})
        except Exception:
            excluded_ids.append(fid)

    return gallery, excluded_ids

def run_1toN(probe_template: Dict, gallery: List[Dict], matcher: SourceAFISMatcher) -> Tuple[List[Dict], float]:
    probe_minu = probe_template.get("minutiae")
    if probe_minu is None or len(probe_minu) < 2:
        return [], 0.0

    ea = safis.build_edges(probe_minu, matcher.params)
    length_tol = float(matcher.params.length_tol) or 1e-6
    angle_tol = matcher.params.angle_tol
    max_roots = matcher.params.max_roots

    results = []
    t0 = time.time()

    for cand in gallery:
        cand_tpl = cand["template"]
        cand_edges = cand_tpl.get("_edges", [])
        cand_buckets = cand_tpl.get("_buckets", {})

        if not cand_edges or not ea:
            raw_score = 0.0
            pairs_count = 0
        else:
            roots = []
            for pe in ea:
                key = int(round(pe.length / length_tol))
                for k in (key - 1, key, key + 1):
                    b = cand_buckets.get(k)
                    if not b:
                        continue
                    for ce in b:
                        if safis._edges_compatible(pe, ce, matcher.params):
                            close = 1.0 - (
                                abs(pe.length - ce.length) / length_tol
                                + safis._angle_diff(pe.ref_angle, ce.ref_angle) / angle_tol
                                + safis._angle_diff(pe.neighbor_angle, ce.neighbor_angle) / angle_tol
                            ) / 3.0
                            roots.append((close, pe, ce))

            if not roots:
                raw_score = 0.0
                pairs_count = 0
            else:
                roots.sort(key=lambda t: t[0], reverse=True)
                roots = roots[:max_roots]
                best_n = 0
                best_pairs_cnt = 0
                for _c, pe, ce in roots:
                    pairing, pairs = safis._crawl(ea, cand_edges, pe.ref, ce.ref, pe.neighbor, ce.neighbor, matcher.params)
                    if len(pairing) > best_n:
                        best_n = len(pairing)
                        best_pairs_cnt = len(pairs)
                raw_score = float(best_n) * 5.0
                if best_n >= 2:
                    raw_score += float(best_pairs_cnt) * 1.5
                pairs_count = best_pairs_cnt

        results.append({
            "fingerprint_id": cand["fingerprint_id"],
            "raw_score": round(raw_score, 2),
            "matched_minutiae": pairs_count,
        })

    match_ms = (time.time() - t0) * 1000.0
    results.sort(key=lambda x: (x["raw_score"], x["matched_minutiae"]), reverse=True)
    return results, round(match_ms, 2)

def get_rank_and_score(results: List[Dict], expected_id: int) -> Tuple[Optional[int], float]:
    for idx, r in enumerate(results, start=1):
        if r["fingerprint_id"] == expected_id:
            return idx, r["raw_score"]
    return None, 0.0

def main():
    print("=" * 80)
    print("INITIALIZING 317-IMAGE LATENT -> CLEAN FVC 1:N EVALUATION")
    print("=" * 80)

    matcher = SourceAFISMatcher()
    print("Pre-loading DDRNet and Classifier...")
    load_classifier()
    load_ddrnet()
    print("Pre-loading complete.")

    print("\nLoading and pre-indexing gallery from newDB.external_fvc_clean_gallery...")
    t_gal = time.time()
    gallery, excluded_ids = load_gallery(matcher)
    gal_ms = (time.time() - t_gal) * 1000.0
    print(f"Loaded {len(gallery)} gallery templates in {gal_ms:.2f} ms.")
    print(f"Excluded gallery IDs: {excluded_ids}")

    records = []
    failures = []

    print("\nStarting evaluation of 317 queries...")
    t_all_start = time.time()

    for qid in range(1, TOTAL_QUERIES + 1):
        filename = f"{qid}.png"
        img_path = LATENT_DIR / filename
        expected_id = qid

        rec = {
            "query_id": qid,
            "query_filename": filename,
            "expected_id": expected_id,
            "baseline_success": False,
            "ddrnet_success": False,
        }

        if not img_path.exists():
            failures.append({
                "query_id": qid,
                "filename": filename,
                "stage": "load",
                "error": "File not found"
            })
            records.append(rec)
            continue

        try:
            img_raw = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img_raw is None or img_raw.size == 0:
                raise ValueError("cv2.imread returned empty image")

            # ----------------------------------------------------
            # EXPERIMENT A: BASELINE
            # ----------------------------------------------------
            t_b0 = time.time()
            t_ext_b0 = time.time()
            tpl_b = matcher.extract_template(img_raw)
            ext_b_ms = (time.time() - t_ext_b0) * 1000.0

            results_b, match_b_ms = run_1toN(tpl_b, gallery, matcher)
            e2e_b_ms = (time.time() - t_b0) * 1000.0

            rank_b, score_b = get_rank_and_score(results_b, expected_id)
            top1_b = results_b[0] if results_b else {}
            top2_b = results_b[1] if len(results_b) > 1 else {}
            top3_b = results_b[2] if len(results_b) > 2 else {}
            top5_b_str = ";".join([f"{r['fingerprint_id']}:{r['raw_score']}" for r in results_b[:5]])
            margin_b = round(top1_b.get("raw_score", 0.0) - top2_b.get("raw_score", 0.0), 2)

            rec.update({
                "baseline_success": True,
                "baseline_minutiae_count": tpl_b.get("minutiae_count", 0),
                "baseline_top1_id": top1_b.get("fingerprint_id"),
                "baseline_top1_score": top1_b.get("raw_score", 0.0),
                "baseline_top2_score": top2_b.get("raw_score", 0.0),
                "baseline_top3_score": top3_b.get("raw_score", 0.0),
                "baseline_top1_top2_margin": margin_b,
                "baseline_expected_score": score_b,
                "baseline_expected_rank": rank_b,
                "baseline_top5": top5_b_str,
                "baseline_ext_ms": round(ext_b_ms, 2),
                "baseline_runtime_ms": round(match_b_ms, 2),
                "baseline_end_to_end_ms": round(e2e_b_ms, 2),
            })

            # ----------------------------------------------------
            # EXPERIMENT B: PROPOSED DDRNET PIPELINE
            # ----------------------------------------------------
            t_d0 = time.time()
            t_cls0 = time.time()
            cls_res = classify_distortion(img_raw)
            cls_ms = (time.time() - t_cls0) * 1000.0

            is_distorted = cls_res["is_distorted"]
            prob_distorted = cls_res["prob_distorted"]
            pred_class = cls_res["predicted_class"]

            ddrnet_applied = False
            ddrnet_inf_ms = 0.0

            if is_distorted:
                t_rect0 = time.time()
                img_proc = rectify_with_ddrnet(img_raw)
                ddrnet_inf_ms = (time.time() - t_rect0) * 1000.0
                ddrnet_applied = True
                
                t_ext_d0 = time.time()
                tpl_d = matcher.extract_template(img_proc)
                ext_d_ms = (time.time() - t_ext_d0) * 1000.0

                results_d, match_d_ms = run_1toN(tpl_d, gallery, matcher)
            else:
                img_proc = img_raw
                tpl_d = tpl_b
                ext_d_ms = 0.0
                results_d = results_b
                match_d_ms = match_b_ms

            e2e_d_ms = (time.time() - t_d0) * 1000.0

            rank_d, score_d = get_rank_and_score(results_d, expected_id)
            top1_d = results_d[0] if results_d else {}
            top2_d = results_d[1] if len(results_d) > 1 else {}
            top3_d = results_d[2] if len(results_d) > 2 else {}
            top5_d_str = ";".join([f"{r['fingerprint_id']}:{r['raw_score']}" for r in results_d[:5]])
            margin_d = round(top1_d.get("raw_score", 0.0) - top2_d.get("raw_score", 0.0), 2)

            # Classify outcome:
            # IMPROVED: rank becomes better (rank_d < rank_b), OR (rank_d == rank_b and score_d > score_b)
            # DEGRADED: rank becomes worse (rank_d > rank_b), OR (rank_d == rank_b and score_d < score_b)
            # UNCHANGED: rank_d == rank_b and score_d == score_b
            if rank_d is not None and rank_b is not None:
                if rank_d < rank_b or (rank_d == rank_b and score_d > score_b):
                    outcome = "IMPROVED"
                elif rank_d > rank_b or (rank_d == rank_b and score_d < score_b):
                    outcome = "DEGRADED"
                else:
                    outcome = "UNCHANGED"
            else:
                outcome = "UNCHANGED"

            rec.update({
                "ddrnet_success": True,
                "classifier_result": pred_class,
                "classifier_score": prob_distorted,
                "ddrnet_applied": ddrnet_applied,
                "ddrnet_minutiae_count": tpl_d.get("minutiae_count", 0),
                "ddrnet_top1_id": top1_d.get("fingerprint_id"),
                "ddrnet_top1_score": top1_d.get("raw_score", 0.0),
                "ddrnet_top2_score": top2_d.get("raw_score", 0.0),
                "ddrnet_top3_score": top3_d.get("raw_score", 0.0),
                "ddrnet_top1_top2_margin": margin_d,
                "ddrnet_expected_score": score_d,
                "ddrnet_expected_rank": rank_d,
                "ddrnet_top5": top5_d_str,
                "ddrnet_inference_ms": round(ddrnet_inf_ms, 2),
                "ddrnet_ext_ms": round(ext_d_ms, 2),
                "ddrnet_runtime_ms": round(match_d_ms, 2),
                "ddrnet_end_to_end_ms": round(e2e_d_ms, 2),
                "outcome": outcome,
            })

            records.append(rec)

        except Exception as e:
            failures.append({
                "query_id": qid,
                "filename": filename,
                "stage": "pipeline",
                "error": str(e)
            })
            rec["outcome"] = "FAILED"
            records.append(rec)

        if qid % 50 == 0 or qid == TOTAL_QUERIES:
            elapsed = time.time() - t_all_start
            print(f"Evaluated {qid}/{TOTAL_QUERIES} queries in {elapsed:.1f}s...")

    # Write CSV
    print(f"\nWriting complete results to CSV: {CSV_PATH}")
    csv_columns = [
        "query_id", "query_filename", "expected_id",
        "baseline_top1_id", "baseline_top1_score", "baseline_expected_score",
        "baseline_expected_rank", "baseline_top5",
        "classifier_result", "classifier_score", "ddrnet_applied",
        "ddrnet_top1_id", "ddrnet_top1_score", "ddrnet_expected_score",
        "ddrnet_expected_rank", "ddrnet_top5",
        "baseline_runtime_ms", "ddrnet_runtime_ms",
        "baseline_end_to_end_ms", "ddrnet_end_to_end_ms",
        "outcome"
    ]

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    # ====================================================
    # COMPUTE METRICS
    # ====================================================
    valid_recs = [r for r in records if r.get("baseline_success") and r.get("ddrnet_success")]
    n_valid = len(valid_recs)

    # Baseline metrics
    b_ranks = [r["baseline_expected_rank"] for r in valid_recs if r["baseline_expected_rank"] is not None]
    b_scores = [r["baseline_expected_score"] for r in valid_recs]
    b_r1 = sum(1 for r in b_ranks if r == 1)
    b_r5 = sum(1 for r in b_ranks if r <= 5)
    b_r10 = sum(1 for r in b_ranks if r <= 10)

    # DDRNet metrics
    d_ranks = [r["ddrnet_expected_rank"] for r in valid_recs if r["ddrnet_expected_rank"] is not None]
    d_scores = [r["ddrnet_expected_score"] for r in valid_recs]
    d_r1 = sum(1 for r in d_ranks if r == 1)
    d_r5 = sum(1 for r in d_ranks if r <= 5)
    d_r10 = sum(1 for r in d_ranks if r <= 10)

    # Outcomes
    improved_count = sum(1 for r in valid_recs if r["outcome"] == "IMPROVED")
    degraded_count = sum(1 for r in valid_recs if r["outcome"] == "DEGRADED")
    unchanged_count = sum(1 for r in valid_recs if r["outcome"] == "UNCHANGED")

    score_changes = [r["ddrnet_expected_score"] - r["baseline_expected_score"] for r in valid_recs]
    rank_changes = [r["baseline_expected_rank"] - r["ddrnet_expected_rank"] for r in valid_recs]  # positive = improved rank

    # Classifier subsets
    ddrnet_applied_recs = [r for r in valid_recs if r["ddrnet_applied"]]
    ddrnet_bypassed_recs = [r for r in valid_recs if not r["ddrnet_applied"]]

    def calc_group_stats(group):
        if not group:
            return {}
        g_br1 = sum(1 for r in group if r["baseline_expected_rank"] == 1)
        g_dr1 = sum(1 for r in group if r["ddrnet_expected_rank"] == 1)
        g_imp = sum(1 for r in group if r["outcome"] == "IMPROVED")
        g_deg = sum(1 for r in group if r["outcome"] == "DEGRADED")
        g_unc = sum(1 for r in group if r["outcome"] == "UNCHANGED")
        g_bscore = float(np.mean([r["baseline_expected_score"] for r in group]))
        g_dscore = float(np.mean([r["ddrnet_expected_score"] for r in group]))
        return {
            "count": len(group),
            "baseline_r1": g_br1,
            "baseline_r1_acc": round(g_br1 / len(group) * 100, 2),
            "ddrnet_r1": g_dr1,
            "ddrnet_r1_acc": round(g_dr1 / len(group) * 100, 2),
            "improved": g_imp,
            "degraded": g_deg,
            "unchanged": g_unc,
            "mean_baseline_score": round(g_bscore, 2),
            "mean_ddrnet_score": round(g_dscore, 2),
        }

    stats_applied = calc_group_stats(ddrnet_applied_recs)
    stats_bypassed = calc_group_stats(ddrnet_bypassed_recs)

    # Runtime analysis
    b_e2e = [r["baseline_end_to_end_ms"] for r in valid_recs]
    d_e2e = [r["ddrnet_end_to_end_ms"] for r in valid_recs]
    d_inf = [r["ddrnet_inference_ms"] for r in ddrnet_applied_recs]

    # Top cases
    # 1. 10 queries where baseline performed best (rank 1, highest score)
    best_baseline = sorted(valid_recs, key=lambda x: (x["baseline_expected_rank"], -x["baseline_expected_score"]))[:10]

    # 2. 10 queries where DDRNet improved the rank the most
    # rank_improvement = baseline_rank - ddrnet_rank
    improved_rank_cases = sorted(
        [r for r in valid_recs if r["baseline_expected_rank"] is not None and r["ddrnet_expected_rank"] is not None],
        key=lambda x: (x["baseline_expected_rank"] - x["ddrnet_expected_rank"], x["ddrnet_expected_score"] - x["baseline_expected_score"]),
        reverse=True
    )[:10]

    # 3. 10 queries where DDRNet degraded the rank the most
    degraded_rank_cases = sorted(
        [r for r in valid_recs if r["baseline_expected_rank"] is not None and r["ddrnet_expected_rank"] is not None],
        key=lambda x: (x["ddrnet_expected_rank"] - x["baseline_expected_rank"], x["baseline_expected_score"] - x["ddrnet_expected_score"]),
        reverse=True
    )[:10]

    # 4. 10 queries with lowest expected-ID scores (in baseline)
    lowest_scores = sorted(valid_recs, key=lambda x: x["baseline_expected_score"])[:10]

    summary = {
        "dataset_total": TOTAL_QUERIES,
        "valid_queries": n_valid,
        "failed_queries": len(failures),
        "excluded_gallery_ids": excluded_ids,
        "baseline": {
            "rank1_count": b_r1,
            "rank1_acc": round(b_r1 / n_valid * 100, 2),
            "rank5_count": b_r5,
            "rank5_acc": round(b_r5 / n_valid * 100, 2),
            "rank10_count": b_r10,
            "rank10_acc": round(b_r10 / n_valid * 100, 2),
            "mean_score": round(float(np.mean(b_scores)), 2),
            "median_score": round(float(np.median(b_scores)), 2),
            "mean_rank": round(float(np.mean(b_ranks)), 2),
            "median_rank": round(float(np.median(b_ranks)), 2),
        },
        "ddrnet": {
            "rank1_count": d_r1,
            "rank1_acc": round(d_r1 / n_valid * 100, 2),
            "rank5_count": d_r5,
            "rank5_acc": round(d_r5 / n_valid * 100, 2),
            "rank10_count": d_r10,
            "rank10_acc": round(d_r10 / n_valid * 100, 2),
            "mean_score": round(float(np.mean(d_scores)), 2),
            "median_score": round(float(np.median(d_scores)), 2),
            "mean_rank": round(float(np.mean(d_ranks)), 2),
            "median_rank": round(float(np.median(d_ranks)), 2),
        },
        "comparison": {
            "improved": improved_count,
            "degraded": degraded_count,
            "unchanged": unchanged_count,
            "rank1_change": d_r1 - b_r1,
            "rank5_change": d_r5 - b_r5,
            "rank10_change": d_r10 - b_r10,
            "mean_score_change": round(float(np.mean(score_changes)), 2),
            "mean_rank_change": round(float(np.mean(rank_changes)), 2),
        },
        "classifier_groups": {
            "ddrnet_applied": stats_applied,
            "ddrnet_bypassed": stats_bypassed,
        },
        "runtime": {
            "total_baseline_ms": round(float(np.sum(b_e2e)), 2),
            "total_ddrnet_ms": round(float(np.sum(d_e2e)), 2),
            "avg_baseline_ms": round(float(np.mean(b_e2e)), 2),
            "avg_ddrnet_ms": round(float(np.mean(d_e2e)), 2),
            "avg_ddrnet_inf_ms": round(float(np.mean(d_inf)), 2) if d_inf else 0.0,
            "min_baseline_ms": round(float(np.min(b_e2e)), 2),
            "max_baseline_ms": round(float(np.max(b_e2e)), 2),
            "min_ddrnet_ms": round(float(np.min(d_e2e)), 2),
            "max_ddrnet_ms": round(float(np.max(d_e2e)), 2),
        },
        "best_baseline": best_baseline,
        "improved_rank_cases": improved_rank_cases,
        "degraded_rank_cases": degraded_rank_cases,
        "lowest_scores": lowest_scores,
        "failures": failures,
    }

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("SUMMARY SAVED TO JSON:", JSON_PATH)
    print("=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    main()
