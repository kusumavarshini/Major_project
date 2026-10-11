"""
SOURCEAFIS 1:N GALLERY VALIDATION TEST SCRIPT
Tests clean gallery self-identification for IDs 1, 2, 100, 200, 316.
Reuses in-memory deserialized candidate templates from external_clean_gallery.
Candidate images are NOT loaded. Matcher is created once per query.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import mysql.connector
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from matcher import SourceAFISMatcher
import afis.algorithms.safis as safis

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
TARGET_DB = "external_fingerprint_db"
TARGET_TABLE = "external_clean_gallery"
DATA_DIR = PROJECT_ROOT / "data" / "clean_fvc"


def load_gallery_from_db() -> List[Dict[str, Any]]:
    """
    Load all 315 generated templates from MySQL external_clean_gallery into memory ONCE.
    Excludes failed templates (such as ID 262).
    """
    t0 = time.time()
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=TARGET_DB,
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"""
        SELECT fingerprint_id, template_data, minutiae_count, template_version, generation_status
        FROM `{TARGET_TABLE}`
        WHERE generation_status = 'generated' AND template_data IS NOT NULL
        ORDER BY fingerprint_id;
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    gallery = []
    for r in rows:
        fid = r["fingerprint_id"]
        tpl = pickle.loads(r["template_data"])
        gallery.append({
            "fingerprint_id": fid,
            "template": tpl,
            "minutiae_count": r["minutiae_count"],
        })

    elapsed_ms = (time.time() - t0) * 1000.0
    print(f"Loaded {len(gallery)} candidate templates from MySQL in {elapsed_ms:.2f} ms")
    return gallery


def run_single_1toN_test(
    expected_id: int,
    gallery: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Execute 1:N search for a single clean probe image.
    Measures extraction time, candidate match times, and rankings.
    """
    img_filename = f"{expected_id}.png"
    img_path = DATA_DIR / img_filename
    assert img_path.exists(), f"Query image does not exist: {img_path}"

    t_e2e_start = time.time()

    # 1. Load image and generate query template ONCE
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Failed to read image: {img_path}"

    matcher = SourceAFISMatcher()

    t_ext_start = time.time()
    probe_template = matcher.extract_template(img)
    t_ext_ms = (time.time() - t_ext_start) * 1000.0

    probe_minu = probe_template.get("minutiae")
    assert probe_minu is not None and len(probe_minu) >= 2, "Probe template has insufficient minutiae"

    # 2. Build probe invariant edges ONCE
    ea = safis.build_edges(probe_minu, matcher.params)
    length_tol = float(matcher.params.length_tol) or 1e-6
    angle_tol = matcher.params.angle_tol
    max_roots = matcher.params.max_roots

    candidate_results = []
    t_match_start = time.time()

    # 3. Match against each candidate in gallery
    for cand_record in gallery:
        t_cand_start = time.time()

        cand_tpl = cand_record["template"]
        cand_edges = cand_tpl.get("_edges")
        cand_buckets = cand_tpl.get("_buckets")

        if cand_edges is None or cand_buckets is None:
            c_minu = cand_tpl.get("minutiae")
            if c_minu is not None and len(c_minu) >= 2:
                cand_edges = safis.build_edges(c_minu, matcher.params)
                cand_buckets = {}
                for e in cand_edges:
                    k = int(round(e.length / length_tol))
                    cand_buckets.setdefault(k, []).append(e)
                cand_tpl["_edges"] = cand_edges
                cand_tpl["_buckets"] = cand_buckets
            else:
                cand_edges = []
                cand_buckets = {}
                cand_tpl["_edges"] = cand_edges
                cand_tpl["_buckets"] = cand_buckets

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

        t_cand_ms = (time.time() - t_cand_start) * 1000.0

        candidate_results.append({
            "fingerprint_id": cand_record["fingerprint_id"],
            "raw_score": round(raw_score, 2),
            "matched_minutiae": pairs_count,
            "matching_time_ms": round(t_cand_ms, 2),
        })

    t_match_ms = (time.time() - t_match_start) * 1000.0
    t_e2e_ms = (time.time() - t_e2e_start) * 1000.0

    # Sort descending by raw_score, then matched_minutiae
    candidate_results.sort(key=lambda x: (x["raw_score"], x["matched_minutiae"]), reverse=True)

    # Find rank of expected candidate
    expected_rank = None
    expected_score = None
    for idx, c in enumerate(candidate_results, start=1):
        if c["fingerprint_id"] == expected_id:
            expected_rank = idx
            expected_score = c["raw_score"]
            break

    rank1_score = candidate_results[0]["raw_score"]
    rank2_score = candidate_results[1]["raw_score"] if len(candidate_results) > 1 else 0.0
    margin = round(rank1_score - rank2_score, 2)

    return {
        "expected_id": expected_id,
        "query_file": img_filename,
        "query_minutiae_count": probe_template.get("minutiae_count", 0),
        "total_candidates": len(gallery),
        "query_extraction_time_ms": round(t_ext_ms, 2),
        "total_matching_time_ms": round(t_match_ms, 2),
        "total_e2e_time_ms": round(t_e2e_ms, 2),
        "expected_rank": expected_rank,
        "expected_score": expected_score,
        "rank1_score": rank1_score,
        "rank2_score": rank2_score,
        "margin": margin,
        "top_10": candidate_results[:10],
    }


def main():
    print("=" * 80)
    print("SOURCEAFIS 1:N GALLERY VALIDATION SUITE (CLEAN FVC)")
    print("=" * 80)

    # 1. Load gallery once
    gallery = load_gallery_from_db()
    assert len(gallery) == 315, f"Expected 315 candidates, got {len(gallery)}"
    assert 262 not in [c["fingerprint_id"] for c in gallery], "ID 262 must be excluded"

    test_ids = [1, 2, 100, 200, 316]
    summary_results = []

    for test_idx, tid in enumerate(test_ids, start=1):
        print("\n" + "#" * 80)
        print(f"TEST {test_idx} — ID {tid} SELF 1:N")
        print("#" * 80)

        res = run_single_1toN_test(tid, gallery)
        summary_results.append(res)

        print(f"Query Image                 : data/clean_fvc/{res['query_file']}")
        print(f"Expected Candidate          : fingerprint_id = {res['expected_id']}")
        print(f"Query Minutiae Count        : {res['query_minutiae_count']}")
        print(f"Total Candidates Searched   : {res['total_candidates']}")
        print(f"Query Extraction Time       : {res['query_extraction_time_ms']:.2f} ms")
        print(f"Total 1:N Matching Time     : {res['total_matching_time_ms']:.2f} ms")
        print(f"Total End-to-End Time       : {res['total_e2e_time_ms']:.2f} ms")
        print(f"Rank of Expected Candidate  : #{res['expected_rank']}")
        print(f"Raw Score of Expected ID    : {res['expected_score']}")
        print(f"Top Candidate Score (#1)    : {res['rank1_score']}")
        print(f"Runner-up Score (#2)        : {res['rank2_score']}")
        print(f"Score Margin (#1 - #2)      : {res['margin']}")
        print(f"Status                      : {'CORRECT (#1)' if res['expected_rank'] == 1 else 'INCORRECT'}")

        print("\nTop 10 Candidates:")
        print(f"{'Rank':<5} | {'fingerprint_id':<14} | {'Raw SourceAFIS Score':<20} | {'Matched Minutiae':<16} | {'Matching Time':<15}")
        print("-" * 78)
        for rank, cand in enumerate(res["top_10"], start=1):
            flag = " <--- Expected Candidate" if cand["fingerprint_id"] == tid else ""
            print(f"{rank:<5} | {cand['fingerprint_id']:<14} | {cand['raw_score']:<20} | {cand['matched_minutiae']:<16} | {cand['matching_time_ms']:<7.2f} ms{flag}")

    print("\n" + "=" * 80)
    print("FINAL VALIDATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Query':<8} | {'Expected ID':<11} | {'Rank of Expected ID':<20} | {'Expected Score':<14} | {'#1 Score':<10} | {'Margin':<10} | {'Search Time':<12}")
    print("-" * 95)
    for r in summary_results:
        print(f"{r['query_file']:<8} | {r['expected_id']:<11} | {r['expected_rank']:<20} | {r['expected_score']:<14.2f} | {r['rank1_score']:<10.2f} | {r['margin']:<10.2f} | {r['total_matching_time_ms']:<8.2f} ms")
    print("=" * 80)


if __name__ == "__main__":
    main()
