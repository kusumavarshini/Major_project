"""
LATENT-TO-CLEAN 1:N MATCHING TEST
Step 1: Ground-truth mapping discovery
Steps 2 & 3: Baseline + DDRNet matching for 5 selected latent images
Step 4: Comparison table

Ground-truth ONLY used for post-match evaluation, never supplied to the matcher.
"""

from __future__ import annotations

import os
import sys
import time
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List

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

# -----------------------------------------------------------------------
# Patch authentication checkpoint paths to the original fingerprint project
# The trial project does not have the model checkpoints; they live in the
# original fingerprint project directory.
# -----------------------------------------------------------------------
FINGERPRINT_PROJECT = Path(r"C:\Users\kusum\OneDrive\Desktop\fingerprint")
import authentication as _auth_module
_auth_module.CLASSIFIER_CHECKPOINT = FINGERPRINT_PROJECT / "experiments" / "classifier_hard_negative" / "best_model.pth"
_auth_module.DDRNET_CHECKPOINT     = FINGERPRINT_PROJECT / "experiments" / "ddrnet_baseline" / "best_model.pth"
# Reset cached singletons so load_ddrnet() / load_classifier() re-load from patched paths
_auth_module.DDRNET = None
_auth_module.CLASSIFIER = None


MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")
TARGET_DB = "external_fingerprint_db"
TARGET_TABLE = "external_clean_gallery"

# Latent images can exist in either of two source directories
LATENT_DIR_DESKTOP = Path(r"C:\Users\kusum\OneDrive\Desktop\latent_fvc")
LATENT_DIR_FVC = Path(r"C:\Users\kusum\OneDrive\Desktop\fingerprint\external_dataset\FVC-Latent_Test_Dataset\data")

# Ground-truth mapping (ONLY used after matching to evaluate rank)
# Confirmed via SHA-256 hash identity: clean_fvc/N.png == external_fvc/VC2000DbsDb2aN/clean/N_X.png
# Selected for diversity: IDs 1, 10, 100 from desktop; 51, 80 from fvc_dataset (spread across range)
GROUND_TRUTH_PAIRS = [
    {"latent_filename": "1_5.png",   "expected_clean_id": 1,   "entry_name": "VC2000DbsDb2a1"},
    {"latent_filename": "10_6.png",  "expected_clean_id": 10,  "entry_name": "VC2000DbsDb2a10"},
    {"latent_filename": "100_6.png", "expected_clean_id": 100, "entry_name": "VC2000DbsDb2a100"},
    {"latent_filename": "2_7.png",   "expected_clean_id": 2,   "entry_name": "VC2000DbsDb2a2"},
    {"latent_filename": "50_4.png",  "expected_clean_id": 50,  "entry_name": "VC2000DbsDb2a50"},
]

def find_latent_path(filename: str, entry_name: str) -> Path:
    """Find latent image in desktop_latent or FVC dataset directory."""
    p1 = LATENT_DIR_DESKTOP / filename
    if p1.exists():
        return p1
    p2 = LATENT_DIR_FVC / entry_name / "latent" / filename
    if p2.exists():
        return p2
    return p1  # return even if missing so assertion catches it

# Verify latent files and clean gallery IDs exist
print("=" * 80)
print("STEP 1 -- GROUND-TRUTH LATENT -> CLEAN MAPPING")
print("=" * 80)
print("Source: fingerprint_external.external_pairs (confirmed via SHA-256 hash identity)")
print()
clean_gallery_ids = set()
conn_ext = mysql.connector.connect(
    host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD,
    database=TARGET_DB
)
cur = conn_ext.cursor()
cur.execute(f"SELECT fingerprint_id FROM `{TARGET_TABLE}` WHERE generation_status='generated' AND template_data IS NOT NULL;")
for (fid,) in cur.fetchall():
    clean_gallery_ids.add(fid)
cur.close()
conn_ext.close()

valid_pairs = []
for pair in GROUND_TRUTH_PAIRS:
    lat_path = find_latent_path(pair["latent_filename"], pair["entry_name"])
    in_gallery = pair["expected_clean_id"] in clean_gallery_ids
    exists = lat_path.exists()
    status = "OK" if (exists and in_gallery) else "MISSING"
    print(f"  [{status}] latent: {pair['latent_filename']:<14}  ->  expected clean ID: {pair['expected_clean_id']:<4}  (file_exists={exists}, in_gallery={in_gallery}, path={lat_path})")
    if exists and in_gallery:
        pair["latent_path"] = lat_path
        valid_pairs.append(pair)

print(f"\nValid pairs ready for testing: {len(valid_pairs)}")

if len(valid_pairs) < 5:
    # Auto-select from DB up to 5 available desktop pairs
    print("Auto-selecting additional pairs from desktop_latent...")
    import mysql.connector as _mc
    _conn2 = _mc.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD, database="fingerprint_external")
    _cur2 = _conn2.cursor(dictionary=True)
    _cur2.execute("SELECT * FROM external_pairs;")
    _all = _cur2.fetchall()
    _cur2.close()
    _conn2.close()
    _already = {p["expected_clean_id"] for p in valid_pairs}
    for ap in _all:
        if len(valid_pairs) >= 5:
            break
        _lname = Path(ap["latent_image_path"]).name
        _cid = int(Path(ap["clean_image_path"]).stem.split("_")[0])
        if _cid in _already:
            continue
        _ename = ap["entry_name"]
        _lpath = find_latent_path(_lname, _ename)
        if _lpath.exists() and _cid in clean_gallery_ids:
            _pair = {"latent_filename": _lname, "expected_clean_id": _cid, "entry_name": _ename, "latent_path": _lpath}
            valid_pairs.append(_pair)
            _already.add(_cid)
            print(f"  Added: {_lname} -> ID {_cid}")

print(f"\nFinal valid pairs: {len(valid_pairs)}")
assert len(valid_pairs) == 5, f"Could not assemble 5 valid pairs, only got {len(valid_pairs)}"


# -----------------------------------------------------------------------
# Gallery loader
# -----------------------------------------------------------------------
def load_gallery() -> List[Dict[str, Any]]:
    conn = mysql.connector.connect(
        host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER, password=MYSQL_PASSWORD,
        database=TARGET_DB,
    )
    cursor = conn.cursor(dictionary=True)
    cursor.execute(f"""
        SELECT fingerprint_id, template_data, minutiae_count
        FROM `{TARGET_TABLE}`
        WHERE generation_status = 'generated' AND template_data IS NOT NULL
        ORDER BY fingerprint_id;
    """)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    gallery = []
    for r in rows:
        tpl = pickle.loads(r["template_data"])
        gallery.append({"fingerprint_id": r["fingerprint_id"], "template": tpl})
    return gallery


# -----------------------------------------------------------------------
# Core 1:N search (probe template dict → ranked list)
# -----------------------------------------------------------------------
def run_1toN(probe_template: Dict, gallery: List[Dict], matcher: SourceAFISMatcher) -> tuple:
    """Returns (ranked_results, total_matching_time_ms)"""
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

        results.append({
            "fingerprint_id": cand["fingerprint_id"],
            "raw_score": round(raw_score, 2),
            "matched_minutiae": pairs_count,
        })

    total_ms = (time.time() - t0) * 1000.0
    results.sort(key=lambda x: (x["raw_score"], x["matched_minutiae"]), reverse=True)
    return results, round(total_ms, 2)


def get_rank_and_score(results, expected_id):
    for idx, r in enumerate(results, start=1):
        if r["fingerprint_id"] == expected_id:
            return idx, r["raw_score"]
    return None, 0.0


def print_top5(results, expected_id):
    print(f"  {'Rank':<5} | {'fingerprint_id':<14} | {'Raw Score':<12} | {'Matched Minutiae':<17}")
    print("  " + "-" * 55)
    for rank, r in enumerate(results[:5], start=1):
        flag = " <-- Expected Candidate" if r["fingerprint_id"] == expected_id else ""
        print(f"  {rank:<5} | {r['fingerprint_id']:<14} | {r['raw_score']:<12} | {r['matched_minutiae']:<17}{flag}")


# -----------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------
print("\nLoading gallery templates from external_clean_gallery...")
t_gallery_start = time.time()
gallery = load_gallery()
gallery_load_ms = (time.time() - t_gallery_start) * 1000.0
print(f"Loaded {len(gallery)} candidate templates in {gallery_load_ms:.2f} ms")
assert 262 not in [c["fingerprint_id"] for c in gallery], "ID 262 must be excluded"

# Load DDRNet + classifier
from authentication import rectify_with_ddrnet, classify_distortion, load_ddrnet, load_classifier
print("Pre-loading DDRNet and Classifier...")
try:
    load_ddrnet()
    load_classifier()
    ddrnet_available = True
    print("DDRNet and Classifier loaded successfully.")
except Exception as e:
    ddrnet_available = False
    print(f"DDRNet/Classifier load failed: {e}")

summary_rows = []

for test_num, pair in enumerate(valid_pairs, start=1):
    latent_file = pair["latent_filename"]
    expected_id = pair["expected_clean_id"]
    latent_path = pair["latent_path"]

    print("\n" + "#" * 80)
    print(f"TEST {test_num}: latent={latent_file}  |  Expected Clean ID: {expected_id}")
    print("#" * 80)

    img_raw = cv2.imread(str(latent_path), cv2.IMREAD_GRAYSCALE)
    assert img_raw is not None, f"Cannot read latent image: {latent_path}"

    # ---- STEP 2: BASELINE (no DDRNet) ----
    print("\n  [STEP 2 — BASELINE WITHOUT DDRNet]")
    matcher_baseline = SourceAFISMatcher()
    t_ext_b = time.time()
    tpl_baseline = matcher_baseline.extract_template(img_raw)
    ext_ms_baseline = (time.time() - t_ext_b) * 1000.0
    print(f"  Query extraction time: {ext_ms_baseline:.2f} ms  |  minutiae: {tpl_baseline['minutiae_count']}")

    results_baseline, match_ms_baseline = run_1toN(tpl_baseline, gallery, matcher_baseline)
    rank_b, score_b = get_rank_and_score(results_baseline, expected_id)
    top1_b = results_baseline[0] if results_baseline else {}
    top2_b = results_baseline[1] if len(results_baseline) > 1 else {}
    margin_b = round(top1_b.get("raw_score", 0) - top2_b.get("raw_score", 0), 2)

    print(f"  Rank of expected ID {expected_id}: #{rank_b}  |  Raw Score: {score_b}  |  #1 Score: {top1_b.get('raw_score')}  |  Margin: {margin_b}  |  Search time: {match_ms_baseline:.2f} ms")
    print(f"  Top 5 Candidates:")
    print_top5(results_baseline, expected_id)

    # ---- STEP 3: WITH DDRNet ----
    print("\n  [STEP 3 — WITH DDRNet RECTIFICATION]")
    rank_d = rank_b
    score_d = score_b
    top1_d = top1_b
    margin_d = margin_b
    ext_ms_ddrnet = 0.0
    match_ms_ddrnet = 0.0
    results_ddrnet = results_baseline
    cls_result = {}
    applied_rectification = False

    if ddrnet_available:
        try:
            t_cls = time.time()
            cls_result = classify_distortion(img_raw)
            cls_ms = (time.time() - t_cls) * 1000.0
            is_distorted = cls_result["is_distorted"]
            print(f"  Distortion classifier: class={cls_result['predicted_class']}  prob_distorted={cls_result['prob_distorted']:.4f}  prob_clean={cls_result['prob_clean']:.4f}  (classification: {cls_ms:.2f} ms)")

            if is_distorted:
                t_rect = time.time()
                img_rectified = rectify_with_ddrnet(img_raw)
                rect_ms = (time.time() - t_rect) * 1000.0
                applied_rectification = True
                print(f"  DDRNet rectification applied ({rect_ms:.2f} ms)")
                probe_img_for_ddrnet = img_rectified
            else:
                print(f"  Classifier predicted CLEAN — DDRNet bypassed")
                probe_img_for_ddrnet = img_raw

            matcher_ddrnet = SourceAFISMatcher()
            t_ext_d = time.time()
            tpl_ddrnet = matcher_ddrnet.extract_template(probe_img_for_ddrnet)
            ext_ms_ddrnet = (time.time() - t_ext_d) * 1000.0
            print(f"  DDRNet probe extraction time: {ext_ms_ddrnet:.2f} ms  |  minutiae: {tpl_ddrnet['minutiae_count']}")

            results_ddrnet, match_ms_ddrnet = run_1toN(tpl_ddrnet, gallery, matcher_ddrnet)
            rank_d, score_d = get_rank_and_score(results_ddrnet, expected_id)
            top1_d = results_ddrnet[0] if results_ddrnet else {}
            top2_d = results_ddrnet[1] if len(results_ddrnet) > 1 else {}
            margin_d = round(top1_d.get("raw_score", 0) - top2_d.get("raw_score", 0), 2)

            print(f"  Rank of expected ID {expected_id}: #{rank_d}  |  Raw Score: {score_d}  |  #1 Score: {top1_d.get('raw_score')}  |  Margin: {margin_d}  |  Search time: {match_ms_ddrnet:.2f} ms")
            print(f"  Top 5 Candidates:")
            print_top5(results_ddrnet, expected_id)

        except Exception as e:
            print(f"  DDRNet pipeline error: {e}")
            traceback.print_exc()
    else:
        print("  DDRNet not available — skipping Step 3.")

    # Determine outcome
    if rank_d is not None and rank_b is not None:
        if rank_d < rank_b:
            outcome = "IMPROVED"
        elif rank_d > rank_b:
            outcome = "DEGRADED"
        else:
            outcome = "UNCHANGED"
    else:
        outcome = "N/A"

    summary_rows.append({
        "latent": latent_file,
        "expected_id": expected_id,
        "baseline_rank": rank_b,
        "baseline_score": score_b,
        "baseline_top1": top1_b.get("raw_score", 0),
        "baseline_margin": margin_b,
        "baseline_search_ms": match_ms_ddrnet,
        "ddrnet_rank": rank_d,
        "ddrnet_score": score_d,
        "ddrnet_top1": top1_d.get("raw_score", 0),
        "ddrnet_margin": margin_d,
        "ddrnet_search_ms": match_ms_ddrnet,
        "rectification_applied": applied_rectification,
        "outcome": outcome,
    })

# ---- STEP 4: COMPARISON TABLE ----
print("\n" + "=" * 100)
print("STEP 4 — COMPARISON TABLE")
print("=" * 100)
print(f"{'Latent':<14} | {'Exp ID':<6} | {'BL Rank':<7} | {'BL Score':<10} | {'DDR Rank':<8} | {'DDR Score':<10} | {'BL Margin':<10} | {'DDR Margin':<10} | {'Outcome'}")
print("-" * 100)
for r in summary_rows:
    print(f"{r['latent']:<14} | {r['expected_id']:<6} | {str(r['baseline_rank'])+'#':<7} | {r['baseline_score']:<10.2f} | {str(r['ddrnet_rank'])+'#':<8} | {r['ddrnet_score']:<10.2f} | {r['baseline_margin']:<10.2f} | {r['ddrnet_margin']:<10.2f} | {r['outcome']}")

improved = sum(1 for r in summary_rows if r["outcome"] == "IMPROVED")
degraded = sum(1 for r in summary_rows if r["outcome"] == "DEGRADED")
unchanged = sum(1 for r in summary_rows if r["outcome"] == "UNCHANGED")
print("=" * 100)
print(f"IMPROVED: {improved}  |  DEGRADED: {degraded}  |  UNCHANGED: {unchanged}")
print("=" * 100)
