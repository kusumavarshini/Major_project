from __future__ import annotations

import os
import sys
import time
import pickle
import csv
from pathlib import Path

import cv2
import mysql.connector

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
import authentication as auth

auth.CLASSIFIER_CHECKPOINT = Path(
    r"C:\Users\kusum\OneDrive\Desktop\fingerprint\experiments\classifier_hard_negative\best_model.pth"
)

auth.DDRNET_CHECKPOINT = Path(
    r"C:\Users\kusum\OneDrive\Desktop\fingerprint\experiments\ddrnet_baseline\best_model.pth"
)

auth.CLASSIFIER = None
auth.DDRNET = None

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")

from matcher import SourceAFISMatcher
from authentication import classify_distortion, rectify_with_ddrnet


# ============================================================
# CONFIG
# ============================================================

LATENT_DIR = PROJECT_ROOT / "data" / "Latent_fvc"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean_fvc_new"

DB_NAME = "newDB"
DB_TABLE = "external_fvc_clean_gallery"

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")

OUTPUT_CSV = PROJECT_ROOT / "latent_fvc_317_1toN_evaluation.csv"


# ============================================================
# DATABASE
# ============================================================

def load_gallery():
    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=DB_NAME,
    )

    cur = conn.cursor(dictionary=True)

    cur.execute(f"""
        SELECT
            fingerprint_id,
            template_data,
            minutiae_count,
            template_version,
            generation_status
        FROM `{DB_TABLE}`
        WHERE generation_status = 'generated'
          AND template_data IS NOT NULL
        ORDER BY fingerprint_id
    """)

    rows = cur.fetchall()

    cur.close()
    conn.close()

    gallery = []

    for row in rows:
        gallery.append({
            "fingerprint_id": int(row["fingerprint_id"]),
            "template": pickle.loads(row["template_data"]),
            "minutiae_count": row["minutiae_count"],
        })

    return gallery


# ============================================================
# 1:N MATCHING
# ============================================================

def run_1toN(probe_template, gallery, matcher):

    import afis.algorithms.safis as safis

    probe_minu = probe_template.get("minutiae")

    if probe_minu is None or len(probe_minu) < 2:
        return []

    probe_edges = safis.build_edges(
        probe_minu,
        matcher.params
    )

    length_tol = float(matcher.params.length_tol) or 1e-6
    angle_tol = matcher.params.angle_tol
    max_roots = matcher.params.max_roots

    results = []

    for candidate in gallery:

        tpl = candidate["template"]

        candidate_edges = tpl.get("_edges")
        candidate_buckets = tpl.get("_buckets")

        if candidate_edges is None or candidate_buckets is None:

            candidate_minu = tpl.get("minutiae")

            if candidate_minu is not None and len(candidate_minu) >= 2:

                candidate_edges = safis.build_edges(
                    candidate_minu,
                    matcher.params
                )

                candidate_buckets = {}

                for edge in candidate_edges:

                    key = int(
                        round(edge.length / length_tol)
                    )

                    candidate_buckets.setdefault(
                        key, []
                    ).append(edge)

                tpl["_edges"] = candidate_edges
                tpl["_buckets"] = candidate_buckets

            else:

                candidate_edges = []
                candidate_buckets = {}

                tpl["_edges"] = candidate_edges
                tpl["_buckets"] = candidate_buckets

        if not candidate_edges or not probe_edges:

            raw_score = 0.0
            pairs_count = 0

        else:

            roots = []

            for probe_edge in probe_edges:

                key = int(
                    round(
                        probe_edge.length /
                        length_tol
                    )
                )

                for k in (
                    key - 1,
                    key,
                    key + 1
                ):

                    bucket = candidate_buckets.get(k)

                    if not bucket:
                        continue

                    for candidate_edge in bucket:

                        if safis._edges_compatible(
                            probe_edge,
                            candidate_edge,
                            matcher.params
                        ):

                            close = 1.0 - (
                                abs(
                                    probe_edge.length -
                                    candidate_edge.length
                                ) / length_tol
                                +
                                safis._angle_diff(
                                    probe_edge.ref_angle,
                                    candidate_edge.ref_angle
                                ) / angle_tol
                                +
                                safis._angle_diff(
                                    probe_edge.neighbor_angle,
                                    candidate_edge.neighbor_angle
                                ) / angle_tol
                            ) / 3.0

                            roots.append(
                                (
                                    close,
                                    probe_edge,
                                    candidate_edge
                                )
                            )

            if not roots:

                raw_score = 0.0
                pairs_count = 0

            else:

                roots.sort(
                    key=lambda x: x[0],
                    reverse=True
                )

                roots = roots[:max_roots]

                best_n = 0
                best_pairs_count = 0

                for _, pe, ce in roots:

                    pairing, pairs = safis._crawl(
                        probe_edges,
                        candidate_edges,
                        pe.ref,
                        ce.ref,
                        pe.neighbor,
                        ce.neighbor,
                        matcher.params
                    )

                    if len(pairing) > best_n:

                        best_n = len(pairing)
                        best_pairs_count = len(pairs)

                raw_score = (
                    float(best_n) * 5.0
                )

                if best_n >= 2:

                    raw_score += (
                        float(best_pairs_count) * 1.5
                    )

                pairs_count = best_pairs_count

        results.append({
            "fingerprint_id":
                candidate["fingerprint_id"],

            "raw_score":
                round(raw_score, 2),

            "matched_minutiae":
                pairs_count
        })

    results.sort(
        key=lambda x: (
            x["raw_score"],
            x["matched_minutiae"]
        ),
        reverse=True
    )

    return results


# ============================================================
# RANK
# ============================================================

def get_rank(results, ground_truth):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if result["fingerprint_id"] == ground_truth:

            return (
                rank,
                result["raw_score"]
            )

    return None, 0.0


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("FVC 317-QUERY 1:N AUTHENTICATION EVALUATION")
    print("=" * 100)

    # --------------------------------------------------------
    # Verify directories
    # --------------------------------------------------------

    if not LATENT_DIR.exists():
        raise FileNotFoundError(
            f"Latent directory not found: {LATENT_DIR}"
        )

    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Clean directory not found: {CLEAN_DIR}"
        )

    latent_files = sorted(
        LATENT_DIR.glob("*.png"),
        key=lambda p: int(p.stem)
    )

    clean_files = sorted(
        CLEAN_DIR.glob("*.png"),
        key=lambda p: int(p.stem)
    )

    print(f"Latent images : {len(latent_files)}")
    print(f"Clean images  : {len(clean_files)}")

    if len(latent_files) != 317:
        raise RuntimeError(
            f"Expected 317 latent images, found {len(latent_files)}"
        )

    if len(clean_files) != 317:
        raise RuntimeError(
            f"Expected 317 clean images, found {len(clean_files)}"
        )

    # --------------------------------------------------------
    # Load gallery
    # --------------------------------------------------------

    print("\nLoading newDB gallery...")

    gallery = load_gallery()

    print(
        f"Gallery templates loaded: {len(gallery)}"
    )

    if len(gallery) != 317:
        raise RuntimeError(
            f"Expected 317 gallery templates, "
            f"found {len(gallery)}"
        )

    gallery_ids = {
        x["fingerprint_id"]
        for x in gallery
    }

    expected_ids = set(range(1, 318))

    if gallery_ids != expected_ids:
        raise RuntimeError(
            "Gallery IDs are not exactly 1..317"
        )

    print("Gallery verification: OK")

    # --------------------------------------------------------
    # Pre-create matcher
    # --------------------------------------------------------

    matcher = SourceAFISMatcher()

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    rows = []

    baseline_rank1 = 0
    baseline_rank5 = 0
    baseline_rank10 = 0

    proposed_rank1 = 0
    proposed_rank5 = 0
    proposed_rank10 = 0

    improved = 0
    degraded = 0
    unchanged = 0

    # --------------------------------------------------------
    # 317 queries
    # --------------------------------------------------------

    for index, latent_path in enumerate(
        latent_files,
        start=1
    ):

        ground_truth = int(latent_path.stem)

        print(
            f"\n[{index}/317] "
            f"Query={latent_path.name} "
            f"GT={ground_truth}"
        )

        # ----------------------------------------------------
        # Load latent
        # ----------------------------------------------------

        image = cv2.imread(
            str(latent_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            print("  ERROR: image could not be loaded")

            rows.append({
                "query_id": ground_truth,
                "error": "image_load_failed"
            })

            continue

        # ====================================================
        # BASELINE
        # ====================================================

        t0 = time.time()

        try:

            t_ext = time.time()

            baseline_template = (
                matcher.extract_template(image)
            )

            baseline_extract_ms = (
                time.time() - t_ext
            ) * 1000

            t_search = time.time()

            baseline_results = run_1toN(
                baseline_template,
                gallery,
                matcher
            )

            baseline_search_ms = (
                time.time() - t_search
            ) * 1000

            baseline_total_ms = (
                time.time() - t0
            ) * 1000

            baseline_rank, baseline_score = (
                get_rank(
                    baseline_results,
                    ground_truth
                )
            )

            baseline_top1 = (
                baseline_results[0]
                if baseline_results
                else {}
            )

            baseline_top5 = [
                x["fingerprint_id"]
                for x in baseline_results[:5]
            ]

            baseline_top10 = [
                x["fingerprint_id"]
                for x in baseline_results[:10]
            ]

            if baseline_rank == 1:
                baseline_rank1 += 1

            if baseline_rank is not None and baseline_rank <= 5:
                baseline_rank5 += 1

            if baseline_rank is not None and baseline_rank <= 10:
                baseline_rank10 += 1

        except Exception as e:

            print(
                f"  BASELINE ERROR: {e}"
            )

            rows.append({
                "query_id": ground_truth,
                "error": f"baseline: {e}"
            })

            continue

        # ====================================================
        # CLASSIFIER
        # ====================================================

        classifier_start = time.time()

        try:

            classifier_result = (
                classify_distortion(image)
            )

            classifier_ms = (
                time.time() -
                classifier_start
            ) * 1000

            is_distorted = bool(
                classifier_result["is_distorted"]
            )

            predicted_class = (
                classifier_result["predicted_class"]
            )

            prob_distorted = float(
                classifier_result["prob_distorted"]
            )

        except Exception as e:

            print(
                f"  CLASSIFIER ERROR: {e}"
            )

            rows.append({
                "query_id": ground_truth,
                "baseline_rank": baseline_rank,
                "baseline_score": baseline_score,
                "error": f"classifier: {e}"
            })

            continue

        # ====================================================
        # PROPOSED PIPELINE
        # ====================================================

        rectification_applied = False
        rectification_ms = 0.0

        proposed_image = image

        if is_distorted:

            try:

                t_rect = time.time()

                proposed_image = (
                    rectify_with_ddrnet(image)
                )

                rectification_ms = (
                    time.time() -
                    t_rect
                ) * 1000

                rectification_applied = True

            except Exception as e:

                print(
                    f"  DDRNET ERROR: {e}"
                )

                rows.append({
                    "query_id": ground_truth,
                    "baseline_rank": baseline_rank,
                    "baseline_score": baseline_score,
                    "classifier": predicted_class,
                    "prob_distorted": prob_distorted,
                    "error": f"ddrnet: {e}"
                })

                continue

        # ----------------------------------------------------
        # Proposed extraction
        # ----------------------------------------------------

        t_proposed = time.time()

        try:

            t_ext = time.time()

            proposed_template = (
                matcher.extract_template(
                    proposed_image
                )
            )

            proposed_extract_ms = (
                time.time() -
                t_ext
            ) * 1000

            t_search = time.time()

            proposed_results = run_1toN(
                proposed_template,
                gallery,
                matcher
            )

            proposed_search_ms = (
                time.time() -
                t_search
            ) * 1000

            proposed_total_ms = (
                time.time() -
                t_proposed
            ) * 1000 + classifier_ms + rectification_ms

            proposed_rank, proposed_score = (
                get_rank(
                    proposed_results,
                    ground_truth
                )
            )

            proposed_top1 = (
                proposed_results[0]
                if proposed_results
                else {}
            )

            proposed_top5 = [
                x["fingerprint_id"]
                for x in proposed_results[:5]
            ]

            proposed_top10 = [
                x["fingerprint_id"]
                for x in proposed_results[:10]
            ]

            if proposed_rank == 1:
                proposed_rank1 += 1

            if proposed_rank is not None and proposed_rank <= 5:
                proposed_rank5 += 1

            if proposed_rank is not None and proposed_rank <= 10:
                proposed_rank10 += 1

        except Exception as e:

            print(
                f"  PROPOSED ERROR: {e}"
            )

            rows.append({
                "query_id": ground_truth,
                "baseline_rank": baseline_rank,
                "baseline_score": baseline_score,
                "classifier": predicted_class,
                "prob_distorted": prob_distorted,
                "error": f"proposed: {e}"
            })

            continue

        # ====================================================
        # OUTCOME
        # ====================================================

        if (
            baseline_rank is not None
            and proposed_rank is not None
        ):

            if proposed_rank < baseline_rank:
                outcome = "IMPROVED"
                improved += 1

            elif proposed_rank > baseline_rank:
                outcome = "DEGRADED"
                degraded += 1

            else:
                outcome = "UNCHANGED"
                unchanged += 1

        else:

            outcome = "NO_RANK"

        # ====================================================
        # PRINT
        # ====================================================

        print(
            f"  Classifier       : "
            f"{predicted_class} "
            f"(P(distorted)={prob_distorted:.4f})"
        )

        print(
            f"  DDRNet applied   : "
            f"{rectification_applied}"
        )

        print(
            f"  Baseline         : "
            f"Rank={baseline_rank}, "
            f"Score={baseline_score}"
        )

        print(
            f"  Proposed         : "
            f"Rank={proposed_rank}, "
            f"Score={proposed_score}"
        )

        print(
            f"  Outcome          : "
            f"{outcome}"
        )

        print(
            f"  Baseline top-5   : "
            f"{baseline_top5}"
        )

        print(
            f"  Proposed top-5   : "
            f"{proposed_top5}"
        )

        # ====================================================
        # SAVE ROW
        # ====================================================

        rows.append({

            "query_id":
                ground_truth,

            "latent_file":
                latent_path.name,

            "ground_truth_id":
                ground_truth,

            "classifier":
                predicted_class,

            "prob_distorted":
                round(prob_distorted, 6),

            "classifier_ms":
                round(classifier_ms, 2),

            "ddrnet_applied":
                rectification_applied,

            "ddrnet_ms":
                round(rectification_ms, 2),

            "baseline_minutiae":
                baseline_template["minutiae_count"],

            "proposed_minutiae":
                proposed_template["minutiae_count"],

            "baseline_rank":
                baseline_rank,

            "proposed_rank":
                proposed_rank,

            "baseline_score":
                baseline_score,

            "proposed_score":
                proposed_score,

            "baseline_top1_id":
                baseline_top1.get("fingerprint_id"),

            "baseline_top1_score":
                baseline_top1.get("raw_score"),

            "proposed_top1_id":
                proposed_top1.get("fingerprint_id"),

            "proposed_top1_score":
                proposed_top1.get("raw_score"),

            "baseline_rank1":
                baseline_rank == 1,

            "baseline_rank5":
                baseline_rank is not None
                and baseline_rank <= 5,

            "baseline_rank10":
                baseline_rank is not None
                and baseline_rank <= 10,

            "proposed_rank1":
                proposed_rank == 1,

            "proposed_rank5":
                proposed_rank is not None
                and proposed_rank <= 5,

            "proposed_rank10":
                proposed_rank is not None
                and proposed_rank <= 10,

            "baseline_extract_ms":
                round(baseline_extract_ms, 2),

            "baseline_search_ms":
                round(baseline_search_ms, 2),

            "baseline_total_ms":
                round(baseline_total_ms, 2),

            "proposed_extract_ms":
                round(proposed_extract_ms, 2),

            "proposed_search_ms":
                round(proposed_search_ms, 2),

            "proposed_total_ms":
                round(proposed_total_ms, 2),

            "outcome":
                outcome,

            "baseline_top5":
                "|".join(
                    map(str, baseline_top5)
                ),

            "proposed_top5":
                "|".join(
                    map(str, proposed_top5)
                )
        })

    # ========================================================
    # CSV
    # ========================================================

    if rows:

        fieldnames = list(rows[0].keys())

        with open(
            OUTPUT_CSV,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(rows)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    completed = len([
        r for r in rows
        if "error" not in r
    ])

    print("\n")
    print("=" * 100)
    print("FINAL 317-QUERY RESULTS")
    print("=" * 100)

    if completed:

        print(
            f"Completed queries : "
            f"{completed}/317"
        )

        print()

        print(
            f"BASELINE Rank-1   : "
            f"{baseline_rank1}/{completed} "
            f"({baseline_rank1/completed*100:.2f}%)"
        )

        print(
            f"BASELINE Rank-5   : "
            f"{baseline_rank5}/{completed} "
            f"({baseline_rank5/completed*100:.2f}%)"
        )

        print(
            f"BASELINE Rank-10  : "
            f"{baseline_rank10}/{completed} "
            f"({baseline_rank10/completed*100:.2f}%)"
        )

        print()

        print(
            f"PROPOSED Rank-1   : "
            f"{proposed_rank1}/{completed} "
            f"({proposed_rank1/completed*100:.2f}%)"
        )

        print(
            f"PROPOSED Rank-5   : "
            f"{proposed_rank5}/{completed} "
            f"({proposed_rank5/completed*100:.2f}%)"
        )

        print(
            f"PROPOSED Rank-10  : "
            f"{proposed_rank10}/{completed} "
            f"({proposed_rank10/completed*100:.2f}%)"
        )

        print()

        print(
            f"IMPROVED          : "
            f"{improved}"
        )

        print(
            f"DEGRADED          : "
            f"{degraded}"
        )

        print(
            f"UNCHANGED         : "
            f"{unchanged}"
        )

    print()

    print(
        f"CSV saved to: "
        f"{OUTPUT_CSV}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()