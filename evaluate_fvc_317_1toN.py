from __future__ import annotations

import os
import sys
import time
import pickle
import csv
import json
from pathlib import Path

import cv2
import mysql.connector
import numpy as np

import torch
import torch.nn as nn

from dotenv import load_dotenv

# ============================================================
# PROJECT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")


# ============================================================
# CHECKPOINTS
# ============================================================

# NEW FVC-ADAPTED CLASSIFIER
CLASSIFIER_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "classifier_fvc_adapted"
    / "best_model.pth"
)

THRESHOLD_FILE = (
    PROJECT_ROOT
    / "experiments"
    / "classifier_fvc_adapted"
    / "threshold.json"
)

# Existing DDRNet checkpoint
# New FVC-trained DDRNet checkpoint
DDRNET_CHECKPOINT = Path(
    r"C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition"
    r"\experiments\ddrnet_baseline\best_model.pth"
)
# ============================================================
# FVC DATASET
# ============================================================

LATENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "Latent_fvc"
)

CLEAN_DIR = (
    PROJECT_ROOT
    / "data"
    / "clean_fvc_new"
)


# ============================================================
# DATABASE
# ============================================================

DB_NAME = "newDB"
DB_TABLE = "external_fvc_clean_gallery"

MYSQL_HOST = os.getenv(
    "MYSQL_HOST",
    "localhost"
)

MYSQL_PORT = int(
    os.getenv(
        "MYSQL_PORT",
        "3306"
    )
)

MYSQL_USER = os.getenv(
    "MYSQL_USER",
    "root"
)

MYSQL_PASSWORD = os.getenv(
    "MYSQL_PASSWORD",
    "1234"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_CSV = (
    PROJECT_ROOT
    / "latent_fvc_317_1toN_evaluation_fvc_adapted.csv"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# EXACT V3 / FVC CLASSIFIER
# ============================================================

class FingerprintCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),
        )

        self.fc = nn.Linear(
            256,
            1
        )

    def forward(self, x):

        x = self.features(x)

        x = torch.mean(
            x,
            dim=(2, 3)
        )

        x = self.fc(x)

        return x


# ============================================================
# LOAD FVC CLASSIFIER
# ============================================================

def load_classifier():

    if not CLASSIFIER_CHECKPOINT.exists():

        raise FileNotFoundError(
            "FVC-adapted classifier checkpoint "
            "not found:\n"
            f"{CLASSIFIER_CHECKPOINT}"
        )

    model = FingerprintCNN()

    checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint[
            "state_dict"
        ]

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[
            key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    # --------------------------------------------------------
    # Load locked validation threshold
    # --------------------------------------------------------

    if not THRESHOLD_FILE.exists():

        raise FileNotFoundError(
            "Threshold file not found:\n"
            f"{THRESHOLD_FILE}"
        )

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        threshold_data = json.load(f)

    threshold = float(
        threshold_data["threshold"]
    )

    return model, threshold


# ============================================================
# CLASSIFIER INFERENCE
# ============================================================

@torch.no_grad()
def classify_fvc(
    model,
    image,
    threshold
):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2GRAY
    ) if len(image.shape) == 3 else image

    resized = cv2.resize(
        gray,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    arr = (
        resized.astype(
            np.float32
        )
        / 255.0
    )

    tensor = torch.from_numpy(
        arr
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(
        DEVICE
    )

    logits = model(tensor)

    probability = float(
        torch.sigmoid(
            logits
        ).item()
    )

    is_distorted = (
        probability >= threshold
    )

    predicted_class = (
        "DISTORTED"
        if is_distorted
        else "CLEAN"
    )

    return {
        "prob_distorted":
            probability,

        "is_distorted":
            bool(is_distorted),

        "predicted_class":
            predicted_class,

        "threshold":
            threshold
    }


# ============================================================
# DDRNET
# ============================================================

def load_ddrnet():

    # Import only after checkpoint path is known.
    import authentication as auth

    auth.DDRNET_CHECKPOINT = (
        DDRNET_CHECKPOINT
    )

    auth.DDRNET = None

    return auth.rectify_with_ddrnet


# ============================================================
# DATABASE
# ============================================================

def load_gallery():

    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=DB_NAME
    )

    cur = conn.cursor(
        dictionary=True
    )

    cur.execute(
        f"""
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
        """
    )

    rows = cur.fetchall()

    cur.close()
    conn.close()

    gallery = []

    for row in rows:

        gallery.append(
            {
                "fingerprint_id":
                    int(
                        row[
                            "fingerprint_id"
                        ]
                    ),

                "template":
                    pickle.loads(
                        row[
                            "template_data"
                        ]
                    ),

                "minutiae_count":
                    row[
                        "minutiae_count"
                    ]
            }
        )

    return gallery


# ============================================================
# SOURCEAFIS 1:N MATCHING
# ============================================================

def run_1toN(
    probe_template,
    gallery,
    matcher
):

    import afis.algorithms.safis as safis

    probe_minu = (
        probe_template.get(
            "minutiae"
        )
    )

    if (
        probe_minu is None
        or len(probe_minu) < 2
    ):

        return []

    probe_edges = safis.build_edges(
        probe_minu,
        matcher.params
    )

    length_tol = (
        float(
            matcher.params.length_tol
        )
        or 1e-6
    )

    angle_tol = (
        matcher.params.angle_tol
    )

    max_roots = (
        matcher.params.max_roots
    )

    results = []

    for candidate in gallery:

        tpl = candidate["template"]

        candidate_edges = (
            tpl.get("_edges")
        )

        candidate_buckets = (
            tpl.get("_buckets")
        )

        if (
            candidate_edges is None
            or candidate_buckets is None
        ):

            candidate_minu = (
                tpl.get("minutiae")
            )

            if (
                candidate_minu is not None
                and len(candidate_minu) >= 2
            ):

                candidate_edges = (
                    safis.build_edges(
                        candidate_minu,
                        matcher.params
                    )
                )

                candidate_buckets = {}

                for edge in candidate_edges:

                    key = int(
                        round(
                            edge.length
                            / length_tol
                        )
                    )

                    candidate_buckets.setdefault(
                        key,
                        []
                    ).append(edge)

                tpl["_edges"] = (
                    candidate_edges
                )

                tpl["_buckets"] = (
                    candidate_buckets
                )

            else:

                candidate_edges = []

                candidate_buckets = {}

                tpl["_edges"] = (
                    candidate_edges
                )

                tpl["_buckets"] = (
                    candidate_buckets
                )

        if (
            not candidate_edges
            or not probe_edges
        ):

            raw_score = 0.0
            pairs_count = 0

        else:

            roots = []

            for probe_edge in probe_edges:

                key = int(
                    round(
                        probe_edge.length
                        / length_tol
                    )
                )

                for k in (
                    key - 1,
                    key,
                    key + 1
                ):

                    bucket = (
                        candidate_buckets.get(k)
                    )

                    if not bucket:
                        continue

                    for candidate_edge in bucket:

                        if safis._edges_compatible(
                            probe_edge,
                            candidate_edge,
                            matcher.params
                        ):

                            close = (
                                1.0
                                - (
                                    abs(
                                        probe_edge.length
                                        - candidate_edge.length
                                    )
                                    / length_tol
                                    +
                                    safis._angle_diff(
                                        probe_edge.ref_angle,
                                        candidate_edge.ref_angle
                                    )
                                    / angle_tol
                                    +
                                    safis._angle_diff(
                                        probe_edge.neighbor_angle,
                                        candidate_edge.neighbor_angle
                                    )
                                    / angle_tol
                                )
                                / 3.0
                            )

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

                roots = roots[
                    :max_roots
                ]

                best_n = 0
                best_pairs_count = 0

                for (
                    _,
                    pe,
                    ce
                ) in roots:

                    pairing, pairs = (
                        safis._crawl(
                            probe_edges,
                            candidate_edges,
                            pe.ref,
                            ce.ref,
                            pe.neighbor,
                            ce.neighbor,
                            matcher.params
                        )
                    )

                    if len(pairing) > best_n:

                        best_n = len(
                            pairing
                        )

                        best_pairs_count = len(
                            pairs
                        )

                raw_score = (
                    float(best_n)
                    * 5.0
                )

                if best_n >= 2:

                    raw_score += (
                        float(
                            best_pairs_count
                        )
                        * 1.5
                    )

                pairs_count = (
                    best_pairs_count
                )

        results.append(
            {
                "fingerprint_id":
                    candidate[
                        "fingerprint_id"
                    ],

                "raw_score":
                    round(
                        raw_score,
                        2
                    ),

                "matched_minutiae":
                    pairs_count
            }
        )

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

def get_rank(
    results,
    ground_truth
):

    for rank, result in enumerate(
        results,
        start=1
    ):

        if (
            result["fingerprint_id"]
            == ground_truth
        ):

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
    print(
        "FVC 317-QUERY 1:N "
        "AUTHENTICATION EVALUATION"
    )
    print("=" * 100)

    print()
    print(
        f"Classifier checkpoint:"
    )
    print(
        f"  {CLASSIFIER_CHECKPOINT}"
    )

    print(
        f"DDRNet checkpoint:"
    )
    print(
        f"  {DDRNET_CHECKPOINT}"
    )

    # --------------------------------------------------------
    # Load classifier
    # --------------------------------------------------------

    classifier, classifier_threshold = (
        load_classifier()
    )

    print()
    print(
        "FVC classifier loaded successfully."
    )

    print(
        f"Locked classifier threshold: "
        f"{classifier_threshold:.3f}"
    )

    # --------------------------------------------------------
    # Load DDRNet
    # --------------------------------------------------------

    rectify_with_ddrnet = (
        load_ddrnet()
    )

    # --------------------------------------------------------
    # Verify directories
    # --------------------------------------------------------

    if not LATENT_DIR.exists():
        raise FileNotFoundError(
            f"Latent directory not found: "
            f"{LATENT_DIR}"
        )

    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            f"Clean directory not found: "
            f"{CLEAN_DIR}"
        )

    SPLIT_PATH = (
        PROJECT_ROOT
        / "experiments"
        / "ddrnet_fvc"
        / "split.json"
    )

    with open(
        SPLIT_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        split_data = json.load(f)

    test_ids = set(
        int(x)
        for x in split_data["test"]
    )

    latent_files = sorted(
        [
            p
            for p in LATENT_DIR.glob("*.png")
            if int(p.stem) in test_ids
        ],
        key=lambda p: int(p.stem)
    )

    clean_files = sorted(
        [
            p
            for p in CLEAN_DIR.glob("*.png")
            if int(p.stem) in test_ids
        ],
        key=lambda p: int(p.stem)
    )

    clean_files = sorted(
        [
            p
            for p in CLEAN_DIR.glob("*.png")
            if int(p.stem) in test_ids
        ],
        key=lambda p: int(
            p.stem
        )
    )

    print()
    print(
        f"Latent images : "
        f"{len(latent_files)}"
    )

    print(
        f"Clean images  : "
        f"{len(clean_files)}"
    )

    if len(latent_files) != 47:

        raise RuntimeError(
            f"Expected 47 latent images, "
            f"found {len(latent_files)}"
        )

    if len(clean_files) != 47:

        raise RuntimeError(
            f"Expected 47 clean images, "
            f"found {len(clean_files)}"
        )

    # --------------------------------------------------------
    # Load gallery
    # --------------------------------------------------------

    print()
    print(
        "Loading newDB gallery..."
    )

    gallery = load_gallery()

    print(
        f"Gallery templates loaded: "
        f"{len(gallery)}"
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

    expected_ids = set(
        range(1, 318)
    )

    if gallery_ids != expected_ids:

        raise RuntimeError(
            "Gallery IDs are not exactly 1..317"
        )

    print(
        "Gallery verification: OK"
    )

    # --------------------------------------------------------
    # SourceAFIS matcher
    # --------------------------------------------------------

    from matcher import SourceAFISMatcher

    matcher = SourceAFISMatcher()

    # --------------------------------------------------------
    # Counters
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

    classifier_clean = 0
    classifier_distorted = 0

    # --------------------------------------------------------
    # 317 queries
    # --------------------------------------------------------

    for index, latent_path in enumerate(
        latent_files,
        start=1
    ):

        ground_truth = int(
            latent_path.stem
        )

        print()
        print(
            f"[{index}/317] "
            f"Query={latent_path.name} "
            f"GT={ground_truth}"
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = cv2.imread(
            str(latent_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:

            print(
                "  ERROR: image could not be loaded"
            )

            rows.append(
                {
                    "query_id":
                        ground_truth,

                    "error":
                        "image_load_failed"
                }
            )

            continue

        # ====================================================
        # BASELINE
        # Raw latent -> SourceAFIS
        # ====================================================

        t0 = time.time()

        try:

            t_ext = time.time()

            baseline_template = (
                matcher.extract_template(
                    image
                )
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

            (
                baseline_rank,
                baseline_score
            ) = get_rank(
                baseline_results,
                ground_truth
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

            if (
                baseline_rank is not None
                and baseline_rank <= 5
            ):
                baseline_rank5 += 1

            if (
                baseline_rank is not None
                and baseline_rank <= 10
            ):
                baseline_rank10 += 1

        except Exception as e:

            print(
                f"  BASELINE ERROR: {e}"
            )

            rows.append(
                {
                    "query_id":
                        ground_truth,

                    "error":
                        f"baseline: {e}"
                }
            )

            continue

        # ====================================================
        # NEW FVC CLASSIFIER
        # ====================================================

        classifier_start = time.time()

        try:

            classifier_result = (
                classify_fvc(
                    classifier,
                    image,
                    classifier_threshold
                )
            )

            classifier_ms = (
                time.time()
                - classifier_start
            ) * 1000

            is_distorted = bool(
                classifier_result[
                    "is_distorted"
                ]
            )

            predicted_class = (
                classifier_result[
                    "predicted_class"
                ]
            )

            prob_distorted = float(
                classifier_result[
                    "prob_distorted"
                ]
            )

            if is_distorted:

                classifier_distorted += 1

            else:

                classifier_clean += 1

        except Exception as e:

            print(
                f"  CLASSIFIER ERROR: {e}"
            )

            rows.append(
                {
                    "query_id":
                        ground_truth,

                    "baseline_rank":
                        baseline_rank,

                    "baseline_score":
                        baseline_score,

                    "error":
                        f"classifier: {e}"
                }
            )

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
                    rectify_with_ddrnet(
                        image
                    )
                )

                rectification_ms = (
                    time.time()
                    - t_rect
                ) * 1000

                rectification_applied = True

            except Exception as e:

                print(
                    f"  DDRNET ERROR: {e}"
                )

                rows.append(
                    {
                        "query_id":
                            ground_truth,

                        "baseline_rank":
                            baseline_rank,

                        "baseline_score":
                            baseline_score,

                        "classifier":
                            predicted_class,

                        "prob_distorted":
                            prob_distorted,

                        "error":
                            f"ddrnet: {e}"
                    }
                )

                continue

        # ----------------------------------------------------
        # Proposed extraction + search
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
                time.time()
                - t_ext
            ) * 1000

            t_search = time.time()

            proposed_results = run_1toN(
                proposed_template,
                gallery,
                matcher
            )

            proposed_search_ms = (
                time.time()
                - t_search
            ) * 1000

            proposed_total_ms = (
                time.time()
                - t_proposed
            ) * 1000 + (
                classifier_ms
                + rectification_ms
            )

            (
                proposed_rank,
                proposed_score
            ) = get_rank(
                proposed_results,
                ground_truth
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

            if (
                proposed_rank is not None
                and proposed_rank <= 5
            ):
                proposed_rank5 += 1

            if (
                proposed_rank is not None
                and proposed_rank <= 10
            ):
                proposed_rank10 += 1

        except Exception as e:

            print(
                f"  PROPOSED ERROR: {e}"
            )

            rows.append(
                {
                    "query_id":
                        ground_truth,

                    "baseline_rank":
                        baseline_rank,

                    "baseline_score":
                        baseline_score,

                    "classifier":
                        predicted_class,

                    "prob_distorted":
                        prob_distorted,

                    "error":
                        f"proposed: {e}"
                }
            )

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
            f"(P(distorted)="
            f"{prob_distorted:.4f})"
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

        rows.append(
            {
                "query_id":
                    ground_truth,

                "latent_file":
                    latent_path.name,

                "ground_truth_id":
                    ground_truth,

                "classifier":
                    predicted_class,

                "prob_distorted":
                    round(
                        prob_distorted,
                        6
                    ),

                "classifier_threshold":
                    classifier_threshold,

                "classifier_ms":
                    round(
                        classifier_ms,
                        2
                    ),

                "ddrnet_applied":
                    rectification_applied,

                "ddrnet_ms":
                    round(
                        rectification_ms,
                        2
                    ),

                "baseline_minutiae":
                    baseline_template[
                        "minutiae_count"
                    ],

                "proposed_minutiae":
                    proposed_template[
                        "minutiae_count"
                    ],

                "baseline_rank":
                    baseline_rank,

                "proposed_rank":
                    proposed_rank,

                "baseline_score":
                    baseline_score,

                "proposed_score":
                    proposed_score,

                "baseline_top1_id":
                    baseline_top1.get(
                        "fingerprint_id"
                    ),

                "baseline_top1_score":
                    baseline_top1.get(
                        "raw_score"
                    ),

                "proposed_top1_id":
                    proposed_top1.get(
                        "fingerprint_id"
                    ),

                "proposed_top1_score":
                    proposed_top1.get(
                        "raw_score"
                    ),

                "baseline_rank1":
                    baseline_rank == 1,

                "baseline_rank5":
                    (
                        baseline_rank is not None
                        and baseline_rank <= 5
                    ),

                "baseline_rank10":
                    (
                        baseline_rank is not None
                        and baseline_rank <= 10
                    ),

                "proposed_rank1":
                    proposed_rank == 1,

                "proposed_rank5":
                    (
                        proposed_rank is not None
                        and proposed_rank <= 5
                    ),

                "proposed_rank10":
                    (
                        proposed_rank is not None
                        and proposed_rank <= 10
                    ),

                "baseline_extract_ms":
                    round(
                        baseline_extract_ms,
                        2
                    ),

                "baseline_search_ms":
                    round(
                        baseline_search_ms,
                        2
                    ),

                "baseline_total_ms":
                    round(
                        baseline_total_ms,
                        2
                    ),

                "proposed_extract_ms":
                    round(
                        proposed_extract_ms,
                        2
                    ),

                "proposed_search_ms":
                    round(
                        proposed_search_ms,
                        2
                    ),

                "proposed_total_ms":
                    round(
                        proposed_total_ms,
                        2
                    ),

                "outcome":
                    outcome,

                "baseline_top5":
                    "|".join(
                        map(
                            str,
                            baseline_top5
                        )
                    ),

                "proposed_top5":
                    "|".join(
                        map(
                            str,
                            proposed_top5
                        )
                    )
            }
        )

    # ========================================================
    # SAVE CSV
    # ========================================================

    if rows:

        fieldnames = list(
            rows[0].keys()
        )

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

    completed = len(
        [
            r for r in rows
            if "error" not in r
        ]
    )

    print()
    print("=" * 100)
    print(
        "FINAL 317-QUERY RESULTS"
    )
    print("=" * 100)

    print(
        f"Completed queries : "
        f"{completed}/317"
    )

    if completed:

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
            f"Classifier CLEAN      : "
            f"{classifier_clean}"
        )

        print(
            f"Classifier DISTORTED  : "
            f"{classifier_distorted}"
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