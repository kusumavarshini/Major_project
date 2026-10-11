from pathlib import Path
import sys
import os
import json
import pickle

import cv2
import mysql.connector
import torch
import torch.nn as nn

# ============================================================
# PROJECT SETUP
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import authentication as auth
from matcher_experimental import SourceAFISMatcher


# ============================================================
# PATHS
# ============================================================

LATENT_IMAGE = ROOT / "data" / "latent_fvc" / "1.png"
GROUND_TRUTH = int(LATENT_IMAGE.stem)

CLASSIFIER_CHECKPOINT = (
    ROOT
    / "experiments"
    / "classifier_fvc_adapted"
    / "best_model.pth"
)

THRESHOLD_FILE = (
    ROOT
    / "experiments"
    / "classifier_fvc_adapted"
    / "threshold.json"
)

DDRNET_CHECKPOINT = (
    ROOT
    / "experiments"
    / "ddrnet_fvc_finetuned"
    / "best_model.pth"
)


# ============================================================
# DATABASE
# ============================================================

DB_NAME = "newDB"
DB_TABLE = "external_fvc_clean_gallery"

MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "1234")


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASSIFIER
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

        self.fc = nn.Linear(256, 1)

    def forward(self, x):

        x = self.features(x)

        x = torch.mean(
            x,
            dim=(2, 3)
        )

        return self.fc(x)


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier():

    print("\nLoading classifier...")

    model = FingerprintCNN().to(DEVICE)

    checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key[7:]

        cleaned_state_dict[key] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.eval()

    with open(
        THRESHOLD_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        threshold_data = json.load(f)

    threshold = float(
        threshold_data["threshold"]
    )

    print(
        f"Classifier threshold: {threshold:.3f}"
    )

    return model, threshold


# ============================================================
# CLASSIFIER INFERENCE
# ============================================================

@torch.no_grad()
def classify_image(
    model,
    image,
    threshold
):

    gray = image

    resized = cv2.resize(
        gray,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    arr = (
        resized.astype("float32")
        / 255.0
    )

    tensor = torch.from_numpy(
        arr
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(DEVICE)

    logits = model(tensor)

    probability = float(
        torch.sigmoid(logits).item()
    )

    is_distorted = (
        probability >= threshold
    )

    label = (
        "DISTORTED"
        if is_distorted
        else "CLEAN"
    )

    return probability, label, is_distorted


# ============================================================
# LOAD DDRNET
# ============================================================

def load_ddrnet():

    print("\nLoading DDRNet...")

    if not DDRNET_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"DDRNet checkpoint not found:\n"
            f"{DDRNET_CHECKPOINT}"
        )

    auth.DDRNET_CHECKPOINT = (
        DDRNET_CHECKPOINT
    )

    auth.DDRNET = None

    return auth.rectify_with_ddrnet


# ============================================================
# LOAD DATABASE GALLERY
# ============================================================

def load_gallery():

    print("\nLoading 317 gallery templates...")

    conn = mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=DB_NAME
    )

    cursor = conn.cursor(
        dictionary=True
    )

    cursor.execute(
        f"""
        SELECT
            fingerprint_id,
            template_data,
            minutiae_count
        FROM `{DB_TABLE}`
        WHERE generation_status = 'generated'
          AND template_data IS NOT NULL
        ORDER BY fingerprint_id
        """
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    gallery = []

    for row in rows:

        template = pickle.loads(
            row["template_data"]
        )

        gallery.append({
            "fingerprint_id":
                int(row["fingerprint_id"]),

            "template":
                template,

            "minutiae_count":
                row["minutiae_count"]
        })

    print(
        f"Gallery loaded: {len(gallery)} templates"
    )

    if len(gallery) != 317:

        raise RuntimeError(
            f"Expected 317 templates, "
            f"found {len(gallery)}"
        )

    return gallery


# ============================================================
# SOURCEAFIS 1:N
# SAME LOGIC AS YOUR 317 EVALUATOR
# ============================================================

def run_1toN(
    probe_template,
    gallery,
    matcher
):

    import afis.algorithms.safis as safis

    probe_minu = (
        probe_template.get("minutiae")
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
        float(matcher.params.length_tol)
        or 1e-6
    )

    angle_tol = matcher.params.angle_tol
    max_roots = matcher.params.max_roots

    results = []

    for candidate in gallery:

        tpl = candidate["template"]

        candidate_edges = tpl.get("_edges")
        candidate_buckets = tpl.get("_buckets")

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

                tpl["_edges"] = candidate_edges
                tpl["_buckets"] = candidate_buckets

            else:

                candidate_edges = []
                candidate_buckets = {}

                tpl["_edges"] = candidate_edges
                tpl["_buckets"] = candidate_buckets

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

                roots = roots[:max_roots]

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

                        best_n = len(pairing)
                        best_pairs_count = len(pairs)

                raw_score = (
                    float(best_n) * 5.0
                )

                if best_n >= 2:

                    raw_score += (
                        float(best_pairs_count)
                        * 1.5
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
# FIND CORRECT ID RANK
# ============================================================

def find_rank(
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

            return rank, result

    return None, None


# ============================================================
# PRINT TOP MATCHES
# ============================================================

def print_top_matches(
    title,
    results,
    ground_truth
):

    print()
    print("=" * 80)
    print(title)
    print("=" * 80)

    rank, correct_result = find_rank(
        results,
        ground_truth
    )

    if correct_result is not None:

        print(
            f"Correct ID       : {ground_truth}"
        )

        print(
            f"Correct rank     : {rank}"
        )

        print(
            f"Correct score    : "
            f"{correct_result['raw_score']:.2f}"
        )

        print(
            f"Matched minutiae : "
            f"{correct_result['matched_minutiae']}"
        )

    print()
    print("TOP 10 MATCHES:")

    for i, result in enumerate(
        results[:10],
        start=1
    ):

        marker = (
            "  <-- CORRECT"
            if result["fingerprint_id"]
            == ground_truth
            else ""
        )

        print(
            f"{i:2d}. "
            f"ID={result['fingerprint_id']:3d}  "
            f"Score={result['raw_score']:6.2f}  "
            f"Pairs={result['matched_minutiae']:3d}"
            f"{marker}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("SINGLE LATENT FINGERPRINT TEST")
    print("=" * 80)

    print(
        f"Input: {LATENT_IMAGE}"
    )

    if not LATENT_IMAGE.exists():

        raise FileNotFoundError(
            f"Latent image not found:\n"
            f"{LATENT_IMAGE}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        str(LATENT_IMAGE),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        raise RuntimeError(
            "Could not read latent image"
        )

    print(
        f"Image size: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    # --------------------------------------------------------
    # Load components
    # --------------------------------------------------------

    classifier, threshold = (
        load_classifier()
    )
    rectify_with_ddrnet = load_ddrnet()
    gallery = load_gallery()

    matcher = SourceAFISMatcher()

    print("\nSourceAFIS matcher ready.")

    # --------------------------------------------------------
    # Classifier
    # --------------------------------------------------------

    probability, label, is_distorted = (
        classify_image(
            classifier,
            image,
            threshold
        )
    )

    print()
    print("=" * 80)
    print("CLASSIFIER RESULT")
    print("=" * 80)

    print(
        f"Probability distorted : "
        f"{probability:.6f}"
    )

    print(
        f"Threshold              : "
        f"{threshold:.6f}"
    )

    print(
        f"Classification         : "
        f"{label}"
    )

    # ========================================================
    # BASELINE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "BASELINE: RAW LATENT -> "
        "SOURCEAFIS -> 317 DATABASE"
    )
    print("=" * 80)

    baseline_template = (
        matcher.extract_template(
            image
        )
    )

    print(
        f"Baseline minutiae: "
        f"{baseline_template['minutiae_count']}"
    )

    baseline_results = run_1toN(
        baseline_template,
        gallery,
        matcher
    )

    print_top_matches(
        "BASELINE RESULTS",
        baseline_results,
        ground_truth=GROUND_TRUTH
    )

    # ========================================================
    # PROPOSED
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PROPOSED: CLASSIFIER -> SOURCEAFIS -> "
        "317 DATABASE"
    )
    print("=" * 80)

      # ========================================================
    # DDRNet RECTIFICATION
    # ========================================================

    if is_distorted:
        print()
        print("Applying DDRNet rectification...")

        proposed_image = rectify_with_ddrnet(image)
        # Save the rectified fingerprint
        output_path = ROOT / "experiments" / "latent_1_rectified.png"
        cv2.imwrite(str(output_path), proposed_image)
        print(f"Saved rectified image: {output_path}")

       

        ddrnet_applied = True

    else:
        print()
        print("Fingerprint classified as CLEAN.")
        print("DDRNet not applied.")

        proposed_image = image.copy()

        ddrnet_applied = False

    # --------------------------------------------------------
    # Extract minutiae from proposed image
    # --------------------------------------------------------

    proposed_template = (
        matcher.extract_template(
            proposed_image
        )
    )

    print(
        f"Proposed minutiae: "
        f"{proposed_template['minutiae_count']}"
    )

    # --------------------------------------------------------
    # Search proposed template against ALL 317
    # --------------------------------------------------------

    proposed_results = run_1toN(
        proposed_template,
        gallery,
        matcher
    )

    print_top_matches(
        "PROPOSED RESULTS",
        proposed_results,
        ground_truth=GROUND_TRUTH
    )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    baseline_rank, baseline_correct = (
        find_rank(
            baseline_results,
            GROUND_TRUTH
        )
    )

    proposed_rank, proposed_correct = (
        find_rank(
            proposed_results,
            GROUND_TRUTH
        )
    )

    print()
    print("=" * 80)
    print(
        f"FINAL COMPARISON FOR LATENT "
        f"{LATENT_IMAGE.name}"
    )
    print("=" * 80)

    print(
        f"Classifier       : {label}"
    )

    print(
        f"DDRNet applied   : {ddrnet_applied}"
    )

    print()

    print("BASELINE")

    print(
        f"  Correct ID     : {GROUND_TRUTH}"
    )

    print(
        f"  Correct rank   : {baseline_rank}"
    )

    if baseline_correct is not None:

        print(
            f"  Correct score  : "
            f"{baseline_correct['raw_score']:.2f}"
        )

    print()

    print("PROPOSED")

    print(
        f"  Correct ID     : {GROUND_TRUTH}"
    )

    print(
        f"  Correct rank   : {proposed_rank}"
    )

    if proposed_correct is not None:

        print(
            f"  Correct score  : "
            f"{proposed_correct['raw_score']:.2f}"
        )

    print()

    if (
        baseline_rank is not None
        and proposed_rank is not None
    ):

        if proposed_rank < baseline_rank:

            print(
                "RESULT: PROPOSED IMPROVED the rank."
            )

        elif proposed_rank > baseline_rank:

            print(
                "RESULT: PROPOSED DEGRADED the rank."
            )

        else:

            print(
                "RESULT: Rank UNCHANGED."
            )

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()