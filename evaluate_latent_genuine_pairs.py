from pathlib import Path
import sys
import json
import time
import pickle
import csv
import os

import cv2
import numpy as np
import torch
import torch.nn as nn

# ============================================================
# PROJECT PATH
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ============================================================
# PROJECT IMPORTS
# ============================================================

import authentication as auth
from matcher import SourceAFISMatcher
import mysql.connector


# ============================================================
# PATHS
# ============================================================

LATENT_DIR = ROOT / "data" / "Latent_fvc"
CLEAN_DIR = ROOT / "data" / "clean_fvc_new"

CLASSIFIER_CHECKPOINT = (
    ROOT / "experiments" / "classifier_fvc_adapted" / "best_model.pth"
)

CLASSIFIER_THRESHOLD_FILE = (
    ROOT / "experiments" / "classifier_fvc_adapted" / "threshold.json"
)

DDRN_CHECKPOINT = Path(
    r"C:\Users\kusum\OneDrive\Desktop\fingerprint"
    r"\experiments\ddrnet_baseline\best_model.pth"
)

OUTPUT_CSV = ROOT / "latent_genuine_pair_evaluation.csv"


# ============================================================
# MYSQL CONFIGURATION
# SAME CONFIGURATION AS WORKING 317 1:N EVALUATOR
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
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# STARTUP INFORMATION
# ============================================================

print("=" * 80)
print("LATENT → CLEAN GENUINE-PAIR EVALUATION")
print("=" * 80)

print(f"Device: {DEVICE}")
print(f"Latent directory: {LATENT_DIR}")
print(f"Clean directory:  {CLEAN_DIR}")
print(f"Classifier:       {CLASSIFIER_CHECKPOINT}")
print(f"Threshold file:   {CLASSIFIER_THRESHOLD_FILE}")
print(f"DDRNet:            {DDRN_CHECKPOINT}")
print(f"MySQL host:       {MYSQL_HOST}")
print(f"MySQL port:       {MYSQL_PORT}")
print(f"MySQL database:   {DB_NAME}")
print(f"MySQL table:      {DB_TABLE}")

print("=" * 80)


# ============================================================
# CLASSIFIER ARCHITECTURE
# EXACT ARCHITECTURE USED BY classifier_v3
# ============================================================

class FingerprintClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32,
                64,
                3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64,
                128,
                3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128,
                256,
                3,
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

        x = self.fc(x)

        return x


# ============================================================
# LOAD CLASSIFIER
# ============================================================

def load_classifier():

    if not CLASSIFIER_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"Classifier checkpoint not found:\n"
            f"{CLASSIFIER_CHECKPOINT}"
        )

    if not CLASSIFIER_THRESHOLD_FILE.exists():

        raise FileNotFoundError(
            f"Classifier threshold file not found:\n"
            f"{CLASSIFIER_THRESHOLD_FILE}"
        )

    model = FingerprintClassifier().to(DEVICE)

    checkpoint = torch.load(
        CLASSIFIER_CHECKPOINT,
        map_location=DEVICE
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    with open(
        CLASSIFIER_THRESHOLD_FILE,
        "r"
    ) as f:

        threshold_data = json.load(f)

    threshold = float(
        threshold_data["threshold"]
    )

    print()
    print("Classifier loaded successfully")
    print(
        f"Classifier threshold: {threshold}"
    )

    return model, threshold


# ============================================================
# CLASSIFY LATENT IMAGE
# ============================================================

@torch.no_grad()
def classify_fvc(
    model,
    image,
    threshold
):

    if image is None:

        raise ValueError(
            "Input image is None"
        )

    if len(image.shape) == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    else:

        gray = image

    resized = cv2.resize(
        gray,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )

    x = (
        resized.astype(np.float32)
        / 255.0
    )

    x = torch.from_numpy(x)

    x = x.unsqueeze(0).unsqueeze(0)

    x = x.to(DEVICE)

    logits = model(x)

    probability = torch.sigmoid(
        logits
    ).item()

    is_distorted = (
        probability >= threshold
    )

    label = (
        "DISTORTED"
        if is_distorted
        else "CLEAN"
    )

    return (
        probability,
        label,
        is_distorted
    )


# ============================================================
# LOAD DDRNET
# ============================================================

def load_ddrnet():

    if not DDRN_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"DDRNet checkpoint not found:\n"
            f"{DDRN_CHECKPOINT}"
        )

    auth.DDRNET_CHECKPOINT = (
        DDRN_CHECKPOINT
    )

    auth.DDRNET = None

    return auth.rectify_with_ddrnet


# ============================================================
# CREATE SOURCEAFIS MATCHER
# ============================================================

def create_matcher():

    return SourceAFISMatcher()


# ============================================================
# MYSQL CONNECTION
# ============================================================

def connect_db():

    return mysql.connector.connect(

        host=MYSQL_HOST,

        port=MYSQL_PORT,

        user=MYSQL_USER,

        password=MYSQL_PASSWORD,

        database=DB_NAME
    )


# ============================================================
# LOAD CLEAN GALLERY TEMPLATES
# ============================================================

def load_gallery_templates():

    print()
    print(
        "Loading clean gallery templates from MySQL..."
    )

    conn = connect_db()

    cursor = conn.cursor()

    query = f"""
        SELECT
            fingerprint_id,
            template_data
        FROM {DB_TABLE}
        WHERE template_data IS NOT NULL
        ORDER BY fingerprint_id
    """

    cursor.execute(query)

    rows = cursor.fetchall()

    cursor.close()

    conn.close()

    gallery = {}

    for fingerprint_id, template_data in rows:

        if template_data is None:

            continue

        try:

            template = pickle.loads(
                template_data
            )

        except Exception:

            continue

        gallery[
            int(fingerprint_id)
        ] = template

    print(
        f"Loaded {len(gallery)} clean templates."
    )

    return gallery


# ============================================================
# SOURCEAFIS PAIR SCORE
# ============================================================

def score_pair(
    matcher,
    query_template,
    gallery_template
):
    """
    Calculate the raw SourceAFIS score for
    one genuine query-gallery pair using
    the project's actual 1:1 matcher.
    """

    result = matcher.match_1to1(
        query_template,
        gallery_template
    )

    return float(result.raw_score)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # CHECK DATASET
    # ========================================================

    latent_files = sorted(
        LATENT_DIR.glob("*.png"),
        key=lambda p: int(p.stem)
    )

    clean_files = sorted(
        CLEAN_DIR.glob("*.png"),
        key=lambda p: int(p.stem)
    )

    if len(latent_files) != 317:

        raise RuntimeError(
            f"Expected 317 latent images, "
            f"found {len(latent_files)}"
        )

    if len(clean_files) != 317:

        raise RuntimeError(
            f"Expected 317 clean images, "
            f"found {len(clean_files)}"
        )

    print()
    print(
        f"Latent images: {len(latent_files)}"
    )

    print(
        f"Clean images:  {len(clean_files)}"
    )


    # ========================================================
    # LOAD CLASSIFIER
    # ========================================================

    classifier, classifier_threshold = (
        load_classifier()
    )


    # ========================================================
    # LOAD DDRNET
    # ========================================================

    rectify_with_ddrnet = (
        load_ddrnet()
    )

    print(
        "DDRNet loader ready."
    )


    # ========================================================
    # SOURCEAFIS
    # ========================================================

    matcher = create_matcher()

    print(
        "SourceAFIS matcher ready."
    )


    # ========================================================
    # LOAD GALLERY
    # ========================================================

    gallery = (
        load_gallery_templates()
    )

    if len(gallery) != 317:

        raise RuntimeError(
            f"Expected 317 gallery templates, "
            f"found {len(gallery)}"
        )


    # ========================================================
    # CSV COLUMNS
    # ========================================================

    fieldnames = [

        "id",

        "latent_file",

        "clean_file",

        "classifier_probability",

        "classifier_threshold",

        "classifier_label",

        "ddrnet_applied",

        "baseline_score",

        "proposed_score",

        "score_change",

        "baseline_time_ms",

        "proposed_time_ms",

        "baseline_minutiae",

        "proposed_minutiae",

        "outcome",
    ]


    results = []


    improved = 0

    degraded = 0

    unchanged = 0


    classifier_clean = 0

    classifier_distorted = 0


    # ========================================================
    # START
    # ========================================================

    print()

    print("=" * 80)

    print(
        "STARTING 317 GENUINE-PAIR EVALUATION"
    )

    print("=" * 80)


    # ========================================================
    # PROCESS ALL 317 PAIRS
    # ========================================================

    for index, latent_path in enumerate(
        latent_files,
        start=1
    ):

        fingerprint_id = int(
            latent_path.stem
        )

        clean_path = (
            CLEAN_DIR
            / f"{fingerprint_id}.png"
        )


        if not clean_path.exists():

            print(
                f"[{index}/317] "
                f"Missing clean pair for ID "
                f"{fingerprint_id}"
            )

            continue


        print(
            f"[{index}/317] "
            f"ID {fingerprint_id}",
            end=" ",
            flush=True
        )


        # ====================================================
        # READ LATENT
        # ====================================================

        latent_image = cv2.imread(
            str(latent_path),
            cv2.IMREAD_GRAYSCALE
        )


        if latent_image is None:

            print(
                "FAILED TO READ"
            )

            continue


        # ====================================================
        # CLASSIFIER
        # ====================================================

        (
            classifier_probability,
            classifier_label,
            is_distorted
        ) = classify_fvc(
            classifier,
            latent_image,
            classifier_threshold
        )


        if is_distorted:

            classifier_distorted += 1

        else:

            classifier_clean += 1


        # ====================================================
        # GET CORRESPONDING CLEAN TEMPLATE
        # ====================================================

        clean_template = gallery.get(
            fingerprint_id
        )


        if clean_template is None:

            print(
                "NO GALLERY TEMPLATE"
            )

            continue


        # ====================================================
        # BASELINE
        #
        # Latent
        #   ↓
        # SourceAFIS
        #   ↓
        # Clean corresponding template
        # ====================================================

        baseline_start = (
            time.perf_counter()
        )


        baseline_template = (
            matcher.extract_template(
                latent_image
            )
        )


        baseline_time_ms = (
            time.perf_counter()
            - baseline_start
        ) * 1000.0


        baseline_score = score_pair(
            matcher,
            baseline_template,
            clean_template
        )


        baseline_minutiae = getattr(
            baseline_template,
            "minutiae_count",
            None
        )


        # ====================================================
        # PROPOSED
        #
        # Latent
        #   ↓
        # Classifier
        #   ↓
        # DDRNet if distorted
        #   ↓
        # SourceAFIS
        #   ↓
        # Clean corresponding template
        # ====================================================

        proposed_start = (
            time.perf_counter()
        )


        ddrnet_applied = False


        if is_distorted:

            proposed_image = (
                rectify_with_ddrnet(
                    latent_image
                )
            )

            ddrnet_applied = True

        else:

            proposed_image = (
                latent_image.copy()
            )


        proposed_template = (
            matcher.extract_template(
                proposed_image
            )
        )


        proposed_time_ms = (
            time.perf_counter()
            - proposed_start
        ) * 1000.0


        proposed_score = score_pair(
            matcher,
            proposed_template,
            clean_template
        )


        proposed_minutiae = getattr(
            proposed_template,
            "minutiae_count",
            None
        )


        # ====================================================
        # SCORE CHANGE
        # ====================================================

        score_change = (
            proposed_score
            - baseline_score
        )


        if score_change > 0:

            outcome = "IMPROVED"

            improved += 1


        elif score_change < 0:

            outcome = "DEGRADED"

            degraded += 1


        else:

            outcome = "UNCHANGED"

            unchanged += 1


        # ====================================================
        # SAVE RESULT
        # ====================================================

        results.append({

            "id":
                fingerprint_id,

            "latent_file":
                latent_path.name,

            "clean_file":
                clean_path.name,

            "classifier_probability":
                round(
                    classifier_probability,
                    6
                ),

            "classifier_threshold":
                round(
                    classifier_threshold,
                    6
                ),

            "classifier_label":
                classifier_label,

            "ddrnet_applied":
                ddrnet_applied,

            "baseline_score":
                round(
                    baseline_score,
                    4
                ),

            "proposed_score":
                round(
                    proposed_score,
                    4
                ),

            "score_change":
                round(
                    score_change,
                    4
                ),

            "baseline_time_ms":
                round(
                    baseline_time_ms,
                    2
                ),

            "proposed_time_ms":
                round(
                    proposed_time_ms,
                    2
                ),

            "baseline_minutiae":
                baseline_minutiae,

            "proposed_minutiae":
                proposed_minutiae,

            "outcome":
                outcome,
        })


        # ====================================================
        # PRINT CURRENT RESULT
        # ====================================================

        print(
            f"| {classifier_label:9s}"
            f"| B={baseline_score:.2f}"
            f"| P={proposed_score:.2f}"
            f"| Δ={score_change:+.2f}"
            f"| {outcome}"
        )


    # ========================================================
    # SAVE CSV
    # ========================================================

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

        writer.writerows(
            results
        )


    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(results)


    print()

    print("=" * 80)

    print(
        "GENUINE-PAIR EVALUATION COMPLETE"
    )

    print("=" * 80)


    print(
        f"Completed pairs : "
        f"{total}/317"
    )


    # ========================================================
    # CLASSIFIER SUMMARY
    # ========================================================

    print()

    print(
        "CLASSIFIER ROUTING"
    )


    if total > 0:

        print(
            f"CLEAN       : "
            f"{classifier_clean}"
            f" ({classifier_clean / total * 100:.2f}%)"
        )

        print(
            f"DISTORTED   : "
            f"{classifier_distorted}"
            f" ({classifier_distorted / total * 100:.2f}%)"
        )


    # ========================================================
    # SCORE CHANGE SUMMARY
    # ========================================================

    print()

    print(
        "GENUINE SCORE CHANGE"
    )


    if total > 0:

        print(
            f"IMPROVED    : "
            f"{improved}"
            f" ({improved / total * 100:.2f}%)"
        )

        print(
            f"DEGRADED    : "
            f"{degraded}"
            f" ({degraded / total * 100:.2f}%)"
        )

        print(
            f"UNCHANGED   : "
            f"{unchanged}"
            f" ({unchanged / total * 100:.2f}%)"
        )


    # ========================================================
    # SCORE STATISTICS
    # ========================================================

    if total > 0:

        baseline_scores = np.array([
            r["baseline_score"]
            for r in results
        ])

        proposed_scores = np.array([
            r["proposed_score"]
            for r in results
        ])

        changes = (
            proposed_scores
            - baseline_scores
        )


        print()

        print(
            "SCORE STATISTICS"
        )


        print(
            f"Baseline mean score : "
            f"{baseline_scores.mean():.4f}"
        )


        print(
            f"Proposed mean score : "
            f"{proposed_scores.mean():.4f}"
        )


        print(
            f"Mean score change   : "
            f"{changes.mean():+.4f}"
        )


        print(
            f"Median score change : "
            f"{np.median(changes):+.4f}"
        )


        print(
            f"Minimum score change: "
            f"{changes.min():+.4f}"
        )


        print(
            f"Maximum score change: "
            f"{changes.max():+.4f}"
        )


    # ========================================================
    # OUTPUT
    # ========================================================

    print()

    print(
        "CSV saved to:"
    )

    print(
        OUTPUT_CSV
    )

    print(
        "=" * 80
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()