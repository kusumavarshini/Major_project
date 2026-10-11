from pathlib import Path
import json
import random

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 224
BATCH_SIZE = 32

MAX_EPOCHS = 30
PATIENCE = 8

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Existing V3 checkpoint
V3_CHECKPOINT = Path(
    r"C:\Users\kusum\OneDrive\Desktop\fingerprint"
    r"\experiments\classifier_v3\best_model.pth"
)

# FVC dataset
CLEAN_ROOT = Path("data/clean_fvc_new")
LATENT_ROOT = Path("data/Latent_fvc")

# Output
EXP_DIR = Path("experiments/classifier_fvc_adapted")

BEST_MODEL = EXP_DIR / "best_model.pth"
THRESHOLD_FILE = EXP_DIR / "threshold.json"
HISTORY_FILE = EXP_DIR / "training_history.json"
RESULTS_FILE = EXP_DIR / "test_results.json"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# EXACT V3 MODEL ARCHITECTURE
# Reconstructed from the actual V3 checkpoint.
#
# Checkpoint:
# features.0 = Conv2d(1,32)
# features.1 = BatchNorm2d(32)
# features.4 = Conv2d(32,64)
# features.5 = BatchNorm2d(64)
# features.8 = Conv2d(64,128)
# features.9 = BatchNorm2d(128)
# features.12 = Conv2d(128,256)
# features.13 = BatchNorm2d(256)
# fc = Linear(256,1)
# ============================================================

class FingerprintCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 256,
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

        # Global average pooling.
        x = torch.mean(
            x,
            dim=(2, 3)
        )

        x = self.fc(x)

        return x


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = Image.open(path).convert("L")

    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR
    )

    image = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0)

    return tensor


# ============================================================
# DATASET
# ============================================================

class FVCDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):

        path, label = self.samples[index]

        image = load_image(path)

        label = torch.tensor(
            [float(label)],
            dtype=torch.float32
        )

        return image, label


# ============================================================
# FIND PAIRED IMAGES
# ============================================================

def get_pairs():

    clean_files = {
        p.stem: p
        for p in CLEAN_ROOT.glob("*.png")
    }

    latent_files = {
        p.stem: p
        for p in LATENT_ROOT.glob("*.png")
    }

    common_ids = sorted(
        set(clean_files) &
        set(latent_files),
        key=lambda x: (
            int(x)
            if x.isdigit()
            else x
        )
    )

    print(
        f"Clean images found : {len(clean_files)}"
    )

    print(
        f"Latent images found: {len(latent_files)}"
    )

    print(
        f"Paired IDs         : {len(common_ids)}"
    )

    if len(common_ids) != 317:

        raise RuntimeError(
            f"Expected 317 pairs, "
            f"found {len(common_ids)}."
        )

    return (
        common_ids,
        clean_files,
        latent_files
    )


# ============================================================
# SPLIT BY PAIR
# ============================================================

def split_ids(ids):

    ids = list(ids)

    rng = random.Random(SEED)

    rng.shuffle(ids)

    n = len(ids)

    n_train = int(
        n * TRAIN_RATIO
    )

    n_val = int(
        n * VAL_RATIO
    )

    train_ids = ids[:n_train]

    val_ids = ids[
        n_train:
        n_train + n_val
    ]

    test_ids = ids[
        n_train + n_val:
    ]

    return (
        train_ids,
        val_ids,
        test_ids
    )


# ============================================================
# BUILD SAMPLES
# ============================================================

def build_samples(
    ids,
    clean_files,
    latent_files
):

    samples = []

    for image_id in ids:

        # CLEAN = 0
        samples.append(
            (
                clean_files[image_id],
                0
            )
        )

        # LATENT = DISTORTED = 1
        samples.append(
            (
                latent_files[image_id],
                1
            )
        )

    return samples


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def find_best_threshold(
    labels,
    probabilities
):

    best_threshold = 0.50
    best_f1 = -1.0

    thresholds = np.linspace(
        0.05,
        0.95,
        181
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        f1 = f1_score(
            labels,
            predictions,
            zero_division=0
        )

        if f1 > best_f1:

            best_f1 = f1

            best_threshold = float(
                threshold
            )

    return (
        best_threshold,
        best_f1
    )


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    model,
    loader
):

    model.eval()

    probabilities = []
    labels = []

    for images, targets in loader:

        images = images.to(
            DEVICE
        )

        logits = model(images)

        probs = torch.sigmoid(
            logits
        )

        probabilities.extend(
            probs.cpu()
            .numpy()
            .reshape(-1)
        )

        labels.extend(
            targets.numpy()
            .reshape(-1)
        )

    return (
        np.asarray(
            labels,
            dtype=np.int64
        ),
        np.asarray(
            probabilities,
            dtype=np.float32
        )
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    labels,
    probabilities,
    threshold
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {

        "threshold":
            float(threshold),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions
                )
            ),

        "precision":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),

        "f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    labels,
                    probabilities
                )
            ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions
            ).tolist()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    EXP_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("FVC CLASSIFIER DOMAIN ADAPTATION")
    print("=" * 70)

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Max epochs   : {MAX_EPOCHS}"
    )

    print(
        f"Patience     : {PATIENCE}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print()

    # --------------------------------------------------------
    # CHECK PATHS
    # --------------------------------------------------------

    if not CLEAN_ROOT.exists():

        raise FileNotFoundError(
            f"Missing dataset: {CLEAN_ROOT}"
        )

    if not LATENT_ROOT.exists():

        raise FileNotFoundError(
            f"Missing dataset: {LATENT_ROOT}"
        )

    if not V3_CHECKPOINT.exists():

        raise FileNotFoundError(
            f"V3 checkpoint not found:\n"
            f"{V3_CHECKPOINT}"
        )

    print("Checking dataset...")

    (
        ids,
        clean_files,
        latent_files
    ) = get_pairs()

    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    (
        train_ids,
        val_ids,
        test_ids
    ) = split_ids(ids)

    print()

    print("Paired split:")

    print(
        f"Train: {len(train_ids)} pairs"
    )

    print(
        f"Val  : {len(val_ids)} pairs"
    )

    print(
        f"Test : {len(test_ids)} pairs"
    )

    train_samples = build_samples(
        train_ids,
        clean_files,
        latent_files
    )

    val_samples = build_samples(
        val_ids,
        clean_files,
        latent_files
    )

    test_samples = build_samples(
        test_ids,
        clean_files,
        latent_files
    )

    print()

    print(
        f"Training images  : "
        f"{len(train_samples)}"
    )

    print(
        f"Validation images: "
        f"{len(val_samples)}"
    )

    print(
        f"Test images      : "
        f"{len(test_samples)}"
    )

    # --------------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------------

    train_loader = DataLoader(
        FVCDataset(train_samples),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        FVCDataset(val_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        FVCDataset(test_samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    print()
    print(
        "Loading existing V3 classifier..."
    )

    model = FingerprintCNN()

    checkpoint = torch.load(
        V3_CHECKPOINT,
        map_location="cpu",
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

    model = model.to(DEVICE)

    print(
        "V3 checkpoint loaded successfully."
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    best_val_f1 = -1.0
    best_epoch = 0
    no_improvement = 0

    history = []

    print()
    print("=" * 70)
    print("STARTING FVC FINE-TUNING")
    print("=" * 70)

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            optimizer.step()

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

        train_loss = (
            total_loss /
            total_samples
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        val_labels, val_probs = predict(
            model,
            val_loader
        )

        val_threshold, val_f1 = (
            find_best_threshold(
                val_labels,
                val_probs
            )
        )

        val_metrics = metrics(
            val_labels,
            val_probs,
            val_threshold
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"Loss={train_loss:.4f} | "
            f"Val F1={val_f1:.4f} | "
            f"Val Acc={val_metrics['accuracy']:.4f} | "
            f"Threshold={val_threshold:.3f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss":
                    float(train_loss),
                "val_f1":
                    float(val_f1),
                "val_accuracy":
                    val_metrics[
                        "accuracy"
                    ],
                "val_precision":
                    val_metrics[
                        "precision"
                    ],
                "val_recall":
                    val_metrics[
                        "recall"
                    ],
                "val_roc_auc":
                    val_metrics[
                        "roc_auc"
                    ],
                "val_threshold":
                    float(
                        val_threshold
                    )
            }
        )

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1

            best_epoch = epoch

            no_improvement = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "val_f1":
                        float(val_f1),

                    "val_threshold":
                        float(
                            val_threshold
                        )
                },
                BEST_MODEL
            )

            with open(
                THRESHOLD_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    {
                        "threshold":
                            float(
                                val_threshold
                            ),

                        "validation_f1":
                            float(
                                val_f1
                            ),

                        "best_epoch":
                            epoch
                    },
                    f,
                    indent=2
                )

            print(
                f"  NEW BEST -> "
                f"epoch={epoch}, "
                f"F1={val_f1:.4f}"
            )

        else:

            no_improvement += 1

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        if no_improvement >= PATIENCE:

            print()

            print(
                "Early stopping."
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best validation F1: "
                f"{best_val_f1:.4f}"
            )

            break

    # --------------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------------

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("HELD-OUT FVC TEST")
    print("=" * 70)

    best_checkpoint = torch.load(
        BEST_MODEL,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        best_checkpoint[
            "model_state_dict"
        ]
    )

    locked_threshold = float(
        best_checkpoint[
            "val_threshold"
        ]
    )

    test_labels, test_probs = predict(
        model,
        test_loader
    )

    test_metrics = metrics(
        test_labels,
        test_probs,
        locked_threshold
    )

    test_predictions = (
        test_probs >= locked_threshold
    ).astype(int)

    test_metrics[
        "best_epoch"
    ] = int(
        best_checkpoint["epoch"]
    )

    test_metrics[
        "validation_f1"
    ] = float(
        best_checkpoint["val_f1"]
    )

    print(
        f"Best epoch      : "
        f"{test_metrics['best_epoch']}"
    )

    print(
        f"Locked threshold: "
        f"{locked_threshold:.3f}"
    )

    print(
        f"Test accuracy   : "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test precision  : "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test recall     : "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1         : "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test ROC-AUC    : "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print()
    print("Confusion matrix:")

    print(
        np.asarray(
            test_metrics[
                "confusion_matrix"
            ]
        )
    )

    print()
    print(
        classification_report(
            test_labels,
            test_predictions,
            target_names=[
                "CLEAN",
                "DISTORTED"
            ],
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "dataset": {
                    "total_pairs":
                        len(ids),

                    "train_pairs":
                        len(train_ids),

                    "validation_pairs":
                        len(val_ids),

                    "test_pairs":
                        len(test_ids)
                },

                "training": {
                    "max_epochs":
                        MAX_EPOCHS,

                    "patience":
                        PATIENCE,

                    "batch_size":
                        BATCH_SIZE,

                    "learning_rate":
                        LEARNING_RATE,

                    "weight_decay":
                        WEIGHT_DECAY,

                    "seed":
                        SEED
                },

                "best_epoch":
                    test_metrics[
                        "best_epoch"
                    ],

                "validation_f1":
                    test_metrics[
                        "validation_f1"
                    ],

                "locked_threshold":
                    locked_threshold,

                "test_metrics":
                    test_metrics
            },
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"Best model : {BEST_MODEL}"
    )

    print(
        f"Threshold  : {THRESHOLD_FILE}"
    )

    print(
        f"History    : {HISTORY_FILE}"
    )

    print(
        f"Test result: {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()