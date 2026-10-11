from pathlib import Path
import json
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence


# ============================================================
# FVC PAIRED DDRNET TRAINING
#
# Latent_fvc/N.png
#        ↓
#     DDRNet
#        ↓
# predicted displacement field
#        ↓
# differentiable warp
#        ↓
# clean_fvc_new/N.png
# ============================================================

ROOT = Path(__file__).resolve().parent

LATENT_DIR = ROOT / "data" / "Latent_fvc"
CLEAN_DIR = ROOT / "data" / "clean_fvc_new"

OUTPUT_DIR = ROOT / "experiments" / "ddrnet_fvc"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"
SPLIT_PATH = OUTPUT_DIR / "split.json"

# Existing DDRNet is used as initialization.
INIT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "ddrnet_baseline"
    / "best_model.pth"
)

SEED = 42

IMAGE_SIZE = 512
GRID_SIZE = 32
DIS_CONST = 16

BATCH_SIZE = 2
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
PATIENCE = 7

# Loss weights
RECON_WEIGHT = 1.0
GRAD_WEIGHT = 0.5
SMOOTH_WEIGHT = 0.02

# Safety limit for predicted displacement
MAX_FIELD_MAG = 60.0


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


# ============================================================
# DATASET
# ============================================================

class FVCPairedDataset(Dataset):

    def __init__(self, ids):

        self.ids = list(ids)

        if not self.ids:
            raise RuntimeError("Dataset split is empty.")

        for idx in self.ids:

            latent = LATENT_DIR / f"{idx}.png"
            clean = CLEAN_DIR / f"{idx}.png"

            if not latent.exists():
                raise FileNotFoundError(
                    f"Missing latent image: {latent}"
                )

            if not clean.exists():
                raise FileNotFoundError(
                    f"Missing clean image: {clean}"
                )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        idx = self.ids[index]

        latent_path = LATENT_DIR / f"{idx}.png"
        clean_path = CLEAN_DIR / f"{idx}.png"

        latent = cv2.imread(
            str(latent_path),
            cv2.IMREAD_GRAYSCALE
        )

        clean = cv2.imread(
            str(clean_path),
            cv2.IMREAD_GRAYSCALE
        )

        if latent is None:
            raise RuntimeError(
                f"Could not read: {latent_path}"
            )

        if clean is None:
            raise RuntimeError(
                f"Could not read: {clean_path}"
            )

        # ----------------------------------------------------
        # Resize to DDRNet training resolution
        # ----------------------------------------------------

        latent = cv2.resize(
            latent,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        clean = cv2.resize(
            clean,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # Same normalization used by current DDRNet inference
        # ----------------------------------------------------

        latent = (
            255.0 - latent.astype(np.float32)
        ) / 255.0

        clean = (
            255.0 - clean.astype(np.float32)
        ) / 255.0

        latent_tensor = (
            torch.from_numpy(latent)
            .unsqueeze(0)
        )

        clean_tensor = (
            torch.from_numpy(clean)
            .unsqueeze(0)
        )

        # ----------------------------------------------------
        # DDRNet segmentation mask
        # ----------------------------------------------------

        latent_u8 = np.clip(
            255.0
            - latent * 255.0,
            0,
            255
        ).astype(np.uint8)

        mask = segmentation_coherence(
            latent_u8,
            win_size=16,
            stride=8
        )

        mask = np.asarray(
            mask,
            dtype=np.float32
        )

        # DDRNet expects 14x14 mask
        if mask.shape != (
            GRID_SIZE,
            GRID_SIZE
        ):

            mask = cv2.resize(
                mask,
                (
                    GRID_SIZE,
                    GRID_SIZE
                ),
                interpolation=cv2.INTER_NEAREST
            )

        mask = (
            mask > 0
        ).astype(np.float32)

        mask_tensor = (
            torch.from_numpy(mask)
            .unsqueeze(0)
        )

        return {
            "latent": latent_tensor,
            "clean": clean_tensor,
            "mask": mask_tensor,
            "id": int(idx)
        }


# ============================================================
# DIFFERENTIABLE WARP
# ============================================================

def warp_with_field(
    image,
    field
):
    """
    image:
        B x 1 x 224 x 224

    field:
        B x 2 x 14 x 14

    Same displacement convention as current inference:

        map_x = grid_x - dx
        map_y = grid_y - dy
    """

    batch_size, _, height, width = image.shape

    # --------------------------------------------------------
    # Upsample 14x14 displacement to 224x224
    # --------------------------------------------------------

    field_full = F.interpolate(
        field,
        size=(height, width),
        mode="bilinear",
        align_corners=True
    )

    dx = field_full[:, 0]
    dy = field_full[:, 1]

    # --------------------------------------------------------
    # Pixel coordinate grid
    # --------------------------------------------------------

    yy, xx = torch.meshgrid(
        torch.arange(
            height,
            device=image.device,
            dtype=image.dtype
        ),
        torch.arange(
            width,
            device=image.device,
            dtype=image.dtype
        ),
        indexing="ij"
    )

    xx = xx.unsqueeze(0).expand(
        batch_size,
        -1,
        -1
    )

    yy = yy.unsqueeze(0).expand(
        batch_size,
        -1,
        -1
    )

    # --------------------------------------------------------
    # Inverse warp
    # --------------------------------------------------------

    map_x = xx + dx
    map_y = yy + dy

    # --------------------------------------------------------
    # Pixel coordinates -> [-1, 1]
    # --------------------------------------------------------

    norm_x = (
        2.0 * map_x / max(width - 1, 1)
        - 1.0
    )

    norm_y = (
        2.0 * map_y / max(height - 1, 1)
        - 1.0
    )

    grid = torch.stack(
        [
            norm_x,
            norm_y
        ],
        dim=-1
    )

    # --------------------------------------------------------
    # Differentiable warp
    # --------------------------------------------------------

    warped = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True
    )

    return warped, field_full


# ============================================================
# LOSSES
# ============================================================

def charbonnier_loss(
    pred,
    target,
    eps=1e-3
):

    diff = pred - target

    return torch.mean(
        torch.sqrt(
            diff * diff
            + eps * eps
        )
    )


def gradient_loss(
    pred,
    target
):

    pred_dx = (
        pred[:, :, :, 1:]
        - pred[:, :, :, :-1]
    )

    pred_dy = (
        pred[:, :, 1:, :]
        - pred[:, :, :-1, :]
    )

    target_dx = (
        target[:, :, :, 1:]
        - target[:, :, :, :-1]
    )

    target_dy = (
        target[:, :, 1:, :]
        - target[:, :, :-1, :]
    )

    loss_x = charbonnier_loss(
        pred_dx,
        target_dx
    )

    loss_y = charbonnier_loss(
        pred_dy,
        target_dy
    )

    return (
        loss_x + loss_y
    ) / 2.0


def smoothness_loss(field):

    dx = field[:, 0:1]
    dy = field[:, 1:2]

    dx_x = (
        dx[:, :, :, 1:]
        - dx[:, :, :, :-1]
    )

    dx_y = (
        dx[:, :, 1:, :]
        - dx[:, :, :-1, :]
    )

    dy_x = (
        dy[:, :, :, 1:]
        - dy[:, :, :, :-1]
    )

    dy_y = (
        dy[:, :, 1:, :]
        - dy[:, :, :-1, :]
    )

    return (
        torch.mean(torch.abs(dx_x))
        + torch.mean(torch.abs(dx_y))
        + torch.mean(torch.abs(dy_x))
        + torch.mean(torch.abs(dy_y))
    ) / 4.0


def total_loss_fn(
    rectified,
    clean,
    field
):

    reconstruction = charbonnier_loss(
        rectified,
        clean
    )

    gradient = gradient_loss(
        rectified,
        clean
    )

    smoothness = smoothness_loss(
        field
    )

    total = (
        RECON_WEIGHT * reconstruction
        + GRAD_WEIGHT * gradient
        + SMOOTH_WEIGHT * smoothness
    )

    return (
        total,
        reconstruction,
        gradient,
        smoothness
    )


# ============================================================
# LOAD INITIAL CHECKPOINT
# ============================================================

def load_initial_checkpoint(
    model
):

    if INIT_CHECKPOINT is None:

        print(
            "Training DDRNet from scratch."
        )

        return

    if not INIT_CHECKPOINT.exists():

        print(
            "Initial checkpoint not found."
        )

        print(
            "Training from scratch:"
        )

        print(
            INIT_CHECKPOINT
        )

        return

    print()
    print(
        "Loading existing DDRNet "
        "as initialization..."
    )

    print(
        INIT_CHECKPOINT
    )

    checkpoint = torch.load(
        INIT_CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    missing, unexpected = (
        model.load_state_dict(
            state_dict,
            strict=False
        )
    )

    print(
        f"Loaded parameters: "
        f"{len(state_dict)}"
    )

    print(
        f"Missing parameters: "
        f"{len(missing)}"
    )

    print(
        f"Unexpected parameters: "
        f"{len(unexpected)}"
    )


# ============================================================
# ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer,
    device,
    training=True
):

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_reconstruction = 0.0
    total_gradient = 0.0
    total_smoothness = 0.0

    total_samples = 0

    for batch in loader:

        latent = batch[
            "latent"
        ].to(device)

        clean = batch[
            "clean"
        ].to(device)

        mask = batch[
            "mask"
        ].to(device)

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            predicted_field, _ = (
                model(
                    latent,
                    mask
                )
            )

            # ------------------------------------------------
            # Prevent unstable extreme displacement
            # ------------------------------------------------

            predicted_field = torch.clamp(
                predicted_field,
                -MAX_FIELD_MAG,
                MAX_FIELD_MAG
            )

            # ------------------------------------------------
            # Warp latent -> predicted rectified
            # ------------------------------------------------

            rectified, full_field = (
                warp_with_field(
                    latent,
                    predicted_field
                )
            )

            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            (
                loss,
                reconstruction,
                gradient,
                smoothness
            ) = total_loss_fn(
                rectified,
                clean,
                full_field
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

        batch_size = (
            latent.size(0)
        )

        total_loss += (
            loss.item()
            * batch_size
        )

        total_reconstruction += (
            reconstruction.item()
            * batch_size
        )

        total_gradient += (
            gradient.item()
            * batch_size
        )

        total_smoothness += (
            smoothness.item()
            * batch_size
        )

        total_samples += batch_size

    return {

        "loss":
            total_loss
            / total_samples,

        "reconstruction":
            total_reconstruction
            / total_samples,

        "gradient":
            total_gradient
            / total_samples,

        "smoothness":
            total_smoothness
            / total_samples
    }


# ============================================================
# DATA SPLIT
# ============================================================

def make_split():

    latent_ids = sorted(
        int(p.stem)
        for p in LATENT_DIR.glob(
            "*.png"
        )
        if p.stem.isdigit()
    )

    clean_ids = sorted(
        int(p.stem)
        for p in CLEAN_DIR.glob(
            "*.png"
        )
        if p.stem.isdigit()
    )

    common_ids = sorted(
        set(latent_ids)
        .intersection(clean_ids)
    )

    if len(common_ids) != 317:

        raise RuntimeError(
            "Expected exactly 317 paired "
            f"images, found {len(common_ids)}."
        )

    rng = random.Random(
        SEED
    )

    ids = common_ids.copy()

    rng.shuffle(ids)

    n = len(ids)

    # 70 / 15 / 15
    n_train = int(
        round(n * 0.70)
    )

    n_val = int(
        round(n * 0.15)
    )

    train_ids = sorted(
        ids[:n_train]
    )

    val_ids = sorted(
        ids[
            n_train:
            n_train + n_val
        ]
    )

    test_ids = sorted(
        ids[
            n_train + n_val:
        ]
    )

    split = {

        "seed": SEED,

        "total": len(ids),

        "train":
            train_ids,

        "validation":
            val_ids,

        "test":
            test_ids
    }

    with open(
        SPLIT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            split,
            f,
            indent=2
        )

    return (
        train_ids,
        val_ids,
        test_ids
    )


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    print(
        "=" * 80
    )

    print(
        "FVC-PAIRED DDRNET "
        "RECTIFICATION TRAINING"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Latent directory:"
    )

    print(
        LATENT_DIR
    )

    print()
    print(
        "Clean directory:"
    )

    print(
        CLEAN_DIR
    )

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    if device.type == "cpu":

        print(
            "WARNING: DDRNet training "
            "is running on CPU."
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_ids,
        val_ids,
        test_ids
    ) = make_split()

    print()
    print(
        "=" * 60
    )

    print(
        "DATASET SPLIT"
    )

    print(
        "=" * 60
    )

    print(
        f"Total      : "
        f"{len(train_ids) + len(val_ids) + len(test_ids)}"
    )

    print(
        f"Train      : "
        f"{len(train_ids)}"
    )

    print(
        f"Validation : "
        f"{len(val_ids)}"
    )

    print(
        f"Test       : "
        f"{len(test_ids)}"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = (
        FVCPairedDataset(
            train_ids
        )
    )

    val_dataset = (
        FVCPairedDataset(
            val_ids
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Creating DDRNet..."
    )

    model = DDRNet_DIR(
        dis_const=DIS_CONST
    )

    load_initial_checkpoint(
        model
    )

    model = model.to(
        device
    )

    num_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        f"Trainable parameters: "
        f"{num_parameters:,}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_val_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    history = []

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 80
    )

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        start_time = time.time()

        print()
        print(
            "-" * 80
        )

        print(
            f"Epoch "
            f"{epoch:02d}/"
            f"{MAX_EPOCHS}"
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            training=True
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            training=False
        )

        scheduler.step(
            val_metrics[
                "loss"
            ]
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        elapsed = (
            time.time()
            - start_time
        )

        print()
        print(
            "TRAIN"
        )

        print(
            f"Loss : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Recon: "
            f"{train_metrics['reconstruction']:.6f}"
        )

        print(
            f"Grad : "
            f"{train_metrics['gradient']:.6f}"
        )

        print(
            f"Smooth: "
            f"{train_metrics['smoothness']:.6f}"
        )

        print()
        print(
            "VALIDATION"
        )

        print(
            f"Loss : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Recon: "
            f"{val_metrics['reconstruction']:.6f}"
        )

        print(
            f"Grad : "
            f"{val_metrics['gradient']:.6f}"
        )

        print(
            f"Smooth: "
            f"{val_metrics['smoothness']:.6f}"
        )

        print()
        print(
            f"Learning rate: "
            f"{current_lr:.2e}"
        )

        print(
            f"Epoch time: "
            f"{elapsed / 60.0:.2f} min"
        )

        # ----------------------------------------------------
        # History
        # ----------------------------------------------------

        record = {

            "epoch":
                epoch,

            "learning_rate":
                current_lr,

            "train":
                train_metrics,

            "validation":
                val_metrics,

            "epoch_minutes":
                elapsed / 60.0
        }

        history.append(
            record
        )

        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                history,
                f,
                indent=2
            )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            val_metrics["loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val_metrics["loss"]
            )

            epochs_without_improvement = 0

            torch.save(
                {

                    "epoch":
                        epoch,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_val_loss":
                        best_val_loss,

                    "config": {

                        "image_size":
                            IMAGE_SIZE,

                        "grid_size":
                            GRID_SIZE,

                        "dis_const":
                            DIS_CONST,

                        "batch_size":
                            BATCH_SIZE,

                        "learning_rate":
                            LEARNING_RATE,

                        "weight_decay":
                            WEIGHT_DECAY,

                        "recon_weight":
                            RECON_WEIGHT,

                        "grad_weight":
                            GRAD_WEIGHT,

                        "smooth_weight":
                            SMOOTH_WEIGHT,

                        "max_field_mag":
                            MAX_FIELD_MAG,

                        "seed":
                            SEED
                    },

                    "train_ids":
                        train_ids,

                    "validation_ids":
                        val_ids,

                    "test_ids":
                        test_ids
                },

                CHECKPOINT_PATH
            )

            print()
            print(
                "✓ BEST MODEL SAVED"
            )

            print(
                f"Validation loss: "
                f"{best_val_loss:.6f}"
            )

        else:

            epochs_without_improvement += 1

            print()
            print(
                f"No improvement: "
                f"{epochs_without_improvement}/"
                f"{PATIENCE}"
            )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print()
    print(
        "=" * 80
    )

    print(
        "FVC DDRNET TRAINING COMPLETE"
    )

    print(
        "=" * 80
    )

    print()
    print(
        f"Best validation loss:"
        f" {best_val_loss:.6f}"
    )

    print(
        f"Checkpoint:"
        f" {CHECKPOINT_PATH}"
    )

    print(
        f"History:"
        f" {HISTORY_PATH}"
    )

    print(
        f"Split:"
        f" {SPLIT_PATH}"
    )

    print()
    print(
        "The test split was NOT used "
        "during training."
    )

    print(
        "Do not replace the production "
        "DDRNet until the held-out test "
        "evaluation is complete."
    )


if __name__ == "__main__":
    main()
