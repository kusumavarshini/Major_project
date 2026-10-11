from pathlib import Path
import json
import random
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence


# ============================================================
# FVC DDRNET RECONSTRUCTION TRAINING
#
# distorted latent
#       ↓
#     DDRNet
#       ↓
# displacement field
#       ↓
# differentiable warp
#       ↓
# warped fingerprint
#       ↓
# reconstruction head
#       ↓
# final reconstructed fingerprint
#       ↓
# compare with CLEAN TARGET
#
# CLEAN fingerprint is TARGET, never an input.
# ============================================================

ROOT = Path(__file__).resolve().parent

LATENT_DIR = ROOT / "data" / "Latent_fvc"
CLEAN_DIR = ROOT / "data" / "clean_fvc_new"

OUTPUT_DIR = ROOT / "experiments" / "ddrnet_fvc_reconstruction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.json"
SPLIT_PATH = OUTPUT_DIR / "split.json"

# Existing working DDRNet baseline
INIT_CHECKPOINT = (
    ROOT
    / "experiments"
    / "ddrnet_baseline"
    / "best_model.pth"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

IMAGE_SIZE = 512
GRID_SIZE = 32
DIS_CONST = 16

BATCH_SIZE = 2
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
PATIENCE = 12

RECON_WEIGHT = 1.0
GRAD_WEIGHT = 0.5
SMOOTH_WEIGHT = 0.02
REFINE_WEIGHT = 1.0

MAX_FIELD_MAG = 60.0


# ============================================================
# SEED
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

        for i in self.ids:

            latent_path = LATENT_DIR / f"{i}.png"
            clean_path = CLEAN_DIR / f"{i}.png"

            if not latent_path.exists():
                raise FileNotFoundError(latent_path)

            if not clean_path.exists():
                raise FileNotFoundError(clean_path)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        i = self.ids[index]

        latent = cv2.imread(
            str(LATENT_DIR / f"{i}.png"),
            cv2.IMREAD_GRAYSCALE
        )

        clean = cv2.imread(
            str(CLEAN_DIR / f"{i}.png"),
            cv2.IMREAD_GRAYSCALE
        )

        if latent is None or clean is None:
            raise RuntimeError(f"Could not read pair {i}")

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

        # Existing DDRNet normalization convention
        latent = (
            255.0 - latent.astype(np.float32)
        ) / 255.0

        clean = (
            255.0 - clean.astype(np.float32)
        ) / 255.0

        # Segmentation mask
        latent_u8 = np.clip(
            255.0 - latent * 255.0,
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

        if mask.shape != (GRID_SIZE, GRID_SIZE):

            mask = cv2.resize(
                mask,
                (GRID_SIZE, GRID_SIZE),
                interpolation=cv2.INTER_NEAREST
            )

        mask = (
            mask > 0
        ).astype(np.float32)

        return {
            "latent": torch.from_numpy(
                latent
            ).unsqueeze(0),

            "clean": torch.from_numpy(
                clean
            ).unsqueeze(0),

            "mask": torch.from_numpy(
                mask
            ).unsqueeze(0),

            "id": int(i)
        }


# ============================================================
# DIFFERENTIABLE WARP
# ============================================================

def warp_with_field(image, field):

    b, _, h, w = image.shape

    full = F.interpolate(
        field,
        size=(h, w),
        mode="bilinear",
        align_corners=True
    )

    dx = full[:, 0]
    dy = full[:, 1]

    yy, xx = torch.meshgrid(

        torch.arange(
            h,
            device=image.device,
            dtype=image.dtype
        ),

        torch.arange(
            w,
            device=image.device,
            dtype=image.dtype
        ),

        indexing="ij"
    )

    xx = xx.unsqueeze(0).expand(
        b, -1, -1
    )

    yy = yy.unsqueeze(0).expand(
        b, -1, -1
    )

    # IMPORTANT:
    # This matches the corrected production
    # inference convention.
    map_x = xx + dx
    map_y = yy + dy

    nx = (
        2.0 * map_x / max(w - 1, 1)
    ) - 1.0

    ny = (
        2.0 * map_y / max(h - 1, 1)
    ) - 1.0

    grid = torch.stack(
        [nx, ny],
        dim=-1
    )

    warped = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
    )

    return warped, full


# ============================================================
# RECONSTRUCTION HEAD
# ============================================================

class ReconstructionHead(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(
                1,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                16,
                1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, warped):

        residual = self.net(warped)

        reconstructed = torch.clamp(
            warped + residual,
            0.0,
            1.0
        )

        return reconstructed


# ============================================================
# COMPLETE MODEL
# ============================================================

class DDRNetReconstruction(nn.Module):

    def __init__(self, dis_const=16):

        super().__init__()

        self.ddrnet = DDRNet_DIR(
            dis_const=dis_const
        )

        self.reconstruction = (
            ReconstructionHead()
        )

    def forward(self, image, mask):

        field, orientation = (
            self.ddrnet(
                image,
                mask
            )
        )

        # Prevent extreme displacement
        field = torch.clamp(
            field,
            -MAX_FIELD_MAG,
            MAX_FIELD_MAG
        )

        # Geometric rectification
        warped, full_field = (
            warp_with_field(
                image,
                field
            )
        )

        # Image reconstruction/refinement
        reconstructed = (
            self.reconstruction(
                warped
            )
        )

        return (
            field,
            orientation,
            warped,
            reconstructed,
            full_field
        )


# ============================================================
# LOSS FUNCTIONS
# ============================================================

def charbonnier(
    pred,
    target,
    eps=1e-3
):

    diff = pred - target

    return torch.sqrt(
        diff * diff + eps * eps
    ).mean()


def image_gradients(x):

    dx = (
        x[:, :, :, 1:]
        - x[:, :, :, :-1]
    )

    dy = (
        x[:, :, 1:, :]
        - x[:, :, :-1, :]
    )

    return dx, dy


def gradient_loss(pred, target):

    pred_dx, pred_dy = (
        image_gradients(pred)
    )

    target_dx, target_dy = (
        image_gradients(target)
    )

    return (
        charbonnier(
            pred_dx,
            target_dx
        )
        +
        charbonnier(
            pred_dy,
            target_dy
        )
    ) / 2.0


def field_smoothness(field):

    dx = (
        field[:, :, :, 1:]
        - field[:, :, :, :-1]
    )

    dy = (
        field[:, :, 1:, :]
        - field[:, :, :-1, :]
    )

    return (
        dx.abs().mean()
        +
        dy.abs().mean()
    )


def compute_losses(
    warped,
    reconstructed,
    clean,
    full_field
):

    # DDRNet geometric warp → clean
    warp_recon = charbonnier(
        warped,
        clean
    )

    warp_grad = gradient_loss(
        warped,
        clean
    )

    # Final reconstruction → clean
    final_recon = charbonnier(
        reconstructed,
        clean
    )

    final_grad = gradient_loss(
        reconstructed,
        clean
    )

    smooth = field_smoothness(
        full_field
    )

    total = (

        RECON_WEIGHT
        * warp_recon

        +

        GRAD_WEIGHT
        * warp_grad

        +

        REFINE_WEIGHT
        * final_recon

        +

        REFINE_WEIGHT
        * 0.5
        * final_grad

        +

        SMOOTH_WEIGHT
        * smooth
    )

    return {
        "loss": total,

        "warp_reconstruction":
            warp_recon,

        "warp_gradient":
            warp_grad,

        "final_reconstruction":
            final_recon,

        "final_gradient":
            final_grad,

        "smoothness":
            smooth
    }


# ============================================================
# DATA SPLIT
# ============================================================

def make_split():

    latent_ids = sorted(

        int(p.stem)

        for p in LATENT_DIR.glob("*.png")

        if p.stem.isdigit()
    )

    clean_ids = sorted(

        int(p.stem)

        for p in CLEAN_DIR.glob("*.png")

        if p.stem.isdigit()
    )

    common_ids = sorted(
        set(latent_ids)
        &
        set(clean_ids)
    )

    if len(common_ids) != 317:

        raise RuntimeError(
            f"Expected 317 paired IDs, "
            f"found {len(common_ids)}."
        )

    rng = random.Random(SEED)

    shuffled = common_ids.copy()

    rng.shuffle(shuffled)

    n = len(shuffled)

    n_train = int(
        0.70 * n
    )

    n_val = int(
        0.15 * n
    )

    train_ids = sorted(
        shuffled[:n_train]
    )

    val_ids = sorted(
        shuffled[
            n_train:
            n_train + n_val
        ]
    )

    test_ids = sorted(
        shuffled[
            n_train + n_val:
        ]
    )

    split = {
        "seed": SEED,
        "total": common_ids,
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids
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


def load_or_create_split():

    # Reuse previously established split
    if SOURCE_SPLIT.exists():

        with open(
            SOURCE_SPLIT,
            "r",
            encoding="utf-8"
        ) as f:

            split = json.load(f)

        train_ids = sorted(
            int(x)
            for x in split["train"]
        )

        val_ids = sorted(
            int(x)
            for x in split["validation"]
        )

        test_ids = sorted(
            int(x)
            for x in split["test"]
        )

        if (
            len(train_ids) == 222
            and len(val_ids) == 48
            and len(test_ids) == 47
        ):

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

    return make_split()


# ============================================================
# CHECKPOINT
# ============================================================

def load_initial_checkpoint(
    model,
    device
):

    if not INIT_CHECKPOINT.exists():

        raise FileNotFoundError(
            "Initial DDRNet checkpoint "
            f"not found:\n{INIT_CHECKPOINT}"
        )

    checkpoint = torch.load(
        INIT_CHECKPOINT,
        map_location=device,
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

    # Load only into the DDRNet portion.
    # Reconstruction head starts from scratch.
    result = (
        model.ddrnet.load_state_dict(
            state_dict,
            strict=True
        )
    )

    print(
        "Loaded baseline DDRNet:"
    )

    print(
        INIT_CHECKPOINT
    )

    print(
        "Missing parameters:",
        len(result.missing_keys)
    )

    print(
        "Unexpected parameters:",
        len(result.unexpected_keys)
    )


# ============================================================
# ONE EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    device,
    optimizer=None
):

    training = (
        optimizer is not None
    )

    if training:
        model.train()
    else:
        model.eval()

    totals = {

        "loss": 0.0,

        "warp_reconstruction": 0.0,

        "warp_gradient": 0.0,

        "final_reconstruction": 0.0,

        "final_gradient": 0.0,

        "smoothness": 0.0
    }

    total_samples = 0

    for batch in loader:

        latent = (
            batch["latent"]
            .to(
                device,
                non_blocking=True
            )
        )

        clean = (
            batch["clean"]
            .to(
                device,
                non_blocking=True
            )
        )

        mask = (
            batch["mask"]
            .to(
                device,
                non_blocking=True
            )
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            (
                field,
                orientation,
                warped,
                reconstructed,
                full_field
            ) = model(
                latent,
                mask
            )

            losses = compute_losses(
                warped,
                reconstructed,
                clean,
                full_field
            )

            loss = losses["loss"]

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0
                )

                optimizer.step()

        bs = latent.size(0)

        total_samples += bs

        for key in totals:

            totals[key] += (
                losses[key]
                .detach()
                .item()
                * bs
            )

    return {

        key:
            value / total_samples

        for key, value
        in totals.items()
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "FVC DDRNET RECONSTRUCTION TRAINING"
    )

    print("=" * 80)

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print()

    print(
        "Latent directory :",
        LATENT_DIR
    )

    print(
        "Clean directory  :",
        CLEAN_DIR
    )

    print(
        "Output directory :",
        OUTPUT_DIR
    )

    print(
        "Initial checkpoint:",
        INIT_CHECKPOINT
    )

    if not LATENT_DIR.exists():
        raise FileNotFoundError(
            LATENT_DIR
        )

    if not CLEAN_DIR.exists():
        raise FileNotFoundError(
            CLEAN_DIR
        )

    (
        train_ids,
        val_ids,
        test_ids
    ) = load_or_create_split()

    print()

    print("=" * 80)
    print("DATASET SPLIT")
    print("=" * 80)

    print(
        "Total      :",
        len(train_ids)
        + len(val_ids)
        + len(test_ids)
    )

    print(
        "Train      :",
        len(train_ids)
    )

    print(
        "Validation :",
        len(val_ids)
    )

    print(
        "Test       :",
        len(test_ids)
    )

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
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print()

    print(
        "Creating DDRNet + "
        "reconstruction head..."
    )

    model = (
        DDRNetReconstruction(
            dis_const=DIS_CONST
        )
        .to(device)
    )

    load_initial_checkpoint(
        model,
        device
    )

    trainable = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{trainable:,}"
    )

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
            patience=3
        )
    )

    best_val = float("inf")

    no_improvement = 0

    history = []

    print()

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        start = time.time()

        print()

        print("-" * 80)

        print(
            f"Epoch {epoch:02d}/"
            f"{MAX_EPOCHS}"
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            device
        )

        scheduler.step(
            val_metrics["loss"]
        )

        elapsed = (
            time.time() - start
        ) / 60.0

        lr = optimizer.param_groups[
            0
        ]["lr"]

        print()

        print("TRAIN")

        print(
            f"Loss : "
            f"{train_metrics['loss']:.6f}"
        )

        print(
            f"Warp Recon : "
            f"{train_metrics['warp_reconstruction']:.6f}"
        )

        print(
            f"Warp Grad  : "
            f"{train_metrics['warp_gradient']:.6f}"
        )

        print(
            f"Final Recon: "
            f"{train_metrics['final_reconstruction']:.6f}"
        )

        print(
            f"Final Grad : "
            f"{train_metrics['final_gradient']:.6f}"
        )

        print(
            f"Smooth     : "
            f"{train_metrics['smoothness']:.6f}"
        )

        print()

        print("VALIDATION")

        print(
            f"Loss : "
            f"{val_metrics['loss']:.6f}"
        )

        print(
            f"Warp Recon : "
            f"{val_metrics['warp_reconstruction']:.6f}"
        )

        print(
            f"Warp Grad  : "
            f"{val_metrics['warp_gradient']:.6f}"
        )

        print(
            f"Final Recon: "
            f"{val_metrics['final_reconstruction']:.6f}"
        )

        print(
            f"Final Grad : "
            f"{val_metrics['final_gradient']:.6f}"
        )

        print(
            f"Smooth     : "
            f"{val_metrics['smoothness']:.6f}"
        )

        print()

        print(
            f"Learning rate: {lr:.2e}"
        )

        print(
            f"Epoch time: "
            f"{elapsed:.2f} min"
        )

        record = {

            "epoch": epoch,

            "learning_rate": lr,

            "train": train_metrics,

            "validation": val_metrics,

            "epoch_minutes": elapsed
        }

        history.append(record)

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

        if (
            val_metrics["loss"]
            < best_val
        ):

            best_val = (
                val_metrics["loss"]
            )

            no_improvement = 0

            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "best_val_loss":
                        best_val,

                    "epoch":
                        epoch,

                    "config": {

                        "seed": SEED,

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

                        "max_epochs":
                            MAX_EPOCHS,

                        "patience":
                            PATIENCE,

                        "recon_weight":
                            RECON_WEIGHT,

                        "grad_weight":
                            GRAD_WEIGHT,

                        "smooth_weight":
                            SMOOTH_WEIGHT,

                        "refine_weight":
                            REFINE_WEIGHT,

                        "max_field_mag":
                            MAX_FIELD_MAG
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
                "NEW BEST -> "
                f"validation loss: "
                f"{best_val:.6f}"
            )

            print(
                "Checkpoint:",
                CHECKPOINT_PATH
            )

        else:

            no_improvement += 1

            print()

            print(
                f"No improvement: "
                f"{no_improvement}/"
                f"{PATIENCE}"
            )

        if (
            no_improvement
            >= PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break

    print()

    print("=" * 80)

    print(
        "FVC DDRNET RECONSTRUCTION "
        "TRAINING COMPLETE"
    )

    print("=" * 80)

    print(
        f"Best validation loss: "
        f"{best_val:.6f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH
    )

    print(
        "History:",
        HISTORY_PATH
    )

    print(
        "Split:",
        SPLIT_PATH
    )

    print()

    print(
        "The test split was NOT used "
        "for training."
    )

    print(
        "Do not replace production DDRNet "
        "until held-out test evaluation "
        "is complete."
    )


if __name__ == "__main__":
    main()