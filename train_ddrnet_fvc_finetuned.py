import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from DDRNet.models.DDRNet_DIR import DDRNet_DIR
from DDRNet.tools.fp_segmentation import segmentation_coherence


# ============================================================
# CONFIG
# ============================================================

ROOT = Path(__file__).resolve().parent

LATENT_DIR = ROOT / "data" / "Latent_fvc"
CLEAN_DIR = ROOT / "data" / "clean_fvc_new"

BASELINE_CHECKPOINT = (
    ROOT / "experiments" / "ddrnet_baseline" / "best_model.pth"
)

OUTPUT_DIR = ROOT / "experiments" / "ddrnet_fvc_finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = OUTPUT_DIR / "best_model.pth"
HISTORY_FILE = OUTPUT_DIR / "training_history.json"

SEED = 42

IMAGE_SIZE = 224
MASK_SIZE = 14

BATCH_SIZE = 4
NUM_WORKERS = 0

LR = 1e-5
WEIGHT_DECAY = 1e-4

MAX_EPOCHS = 30
PATIENCE = 8

# Loss weights
LAMBDA_GRAD = 0.20
LAMBDA_SMOOTH = 0.01
LAMBDA_FIELD = 0.10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class FVCFinetuneDataset(Dataset):

    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):

        image_id = self.ids[index]

        latent_path = LATENT_DIR / f"{image_id}.png"
        clean_path = CLEAN_DIR / f"{image_id}.png"

        latent = cv2.imread(str(latent_path), cv2.IMREAD_GRAYSCALE)
        clean = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)

        if latent is None:
            raise FileNotFoundError(latent_path)

        if clean is None:
            raise FileNotFoundError(clean_path)

        latent = cv2.resize(
            latent,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        clean = cv2.resize(
            clean,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        # DDRNet expects normalized fingerprint intensity.
        latent_norm = (255.0 - latent.astype(np.float32)) / 255.0
        clean_norm = clean.astype(np.float32) / 255.0

        latent_tensor = torch.from_numpy(
            latent_norm
        ).unsqueeze(0)

        clean_tensor = torch.from_numpy(
            clean_norm
        ).unsqueeze(0)

        # Original DDRNet mask generation.
        mask = segmentation_coherence(
            latent,
            win_size=16,
            stride=8,
        )

        mask = np.asarray(mask, dtype=np.float32)

        mask = cv2.resize(
            mask,
            (MASK_SIZE, MASK_SIZE),
            interpolation=cv2.INTER_NEAREST,
        )

        mask = torch.from_numpy(mask).unsqueeze(0)

        return {
            "latent": latent_tensor,
            "clean": clean_tensor,
            "mask": mask,
            "id": image_id,
        }


# ============================================================
# IMAGE GRADIENT
# ============================================================

def gradient_x(x):
    return x[:, :, :, 1:] - x[:, :, :, :-1]


def gradient_y(x):
    return x[:, :, 1:, :] - x[:, :, :-1, :]


def gradient_loss(pred, target):
    gx = F.l1_loss(
        gradient_x(pred),
        gradient_x(target),
    )

    gy = F.l1_loss(
        gradient_y(pred),
        gradient_y(target),
    )

    return gx + gy


# ============================================================
# SMOOTHNESS LOSS
# ============================================================

def smoothness_loss(field):

    dx = field[:, 0:1]
    dy = field[:, 1:2]

    loss_dx = (
        torch.mean(torch.abs(gradient_x(dx)))
        + torch.mean(torch.abs(gradient_y(dx)))
    )

    loss_dy = (
        torch.mean(torch.abs(gradient_x(dy)))
        + torch.mean(torch.abs(gradient_y(dy)))
    )

    return loss_dx + loss_dy


# ============================================================
# WARP
# ============================================================

def warp_image(image, field):

    batch, _, height, width = image.shape

    full_field = F.interpolate(
        field,
        size=(height, width),
        mode="bilinear",
        align_corners=True,
    )

    dx = full_field[:, 0]
    dy = full_field[:, 1]

    yy, xx = torch.meshgrid(
        torch.arange(
            height,
            device=image.device,
            dtype=torch.float32,
        ),
        torch.arange(
            width,
            device=image.device,
            dtype=torch.float32,
        ),
        indexing="ij",
    )

    map_x = xx.unsqueeze(0) + dx
    map_y = yy.unsqueeze(0) + dy

    map_x = 2.0 * map_x / max(width - 1, 1) - 1.0
    map_y = 2.0 * map_y / max(height - 1, 1) - 1.0

    grid = torch.stack(
        [map_x, map_y],
        dim=-1,
    )

    warped = F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    return warped


# ============================================================
# MODEL
# ============================================================

def load_baseline_model():

    model = DDRNet_DIR(dis_const=16)

    checkpoint = torch.load(
        BASELINE_CHECKPOINT,
        map_location="cpu",
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        else:
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Remove DataParallel prefix if present.
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

    result = model.load_state_dict(
        state_dict,
        strict=True,
    )

    print("Baseline checkpoint loaded.")
    print(result)

    return model


# ============================================================
# FREEZING
# ============================================================

def configure_trainable_layers(model):

    # Freeze everything first.
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Fine-tune only the final displacement-producing layers.
    trainable_prefixes = [
        "end_conv3",
        "end_conv4",
        "conv_out",
    ]

    for name, parameter in model.named_parameters():

        if any(
            name.startswith(prefix)
            for prefix in trainable_prefixes
        ):
            parameter.requires_grad = True

    print("\nTrainable parameters:")

    total = 0
    trainable = 0

    for name, parameter in model.named_parameters():

        count = parameter.numel()
        total += count

        if parameter.requires_grad:
            trainable += count
            print("  ", name)

    print(f"\nTotal parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print()


# ============================================================
# ORIGINAL FIELD
# ============================================================

@torch.no_grad()
def get_baseline_field(model, latent, mask):

    was_training = model.training
    model.eval()

    field, _ = model(
        latent,
        mask,
    )

    if was_training:
        model.train()

    return field.detach()


# ============================================================
# TRAINING
# ============================================================

def run_epoch(
    model,
    loader,
    optimizer=None,
):

    training = optimizer is not None

    if training:
        model.train()

        # Frozen BatchNorm layers should stay in evaluation mode.
        for module in model.modules():

            if isinstance(
                module,
                nn.BatchNorm2d,
            ):
                module.eval()

    else:
        model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_grad = 0.0
    total_smooth = 0.0
    total_field = 0.0

    count = 0

    for batch in loader:

        latent = batch["latent"].to(
            DEVICE,
            non_blocking=True,
        )

        clean = batch["clean"].to(
            DEVICE,
            non_blocking=True,
        )

        mask = batch["mask"].to(
            DEVICE,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Get OLD baseline field.
        # ----------------------------------------------------

        with torch.no_grad():

            baseline_field = get_baseline_field(
                model,
                latent,
                mask,
            )

        # ----------------------------------------------------
        # New field.
        # ----------------------------------------------------

        field, _ = model(
            latent,
            mask,
        )

        # Conservative safety limit.
        field = torch.clamp(
            field,
            min=-60.0,
            max=60.0,
        )

        # ----------------------------------------------------
        # Rectification.
        #
        # IMPORTANT:
        # map = coordinate + displacement
        # ----------------------------------------------------

        rectified = warp_image(
            latent,
            field,
        )

        # ----------------------------------------------------
        # Losses.
        # ----------------------------------------------------

        recon_loss = F.smooth_l1_loss(
            rectified,
            clean,
        )

        grad_loss = gradient_loss(
            rectified,
            clean,
        )

        smooth_loss = smoothness_loss(
            field,
        )

        field_loss = F.smooth_l1_loss(
            field,
            baseline_field,
        )

        loss = (
            recon_loss
            + LAMBDA_GRAD * grad_loss
            + LAMBDA_SMOOTH * smooth_loss
            + LAMBDA_FIELD * field_loss
        )

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

        batch_size = latent.size(0)

        total_loss += loss.item() * batch_size
        total_recon += recon_loss.item() * batch_size
        total_grad += grad_loss.item() * batch_size
        total_smooth += smooth_loss.item() * batch_size
        total_field += field_loss.item() * batch_size

        count += batch_size

    return {
        "loss": total_loss / count,
        "recon": total_recon / count,
        "grad": total_grad / count,
        "smooth": total_smooth / count,
        "field": total_field / count,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("DDRNet FVC CONSERVATIVE FINE-TUNING")
    print("=" * 70)

    print("Device:", DEVICE)
    print("Image size:", IMAGE_SIZE)
    print("Mask size:", MASK_SIZE)
    print("Learning rate:", LR)
    print()

    # --------------------------------------------------------
    # Find IDs.
    # --------------------------------------------------------

    latent_ids = {
        p.stem
        for p in LATENT_DIR.glob("*.png")
    }

    clean_ids = {
        p.stem
        for p in CLEAN_DIR.glob("*.png")
    }

    ids = sorted(
        latent_ids & clean_ids,
        key=lambda x: int(x),
    )

    print("Latent images:", len(latent_ids))
    print("Clean images :", len(clean_ids))
    print("Paired images:", len(ids))

    # --------------------------------------------------------
    # Recreate the verified 222/48/47 split.
    # --------------------------------------------------------

    rng = random.Random(SEED)

    shuffled_ids = ids.copy()
    rng.shuffle(shuffled_ids)

    train_ids = shuffled_ids[:222]
    val_ids = shuffled_ids[222:270]
    test_ids = shuffled_ids[270:317]

    split = {
        "seed": SEED,
        "train": train_ids,
        "validation": val_ids,
        "test": test_ids,
    }

    with open(
        OUTPUT_DIR / "split.json",
        "w",
    ) as f:

        json.dump(
            split,
            f,
            indent=2,
        )

    print()
    print("Train:", len(train_ids))
    print("Validation:", len(val_ids))
    print("Test:", len(test_ids))
    print()

    # --------------------------------------------------------
    # Datasets.
    # --------------------------------------------------------

    train_dataset = FVCFinetuneDataset(
        train_ids
    )

    val_dataset = FVCFinetuneDataset(
        val_ids
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # --------------------------------------------------------
    # Model.
    # --------------------------------------------------------

    model = load_baseline_model()

    configure_trainable_layers(model)

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Optimizer.
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training.
    # --------------------------------------------------------

    history = []

    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0

    for epoch in range(
        1,
        MAX_EPOCHS + 1,
    ):

        print(
            f"\nEpoch {epoch}/{MAX_EPOCHS}"
        )

        train_metrics = run_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            None,
        )

        print(
            "Train:"
            f" loss={train_metrics['loss']:.6f}"
            f" recon={train_metrics['recon']:.6f}"
            f" grad={train_metrics['grad']:.6f}"
            f" field={train_metrics['field']:.6f}"
        )

        print(
            "Val  :"
            f" loss={val_metrics['loss']:.6f}"
            f" recon={val_metrics['recon']:.6f}"
            f" grad={val_metrics['grad']:.6f}"
            f" field={val_metrics['field']:.6f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
            }
        )

        # ----------------------------------------------------
        # Save best.
        # ----------------------------------------------------

        if val_metrics["loss"] < best_val_loss:

            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": best_val_loss,
                    "image_size": IMAGE_SIZE,
                    "mask_size": MASK_SIZE,
                    "dis_const": 16,
                },
                BEST_CHECKPOINT,
            )

            print(
                f"  -> Saved best checkpoint "
                f"(epoch {epoch})"
            )

        else:

            patience_counter += 1

            print(
                f"  -> No improvement "
                f"({patience_counter}/{PATIENCE})"
            )

            if patience_counter >= PATIENCE:

                print(
                    "\nEarly stopping."
                )

                break

        with open(
            HISTORY_FILE,
            "w",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        "Best validation loss:",
        best_val_loss,
    )

    print(
        "Checkpoint:",
        BEST_CHECKPOINT,
    )

    print(
        "History:",
        HISTORY_FILE,
    )


if __name__ == "__main__":
    main()