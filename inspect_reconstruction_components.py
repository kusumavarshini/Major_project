from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import zoom

import authentication as auth
from DDRNet.tools.fp_segmentation import segmentation_coherence


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

LATENT_PATH = PROJECT_ROOT / "data" / "Latent_fvc" / "215.png"
CLEAN_PATH = PROJECT_ROOT / "data" / "clean_fvc_new" / "215.png"

OUTPUT_DIR = PROJECT_ROOT / "experiments" / "ddrnet_reconstruction_inspection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SIZE = 512
DEVICE = torch.device("cpu")


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    str(LATENT_PATH),
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    raise FileNotFoundError(LATENT_PATH)

original_h, original_w = image.shape

gray = cv2.resize(
    image,
    (MODEL_SIZE, MODEL_SIZE),
    interpolation=cv2.INTER_AREA
)

# Same normalization as training
normalized = (
    255.0 - gray.astype(np.float32)
) / 255.0

x = normalized[None, None, :, :]


# ============================================================
# SEGMENTATION MASK
# ============================================================

mask = segmentation_coherence(
    gray,
    win_size=16,
    stride=8
)

mask = np.asarray(
    mask,
    dtype=np.float32
)

if mask.shape != (32, 32):
    mask = cv2.resize(
        mask,
        (32, 32),
        interpolation=cv2.INTER_NEAREST
    )

mask = (mask > 0).astype(np.float32)

mask_tensor = mask[None, None, :, :]


# ============================================================
# LOAD FULL RECONSTRUCTION MODEL
# ============================================================

model = auth.load_ddrnet()
model.eval()


# ============================================================
# RUN MODEL
# ============================================================

with torch.no_grad():

    field, direction, warped, reconstructed = model(
    torch.from_numpy(x).float().to(DEVICE),
    torch.from_numpy(mask_tensor).float().to(DEVICE)
    )

    full_field = torch.nn.functional.interpolate(
    field,
    size=(MODEL_SIZE, MODEL_SIZE),
    mode="bilinear",
    align_corners=True
    )


# ============================================================
# CONVERT WARPED BACK TO NORMAL GRAYSCALE
# ============================================================

warped_np = warped[0, 0].cpu().numpy()

warped_u8 = (
    255.0 * (1.0 - warped_np)
)

warped_u8 = np.clip(
    warped_u8,
    0,
    255
).astype(np.uint8)


# ============================================================
# CONVERT RECONSTRUCTED BACK TO NORMAL GRAYSCALE
# ============================================================

reconstructed_np = reconstructed[0, 0].cpu().numpy()

reconstructed_u8 = (
    255.0 * (1.0 - reconstructed_np)
)

reconstructed_u8 = np.clip(
    reconstructed_u8,
    0,
    255
).astype(np.uint8)


# ============================================================
# RESIZE TO ORIGINAL SIZE
# ============================================================

if (original_w, original_h) != (MODEL_SIZE, MODEL_SIZE):

    warped_u8 = cv2.resize(
        warped_u8,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR
    )

    reconstructed_u8 = cv2.resize(
        reconstructed_u8,
        (original_w, original_h),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# SAVE RESULTS
# ============================================================

cv2.imwrite(
    str(OUTPUT_DIR / "215_original.png"),
    image
)

cv2.imwrite(
    str(OUTPUT_DIR / "215_warp_only.png"),
    warped_u8
)

cv2.imwrite(
    str(OUTPUT_DIR / "215_reconstructed.png"),
    reconstructed_u8
)

if CLEAN_PATH.exists():

    clean = cv2.imread(
        str(CLEAN_PATH),
        cv2.IMREAD_GRAYSCALE
    )

    if clean is not None:

        cv2.imwrite(
            str(OUTPUT_DIR / "215_clean_ground_truth.png"),
            clean
        )


# ============================================================
# PRINT FIELD STATISTICS
# ============================================================

field_np = full_field[0].cpu().numpy()

magnitude = np.sqrt(
    field_np[0] ** 2 +
    field_np[1] ** 2
)

print()
print("=" * 70)
print("ID 215 RECONSTRUCTION INSPECTION")
print("=" * 70)

print("Original size :", image.shape)
print("Model size    :", (MODEL_SIZE, MODEL_SIZE))

print()
print("DISPLACEMENT FIELD")
print("Mean magnitude:", float(magnitude.mean()))
print("Max magnitude :", float(magnitude.max()))

print()
print("OUTPUTS SAVED TO:")
print(OUTPUT_DIR)

print()
print("Files:")
print("  215_original.png")
print("  215_warp_only.png")
print("  215_reconstructed.png")
print("  215_clean_ground_truth.png")
