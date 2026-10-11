from pathlib import Path
import cv2
import authentication as auth


PROJECT_ROOT = Path(__file__).resolve().parent

LATENT_DIR = (
    PROJECT_ROOT
    / "data"
    / "Latent_fvc"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "ddrnet_inspection2"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# New FVC-trained DDRNet checkpoint
auth.DDRNET_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "ddrnet_fvc_finetuned"
    / "best_model.pth"
)

auth.DDRNET = None


# Representative cases:
# 215: Rank 1 -> 172
# 14 : Rank 27 -> 243
# 41 : Rank 248 -> 21
# 175: Rank 180 -> 26

CASE_IDS = [
    215,
    14,
    41,
    175,
]


for fingerprint_id in CASE_IDS:

    input_path = (
        LATENT_DIR
        / f"{fingerprint_id}.png"
    )

    image = cv2.imread(
        str(input_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        print(
            f"ERROR: Could not load "
            f"{input_path}"
        )
        continue

    print(
        f"Processing ID {fingerprint_id}..."
    )

    rectified = (
        auth.rectify_with_ddrnet(
            image
        )
    )

    original_path = (
        OUTPUT_DIR
        / f"{fingerprint_id}_original.png"
    )

    rectified_path = (
        OUTPUT_DIR
        / f"{fingerprint_id}_rectified.png"
    )

    cv2.imwrite(
        str(original_path),
        image
    )

    cv2.imwrite(
        str(rectified_path),
        rectified
    )

    print(
        f"  Original : {original_path}"
    )

    print(
        f"  Rectified: {rectified_path}"
    )


print()
print(
    "Inspection complete."
)

print(
    f"Outputs saved to:\n{OUTPUT_DIR}"
)
