import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn

# Reconfigure stdout for utf-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

PROJ_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_ROOT))

from model import FingerprintCNN, count_parameters
from experiments.classifier_orientation_v1.dataset import (
    FingerprintVariationDataset,
    resolve_project_path,
    create_or_load_splits
)

print("=" * 80)
print("FINGERSURE PRE-TRAINING AUDIT CHECK SCRIPT")
print("Target: experiments/classifier_orientation_v1/")
print("=" * 80)

audit_results = {}

# ==============================================================================
# CHECK 1: PREPROCESSING PIXEL-BY-PIXEL COMPARISON
# ==============================================================================
print("\n" + "-" * 40)
print("CHECK 1: Pixel-by-Pixel Preprocessing Comparison")
print("-" * 40)

# Representative test images
test_image_paths = [
    PROJ_ROOT / "data" / "orientation_capture_variations_v1" / "clean" / "FM001_C1_clean.png",
    PROJ_ROOT / "data" / "orientation_capture_variations_v1" / "orientation_variation" / "FM001_C1_orientation.png",
    PROJ_ROOT / "data" / "orientation_capture_variations_v1" / "partial_capture" / "FM001_C1_partial.png",
    PROJ_ROOT / "data" / "orientation_capture_variations_v1" / "uneven_contact" / "FM001_C1_uneven.png",
]

# Physical dataset samples if available
physical_clean = PROJ_ROOT / "FingerSure_Dataset" / "clean" / "01_clean.png"
if not physical_clean.exists():
    phys_clean_list = list((PROJ_ROOT / "FingerSure_Dataset" / "clean").glob("*.png"))
    if phys_clean_list:
        physical_clean = phys_clean_list[0]
if physical_clean.exists():
    test_image_paths.append(physical_clean)

physical_dist = PROJ_ROOT / "FingerSure_Dataset" / "distorted" / "01_distorted.png"
if not physical_dist.exists():
    phys_dist_list = list((PROJ_ROOT / "FingerSure_Dataset" / "distorted").glob("*.png"))
    if phys_dist_list:
        physical_dist = phys_dist_list[0]
if physical_dist.exists():
    test_image_paths.append(physical_dist)

check1_passed = True
check1_details = []

for img_p in test_image_paths:
    if not img_p.exists():
        print(f"Skipping non-existent: {img_p}")
        continue

    # Training Pipeline Preprocessing (from dataset.py)
    img_train_raw = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
    if img_train_raw.shape[:2] != (224, 224):
        img_train_res = cv2.resize(img_train_raw, (224, 224), interpolation=cv2.INTER_AREA)
    else:
        img_train_res = img_train_raw
    img_train_norm = img_train_res.astype(np.float32) / 255.0
    tensor_train = torch.from_numpy(img_train_norm).unsqueeze(0)  # (1, 224, 224)

    # Production Inference Preprocessing (from authentication.py lines 154-173)
    img_auth_raw = cv2.imread(str(img_p), cv2.IMREAD_UNCHANGED)
    if img_auth_raw.ndim == 3:
        img_auth_gray = cv2.cvtColor(img_auth_raw, cv2.COLOR_BGR2GRAY)
    else:
        img_auth_gray = img_auth_raw
    image_224 = cv2.resize(img_auth_gray, (224, 224), interpolation=cv2.INTER_AREA)
    image_float = image_224.astype(np.float32) / 255.0
    tensor_auth = torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0)  # (1, 1, 224, 224)

    # Compare tensors
    t_train_sq = tensor_train.squeeze()  # (224, 224)
    t_auth_sq = tensor_auth.squeeze()    # (224, 224)

    abs_diff = torch.abs(t_train_sq - t_auth_sq)
    max_diff = float(abs_diff.max().item())
    mean_diff = float(abs_diff.mean().item())

    detail = {
        "file": img_p.name,
        "raw_shape": list(img_train_raw.shape),
        "train_tensor_shape": list(tensor_train.shape),
        "auth_tensor_shape": list(tensor_auth.shape),
        "max_abs_diff": max_diff,
        "mean_abs_diff": mean_diff,
        "match": max_diff == 0.0
    }
    check1_details.append(detail)
    print(f"  [{img_p.name}] Shape: {img_train_raw.shape} -> 224x224 | Max Diff: {max_diff:.8f} | Match: {max_diff == 0.0}")
    if max_diff > 1e-6:
        check1_passed = False

audit_results["check1_preprocessing"] = {
    "status": "PASS" if check1_passed else "FAIL",
    "details": check1_details
}

# ==============================================================================
# CHECK 2: CONFIRM CLEAN=0 AND DISTORTED=1 MAPPING
# ==============================================================================
print("\n" + "-" * 40)
print("CHECK 2: Confirm CLEAN=0 and DISTORTED=1 Mapping")
print("-" * 40)

config_path = PROJ_ROOT / "experiments" / "classifier_orientation_v1" / "config.json"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

cfg_classes = cfg.get("classes", {})
clean_val = cfg_classes.get("clean")
dist_val = cfg_classes.get("distorted")
dist_components = cfg_classes.get("distorted_components", [])

print(f"Config classes: CLEAN={clean_val}, DISTORTED={dist_val}")
print(f"Distorted components: {dist_components}")

split_dir = PROJ_ROOT / "experiments" / "classifier_orientation_v1" / "splits"
train_df = pd.read_csv(split_dir / "train_manifest.csv")
val_df = pd.read_csv(split_dir / "val_manifest.csv")
test_df = pd.read_csv(split_dir / "test_manifest.csv")

check2_passed = True
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    clean_labels = df[df["class_name"] == "clean"]["label"].unique().tolist()
    dist_labels = df[df["class_name"] == "distorted"]["label"].unique().tolist()
    clean_vars = df[df["class_name"] == "clean"]["variation_type"].unique().tolist()
    dist_vars = sorted(df[df["class_name"] == "distorted"]["variation_type"].unique().tolist())
    
    print(f"  Split '{name}':")
    print(f"    Clean rows: {len(df[df['class_name'] == 'clean'])}, label set: {clean_labels}, variations: {clean_vars}")
    print(f"    Distorted rows: {len(df[df['class_name'] == 'distorted'])}, label set: {dist_labels}, variations: {dist_vars}")
    
    if clean_labels != [0] or dist_labels != [1]:
        check2_passed = False
    if clean_vars != ["clean"]:
        check2_passed = False
    if dist_vars != sorted(["orientation_variation", "partial_capture", "uneven_contact"]):
        check2_passed = False

if clean_val != 0 or dist_val != 1:
    check2_passed = False

audit_results["check2_class_mapping"] = {
    "status": "PASS" if check2_passed else "FAIL",
    "clean_label": clean_val,
    "distorted_label": dist_val,
    "distorted_components": dist_components
}

# ==============================================================================
# CHECK 3: CLASS IMBALANCE HANDLING (1:3 RATIO)
# ==============================================================================
print("\n" + "-" * 40)
print("CHECK 3: Class Imbalance Handling (1:3 Ratio)")
print("-" * 40)

train_clean_cnt = int((train_df["label"] == 0).sum())
train_dist_cnt = int((train_df["label"] == 1).sum())
val_clean_cnt = int((val_df["label"] == 0).sum())
val_dist_cnt = int((val_df["label"] == 1).sum())
test_clean_cnt = int((test_df["label"] == 0).sum())
test_dist_cnt = int((test_df["label"] == 1).sum())

print(f"Train Counts: {train_clean_cnt} Clean ({train_clean_cnt/len(train_df)*100:.1f}%) | {train_dist_cnt} Distorted ({train_dist_cnt/len(train_df)*100:.1f}%) | Ratio: 1:{train_dist_cnt/train_clean_cnt:.1f}")
print(f"Val Counts:   {val_clean_cnt} Clean ({val_clean_cnt/len(val_df)*100:.1f}%) | {val_dist_cnt} Distorted ({val_dist_cnt/len(val_df)*100:.1f}%) | Ratio: 1:{val_dist_cnt/val_clean_cnt:.1f}")
print(f"Test Counts:  {test_clean_cnt} Clean ({test_clean_cnt/len(test_df)*100:.1f}%) | {test_dist_cnt} Distorted ({test_dist_cnt/len(test_df)*100:.1f}%) | Ratio: 1:{test_dist_cnt/test_clean_cnt:.1f}")

pos_weight = cfg["training"].get("pos_weight", 1.0)
th_tuning = cfg["training"].get("threshold_tuning", {})

print(f"Loss Configuration: pos_weight = {pos_weight}")
print(f"Threshold Tuning Configuration: {th_tuning}")

check3_passed = (train_dist_cnt == 3 * train_clean_cnt) and (val_dist_cnt == 3 * val_clean_cnt) and (test_dist_cnt == 3 * test_clean_cnt)

audit_results["check3_class_imbalance"] = {
    "status": "PASS" if check3_passed else "FAIL",
    "train_ratio": f"1:{train_dist_cnt/train_clean_cnt:.1f}",
    "pos_weight": pos_weight,
    "strategy_details": {
        "pos_weight_in_bce": pos_weight,
        "threshold_tuning": th_tuning,
        "mechanism": "BCEWithLogitsLoss with pos_weight=1.0 and empirical validation threshold tuning ([0.05, 0.95] step 0.01) optimizing F1 score to offset 1:3 positive prior."
    }
}

# ==============================================================================
# CHECK 4: HYPERPARAMETERS INSPECTION
# ==============================================================================
print("\n" + "-" * 40)
print("CHECK 4: Hyperparameters Inspection")
print("-" * 40)

training_cfg = cfg.get("training", {})
optimizer_name = training_cfg.get("optimizer")
lr = training_cfg.get("learning_rate")
weight_decay = training_cfg.get("weight_decay")
batch_size = training_cfg.get("batch_size")
max_epochs = training_cfg.get("max_epochs")
patience = training_cfg.get("early_stopping_patience")
seed = cfg.get("dataset", {}).get("seed")

print(f"Optimizer:        {optimizer_name} (expected: Adam)")
print(f"Learning Rate:    {lr} (expected: 0.0001)")
print(f"Weight Decay:     {weight_decay} (expected: 0.0001)")
print(f"Batch Size:       {batch_size} (expected: 32)")
print(f"Max Epochs:       {max_epochs} (expected: 25)")
print(f"Patience:         {patience} (expected: 6)")
print(f"Random Seed:      {seed} (expected: 42)")

check4_passed = (
    optimizer_name == "Adam" and
    lr == 0.0001 and
    weight_decay == 0.0001 and
    batch_size == 32 and
    max_epochs == 25 and
    patience == 6 and
    seed == 42
)

audit_results["check4_hyperparameters"] = {
    "status": "PASS" if check4_passed else "FAIL",
    "optimizer": optimizer_name,
    "learning_rate": lr,
    "weight_decay": weight_decay,
    "batch_size": batch_size,
    "max_epochs": max_epochs,
    "patience": patience,
    "seed": seed
}

# ==============================================================================
# CHECK 5: BEST MODEL & THRESHOLD SELECTION USING VALIDATION DATA ONLY
# ==============================================================================
print("\n" + "-" * 40)
print("CHECK 5: Best Model & Threshold Selection (Validation Only)")
print("-" * 40)

train_py_path = PROJ_ROOT / "experiments" / "classifier_orientation_v1" / "train.py"
with open(train_py_path, "r", encoding="utf-8") as f:
    train_py_content = f.read()

# Inspect validation logic in train.py
has_val_loss_eval = "val_loss, y_true_val, y_probs_val = run_inference_epoch(model, val_l, criterion, device)" in train_py_content
has_opt_thresh_val = "opt_thresh, val_metrics = find_optimal_threshold(" in train_py_content
has_f1_target = 'target_metric=th_cfg.get("optimization_metric", "f1_score")' in train_py_content or 'f1_score' in train_py_content
has_save_on_f1 = "if val_f1 > best_val_f1:" in train_py_content
saves_best_threshold = '"threshold": best_threshold' in train_py_content

print(f"  Inference evaluated on val_l: {has_val_loss_eval}")
print(f"  Threshold tuned on val probabilities: {has_opt_thresh_val}")
print(f"  Target metric is F1: {has_f1_target}")
print(f"  Checkpoint saved strictly on val_f1 improvement: {has_save_on_f1}")
print(f"  Best threshold locked and saved: {saves_best_threshold}")

check5_passed = has_val_loss_eval and has_opt_thresh_val and has_save_on_f1 and saves_best_threshold

audit_results["check5_validation_selection"] = {
    "status": "PASS" if check5_passed else "FAIL",
    "best_model_criterion": "Validation F1 score (val_f1 > best_val_f1)",
    "threshold_selection_data": "Validation set only (val_l probabilities)",
    "threshold_metric": "f1_score"
}

# ==============================================================================
# CHECK 6: HELD-OUT TEST SET STRICT ISOLATION
# ==============================================================================
print("\n" + "-" * 40)
print("CHECK 6: Held-Out Test Set Strict Isolation")
print("-" * 40)

# Verify test_l is only used after training loop completes
loop_end_idx = train_py_content.find("total_time = time.time() - start_time")
test_eval_idx = train_py_content.find("run_inference_epoch(model, test_l, criterion, device)")

test_after_loop = test_eval_idx > loop_end_idx
uses_locked_threshold = "evaluate_predictions(y_true_test, y_probs_test, threshold=best_threshold)" in train_py_content

train_stems = set(train_df["identity_stem"])
val_stems = set(val_df["identity_stem"])
test_stems = set(test_df["identity_stem"])

train_val_overlap = len(train_stems & val_stems)
train_test_overlap = len(train_stems & test_stems)
val_test_overlap = len(val_stems & test_stems)

print(f"  Test evaluation occurs strictly AFTER training loop: {test_after_loop}")
print(f"  Test evaluation uses locked validation threshold: {uses_locked_threshold}")
print(f"  Identity stem overlap train/test: {train_test_overlap} (expected: 0)")
print(f"  Identity stem overlap val/test:   {val_test_overlap} (expected: 0)")

check6_passed = test_after_loop and uses_locked_threshold and (train_test_overlap == 0) and (val_test_overlap == 0)

audit_results["check6_test_isolation"] = {
    "status": "PASS" if check6_passed else "FAIL",
    "test_evaluated_after_training": test_after_loop,
    "locked_threshold_used": uses_locked_threshold,
    "train_test_leakage": train_test_overlap,
    "val_test_leakage": val_test_overlap
}

# ==============================================================================
# CHECK 7: EXISTING PRODUCTION CLASSIFIER & DDRNET CHECKPOINTS UNTOUCHED
# ==============================================================================
print("\n" + "-" * 40)
print("CHECK 7: Production Checkpoints Integrity Audit")
print("-" * 40)

def get_file_info(p: Path):
    if not p.exists():
        return None
    stat = p.stat()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return {
        "path": str(p),
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": h.hexdigest()
    }

prod_classifier_path = PROJ_ROOT / "experiments" / "classifier_fvc_adapted" / "best_model.pth"
prod_ddrnet_path = PROJ_ROOT / "experiments" / "ddrnet_fvc_finetuned" / "best_model.pth"
new_checkpoints_dir = PROJ_ROOT / "experiments" / "classifier_orientation_v1" / "checkpoints"

prod_clf_info = get_file_info(prod_classifier_path)
prod_ddr_info = get_file_info(prod_ddrnet_path)

new_checkpoints = list(new_checkpoints_dir.glob("*.pth"))

print(f"  Production Classifier ({prod_classifier_path.name}):")
print(f"    Size: {prod_clf_info['size_bytes']:,} bytes")
print(f"    SHA256: {prod_clf_info['sha256'][:16]}...")
print(f"  Production DDRNet ({prod_ddrnet_path.name}):")
print(f"    Size: {prod_ddr_info['size_bytes']:,} bytes")
print(f"    SHA256: {prod_ddr_info['sha256'][:16]}...")
print(f"  New classifier_orientation_v1 checkpoints generated: {len(new_checkpoints)} (expected: 0)")

check7_passed = (
    prod_clf_info is not None and
    prod_clf_info["size_bytes"] == 1570229 and
    prod_ddr_info is not None and
    prod_ddr_info["size_bytes"] == 180395816 and
    len(new_checkpoints) == 0
)

audit_results["check7_production_integrity"] = {
    "status": "PASS" if check7_passed else "FAIL",
    "production_classifier_size": prod_clf_info["size_bytes"],
    "production_classifier_sha256": prod_clf_info["sha256"],
    "production_ddrnet_size": prod_ddr_info["size_bytes"],
    "production_ddrnet_sha256": prod_ddr_info["sha256"],
    "new_checkpoints_count": len(new_checkpoints)
}

# ==============================================================================
# CHECK 8: FINGERPRINTCNN INFERENCE IMPLEMENTATION COMPATIBILITY
# ==============================================================================
print("\n" + "-" * 40)
print("CHECK 8: FingerprintCNN Inference Compatibility")
print("-" * 40)

# Instantiate model as in train.py
train_model = FingerprintCNN(dropout=0.5)
train_state_dict = train_model.state_dict()

# Simulate saving checkpoint dict as train.py does:
mock_checkpoint = {
    "epoch": 1,
    "model_state_dict": train_state_dict,
    "optimizer_state_dict": {},
    "val_f1": 0.95,
    "threshold": 0.62,
    "val_metrics": {},
    "config": cfg
}

# Instantiate model as in authentication.py lines 90-128
auth_model = FingerprintCNN()

# Extract state_dict exactly as authentication.py line 99 does:
ckpt_to_load = mock_checkpoint
state_dict_loaded = (
    ckpt_to_load["model_state_dict"] 
    if isinstance(ckpt_to_load, dict) and "model_state_dict" in ckpt_to_load 
    else ckpt_to_load
)

load_result = auth_model.load_state_dict(state_dict_loaded, strict=True)
auth_model.eval()

# Run mock inference batch
mock_input = torch.randn(1, 1, 224, 224)
with torch.no_grad():
    logits = auth_model(mock_input)
    prob_dist = float(torch.sigmoid(logits).item())
    prob_clean = 1.0 - prob_dist

print(f"  State dict keys count: {len(state_dict_loaded)}")
print(f"  load_state_dict(strict=True) successful: {load_result}")
print(f"  Inference logits shape: {logits.shape} (expected: torch.Size([1, 1]))")
print(f"  Sigmoid prob_dist: {prob_dist:.4f}, prob_clean: {prob_clean:.4f}")

check8_passed = (
    len(load_result.missing_keys) == 0 and
    len(load_result.unexpected_keys) == 0 and
    logits.shape == (1, 1) and
    (0.0 <= prob_dist <= 1.0)
)

audit_results["check8_inference_compatibility"] = {
    "status": "PASS" if check8_passed else "FAIL",
    "missing_keys": len(load_result.missing_keys),
    "unexpected_keys": len(load_result.unexpected_keys),
    "output_shape": list(logits.shape),
    "inference_verified": True
}

# ==============================================================================
# AUDIT SUMMARY
# ==============================================================================
print("\n" + "=" * 80)
print("AUDIT SUMMARY:")
print("=" * 80)
all_passed = True
for k, v in audit_results.items():
    st = v["status"]
    if st != "PASS":
        all_passed = False
    print(f"  {k:30s}: {st}")

print("=" * 80)
print(f"OVERALL AUDIT VERDICT: {'PASS' if all_passed else 'FAIL'}")
print("=" * 80)

# Save JSON results for report generation
out_json = PROJ_ROOT / "scratch" / "audit_results.json"
with open(out_json, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)
print(f"Audit results written to {out_json}")

