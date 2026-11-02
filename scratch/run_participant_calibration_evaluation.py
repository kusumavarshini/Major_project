import os
import sys
import json
import time
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
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 80)
print("FINGERSURE PARTICIPANT-LEVEL CALIBRATION & VALIDATION (V2)")
print("=" * 80)

# ==============================================================================
# 1. SETUP DIRECTORIES & MODEL
# ==============================================================================
exp2_dir = PROJ_ROOT / "experiments" / "classifier_orientation_v2"
target_dir = exp2_dir / "participant_calibration_evaluation"
target_dir.mkdir(parents=True, exist_ok=True)

model_path = exp2_dir / "best_model.pth"
if not model_path.exists():
    downloaded_pth = exp2_dir / "best_model (4).pth"
    if downloaded_pth.exists():
        import shutil
        shutil.copy2(downloaded_pth, model_path)
    else:
        raise FileNotFoundError(f"Model checkpoint missing: {model_path}")

print(f"[MODEL] Loading existing V2 checkpoint: {model_path.name}")
ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
threshold_ref = float(ckpt.get("threshold", 0.50))

model = FingerprintCNN(dropout=0.5)
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"[MODEL] Loaded FingerprintCNN ({count_parameters(model):,} parameters) in EVAL mode.")

# ==============================================================================
# 2. PHYSICAL DATASET DISCOVERY & AUDIT
# ==============================================================================
ds_dir = PROJ_ROOT / "FingerSure_Dataset"
clean_dir = ds_dir / "clean"
dist_dir = ds_dir / "distorted"
meta_path = ds_dir / "metadata.csv"

meta_df = pd.read_csv(meta_path) if meta_path.exists() else None

clean_files = sorted(list(clean_dir.glob("*.png")))
dist_files = sorted(list(dist_dir.glob("*.png")))

assert len(clean_files) == 42, f"Expected 42 clean images, found {len(clean_files)}"
assert len(dist_files) == 42, f"Expected 42 distorted images, found {len(dist_files)}"

def parse_filename(name):
    # e.g. 'A_LEFT_THUMB_clean.png' -> participant='A', finger='LEFT_THUMB'
    base = name.replace("_clean.png", "").replace("_distorted.png", "")
    parts = base.split("_")
    finger = f"{parts[-2]}_{parts[-1]}"
    participant = "_".join(parts[:-2])
    return participant, finger

records = []
all_image_paths = clean_files + dist_files

print(f"[DATASET AUDIT] Scanning {len(all_image_paths)} images across 21 participants...")

with torch.no_grad():
    for p in all_image_paths:
        part, finger = parse_filename(p.name)
        is_clean = "_clean.png" in p.name
        gt_label = 0 if is_clean else 1
        gt_class = "CLEAN" if is_clean else "DISTORTED"
        
        raw_img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        h, w = raw_img.shape[:2]
        
        # Image characterization
        mean_int = float(raw_img.mean())
        std_int = float(raw_img.std())
        lap_var = float(cv2.Laplacian(raw_img, cv2.CV_64F).var())
        
        # Non-background foreground mask (optical SecuGen background is white ~220-255)
        fg_mask = (raw_img < 220).astype(np.uint8)
        fg_cov = float(np.sum(fg_mask) / (h * w) * 100.0)
        
        # Moments for orientation
        moments = cv2.moments(fg_mask)
        if moments["mu20"] != moments["mu02"]:
            angle_rad = 0.5 * np.arctan2(2 * moments["mu11"], (moments["mu20"] - moments["mu02"]))
            angle_deg = float(np.degrees(angle_rad))
        else:
            angle_deg = 0.0
            
        # Variation classification
        if is_clean:
            var_type = "Normal Relaxed"
        else:
            if fg_cov < 28.0:
                var_type = "Partial Contact"
            elif mean_int > 195.0 or std_int < 55.0:
                var_type = "Uneven / Light Contact"
            elif abs(angle_deg) > 20.0:
                var_type = "Rotated / Sideways"
            else:
                var_type = "Tilted / Shear Distortion"

        # Preprocessing: cv2.resize((224, 224), INTER_AREA) -> float32 / 255.0 -> tensor
        img_224 = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_AREA)
        img_norm = img_224.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
        
        logit = model(tensor)
        p_dist = float(torch.sigmoid(logit).item())
        p_clean = 1.0 - p_dist
        
        pred_050 = "DISTORTED" if p_dist >= 0.50 else "CLEAN"
        status_050 = "CORRECT" if pred_050 == gt_class else "INCORRECT"
        
        records.append({
            "filename": p.name,
            "participant_id": part,
            "finger": finger,
            "actual_label": gt_label,
            "actual_class": gt_class,
            "prob_distorted": round(p_dist, 6),
            "prob_clean": round(p_clean, 6),
            "prediction_at_threshold_050": pred_050,
            "status_at_threshold_050": status_050,
            "variation_type": var_type,
            "image_width": w,
            "image_height": h,
            "image_format": "PNG",
            "mean_intensity": round(mean_int, 2),
            "std_intensity": round(std_int, 2),
            "sharpness_laplacian": round(lap_var, 2),
            "fg_coverage_pct": round(fg_cov, 2),
            "estimated_angle_deg": round(angle_deg, 2)
        })

df_all = pd.DataFrame(records)

# Save predictions.csv
pred_csv_path = target_dir / "predictions.csv"
df_all[[
    "filename",
    "participant_id",
    "finger",
    "actual_label",
    "actual_class",
    "prob_distorted",
    "prob_clean",
    "prediction_at_threshold_050",
    "status_at_threshold_050",
    "variation_type"
]].to_csv(pred_csv_path, index=False)
print(f"[OUTPUT] Saved predictions.csv to {pred_csv_path}")

# ==============================================================================
# 3. PARTICIPANT-LEVEL STRATIFIED GROUP K-FOLD CROSS-VALIDATION
# ==============================================================================
print("\n" + "=" * 60)
print("RUNNING 7-FOLD STRATIFIED GROUP K-FOLD CV (3 PARTICIPANTS/FOLD)")
print("=" * 60)

participants = sorted(df_all["participant_id"].unique())
print(f"Total Unique Participants: {len(participants)}")

# Setup StratifiedGroupKFold
groups = df_all["participant_id"].to_numpy()
y_true_all = df_all["actual_label"].to_numpy()
probs_all = df_all["prob_distorted"].to_numpy()
X_dummy = np.arange(len(df_all))

sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)

cv_rows = []
oof_preds = np.zeros(len(df_all), dtype=int)
oof_thresh = np.zeros(len(df_all), dtype=float)

def compute_metrics(y_t, y_p, probs):
    acc = accuracy_score(y_t, y_p)
    prec = precision_score(y_t, y_p, zero_division=0)
    rec = recall_score(y_t, y_p, zero_division=0)
    f1 = f1_score(y_t, y_p, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (rec + spec)
    try:
        auc = roc_auc_score(y_t, probs)
    except Exception:
        auc = 0.5
    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "specificity": spec,
        "balanced_accuracy": bal_acc,
        "roc_auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_dummy, y_true_all, groups)):
    fold_num = fold + 1
    cal_participants = sorted(list(set(groups[train_idx])))
    val_participants = sorted(list(set(groups[val_idx])))
    
    # Assert zero overlap
    assert len(set(cal_participants) & set(val_participants)) == 0, "DATA LEAKAGE IN FOLD!"
    
    y_cal = y_true_all[train_idx]
    probs_cal = probs_all[train_idx]
    
    y_val = y_true_all[val_idx]
    probs_val = probs_all[val_idx]
    
    # Grid search threshold strictly on calibration participants (maximize F1)
    best_th = 0.50
    best_cal_f1 = -1.0
    for t in np.arange(0.005, 0.40, 0.001):
        t_val = round(float(t), 4)
        pred_cal = (probs_cal >= t_val).astype(int)
        score = f1_score(y_cal, pred_cal, zero_division=0)
        if score > best_cal_f1:
            best_cal_f1 = score
            best_th = t_val
            
    # Evaluate selected threshold on completely unseen validation participants
    pred_val = (probs_val >= best_th).astype(int)
    oof_preds[val_idx] = pred_val
    oof_thresh[val_idx] = best_th
    
    val_m = compute_metrics(y_val, pred_val, probs_val)
    
    print(f"Fold {fold_num:d}: Cal Th={best_th:.4f} (Cal F1={best_cal_f1:.3f}) | Val: Acc={val_m['accuracy']*100:.1f}%, F1={val_m['f1_score']*100:.1f}%, Rec={val_m['recall']*100:.1f}%, Spec={val_m['specificity']*100:.1f}% | Val Participants={val_participants}")
    
    cv_rows.append({
        "fold": fold_num,
        "calibration_participants_count": len(cal_participants),
        "validation_participants_count": len(val_participants),
        "calibration_participants": ", ".join(cal_participants),
        "validation_participants": ", ".join(val_participants),
        "selected_threshold": best_th,
        "calibration_best_f1": round(best_cal_f1, 4),
        "accuracy": round(val_m["accuracy"], 4),
        "precision": round(val_m["precision"], 4),
        "recall": round(val_m["recall"], 4),
        "f1_score": round(val_m["f1_score"], 4),
        "specificity": round(val_m["specificity"], 4),
        "balanced_accuracy": round(val_m["balanced_accuracy"], 4),
        "roc_auc": round(val_m["roc_auc"], 4),
        "fpr": round(val_m["fpr"], 4),
        "fnr": round(val_m["fnr"], 4),
        "tp": val_m["tp"],
        "fp": val_m["fp"],
        "tn": val_m["tn"],
        "fn": val_m["fn"]
    })

df_cv = pd.DataFrame(cv_rows)
cv_csv_path = target_dir / "cv_fold_results.csv"
df_cv.to_csv(cv_csv_path, index=False)
print(f"[OUTPUT] Saved cv_fold_results.csv to {cv_csv_path}")

# Out-of-fold summary
df_all["oof_calibrated_threshold"] = oof_thresh
df_all["oof_predicted_label"] = oof_preds
df_all["oof_predicted_class"] = ["DISTORTED" if p == 1 else "CLEAN" for p in oof_preds]
df_all["oof_is_correct"] = (df_all["oof_predicted_label"] == df_all["actual_label"])
df_all["oof_status"] = ["CORRECT" if c else "INCORRECT" for c in df_all["oof_is_correct"]]

oof_pred_csv_path = target_dir / "out_of_fold_predictions.csv"
df_all[[
    "filename",
    "participant_id",
    "finger",
    "actual_label",
    "prob_distorted",
    "oof_calibrated_threshold",
    "oof_predicted_label",
    "oof_status",
    "variation_type"
]].to_csv(oof_pred_csv_path, index=False)
print(f"[OUTPUT] Saved out_of_fold_predictions.csv to {oof_pred_csv_path}")

# Calculate overall out-of-fold metrics
oof_metrics = compute_metrics(y_true_all, oof_preds, probs_all)
mean_cv_metrics = {
    "mean_threshold": float(df_cv["selected_threshold"].mean()),
    "std_threshold": float(df_cv["selected_threshold"].std()),
    "mean_accuracy": float(df_cv["accuracy"].mean()),
    "std_accuracy": float(df_cv["accuracy"].std()),
    "mean_precision": float(df_cv["precision"].mean()),
    "std_precision": float(df_cv["precision"].std()),
    "mean_recall": float(df_cv["recall"].mean()),
    "std_recall": float(df_cv["recall"].std()),
    "mean_f1": float(df_cv["f1_score"].mean()),
    "std_f1": float(df_cv["f1_score"].std()),
    "mean_specificity": float(df_cv["specificity"].mean()),
    "std_specificity": float(df_cv["specificity"].std()),
    "mean_balanced_accuracy": float(df_cv["balanced_accuracy"].mean()),
    "std_balanced_accuracy": float(df_cv["balanced_accuracy"].std()),
    "mean_roc_auc": float(df_cv["roc_auc"].mean()),
    "std_roc_auc": float(df_cv["roc_auc"].std())
}

print("\n" + "-" * 50)
print("OUT-OF-FOLD CROSS-VALIDATION SUMMARY:")
print("-" * 50)
print(f"Mean Selected Threshold: {mean_cv_metrics['mean_threshold']:.4f} ± {mean_cv_metrics['std_threshold']:.4f}")
print(f"Overall OOF Accuracy:    {oof_metrics['accuracy']*100:.2f}% (Mean across folds: {mean_cv_metrics['mean_accuracy']*100:.2f}% ± {mean_cv_metrics['std_accuracy']*100:.2f}%)")
print(f"Overall OOF F1-Score:    {oof_metrics['f1_score']*100:.2f}% (Mean across folds: {mean_cv_metrics['mean_f1']*100:.2f}% ± {mean_cv_metrics['std_f1']*100:.2f}%)")
print(f"Overall OOF Recall:      {oof_metrics['recall']*100:.2f}% (Mean across folds: {mean_cv_metrics['mean_recall']*100:.2f}% ± {mean_cv_metrics['std_recall']*100:.2f}%)")
print(f"Overall OOF Specificity: {oof_metrics['specificity']*100:.2f}% (Mean across folds: {mean_cv_metrics['mean_specificity']*100:.2f}% ± {mean_cv_metrics['std_specificity']*100:.2f}%)")
print(f"Overall OOF ROC-AUC:     {oof_metrics['roc_auc']:.4f}")
print(f"Overall OOF Confusion Matrix: TN={oof_metrics['tn']}, FP={oof_metrics['fp']}, FN={oof_metrics['fn']}, TP={oof_metrics['tp']}")

# ==============================================================================
# 4. DIAGNOSTIC THRESHOLD SWEEP (0.50, 0.10, 0.05, 0.04, 0.035, 0.03, 0.02)
# ==============================================================================
print("\n" + "=" * 60)
print("EVALUATING DIAGNOSTIC THRESHOLDS")
print("=" * 60)

test_thresholds = [0.50, 0.10, 0.05, 0.04, 0.035, 0.03, 0.02]
sweep_rows = []

for th in test_thresholds:
    pred_th = (probs_all >= th).astype(int)
    m = compute_metrics(y_true_all, pred_th, probs_all)
    sweep_rows.append({
        "threshold": th,
        "accuracy_pct": round(m["accuracy"] * 100, 2),
        "precision_pct": round(m["precision"] * 100, 2),
        "recall_pct": round(m["recall"] * 100, 2),
        "f1_score_pct": round(m["f1_score"] * 100, 2),
        "specificity_pct": round(m["specificity"] * 100, 2),
        "balanced_accuracy_pct": round(m["balanced_accuracy"] * 100, 2),
        "tp": m["tp"],
        "fp": m["fp"],
        "tn": m["tn"],
        "fn": m["fn"],
        "fpr_pct": round(m["fpr"] * 100, 2),
        "fnr_pct": round(m["fnr"] * 100, 2)
    })
    print(f"Th={th:5.3f} | Acc={m['accuracy']*100:5.2f}% | Rec={m['recall']*100:5.2f}% | Spec={m['specificity']*100:5.2f}% | Prec={m['precision']*100:5.2f}% | F1={m['f1_score']*100:5.2f}% | BalAcc={m['balanced_accuracy']*100:5.2f}% | TP={m['tp']:2d}, FP={m['fp']:2d}")

df_sweep = pd.DataFrame(sweep_rows)
sweep_csv_path = target_dir / "threshold_sweep_diagnostics.csv"
df_sweep.to_csv(sweep_csv_path, index=False)
print(f"[OUTPUT] Saved threshold_sweep_diagnostics.csv to {sweep_csv_path}")

# ==============================================================================
# 5. SYNTHETIC-TO-PHYSICAL DOMAIN GAP INVESTIGATION
# ==============================================================================
print("\n" + "=" * 60)
print("INVESTIGATING SYNTHETIC-TO-PHYSICAL DOMAIN GAP")
print("=" * 60)

synth_test_manifest = PROJ_ROOT / "experiments" / "classifier_orientation_v1" / "splits" / "test_manifest.csv"
synth_records = []

if synth_test_manifest.exists():
    df_synth = pd.read_csv(synth_test_manifest)
    print(f"[DOMAIN GAP] Evaluating {len(df_synth)} synthetic test samples...")
    
    with torch.no_grad():
        for _, row in df_synth.iterrows():
            img_p = PROJ_ROOT / row["filepath"]
            if not img_p.exists():
                continue
            
            raw_s = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
            if raw_s is None:
                continue
            
            mean_s = float(raw_s.mean())
            std_s = float(raw_s.std())
            lap_s = float(cv2.Laplacian(raw_s, cv2.CV_64F).var())
            fg_mask_s = (raw_s < 220).astype(np.uint8)
            fg_cov_s = float(np.sum(fg_mask_s) / (raw_s.shape[0] * raw_s.shape[1]) * 100.0)
            
            # Non-background intensity (ridge intensity)
            ridge_pixels_s = raw_s[raw_s < 220]
            ridge_mean_s = float(ridge_pixels_s.mean()) if len(ridge_pixels_s) > 0 else 0.0
            ridge_p5_s = float(np.percentile(ridge_pixels_s, 5)) if len(ridge_pixels_s) > 0 else 0.0
            
            # Background intensity (pixels >= 220)
            bg_pixels_s = raw_s[raw_s >= 220]
            bg_mean_s = float(bg_pixels_s.mean()) if len(bg_pixels_s) > 0 else 255.0
            
            # Squeeze to 224x224
            img_s_224 = cv2.resize(raw_s, (224, 224), interpolation=cv2.INTER_AREA)
            img_s_norm = img_s_224.astype(np.float32) / 255.0
            tensor_s = torch.from_numpy(img_s_norm).unsqueeze(0).unsqueeze(0)
            
            logit_s = model(tensor_s)
            p_dist_s = float(torch.sigmoid(logit_s).item())
            
            synth_records.append({
                "filename": row["filename"],
                "domain": "Synthetic",
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "prob_distorted": p_dist_s,
                "mean_intensity": mean_s,
                "std_intensity": std_s,
                "sharpness_laplacian": lap_s,
                "fg_coverage_pct": fg_cov_s,
                "ridge_mean_intensity": ridge_mean_s,
                "ridge_p5_intensity": ridge_p5_s,
                "bg_mean_intensity": bg_mean_s
            })

df_synth_res = pd.DataFrame(synth_records)

# Extract physical optical properties
phys_records = []
for p in all_image_paths:
    raw_p = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    is_clean = "_clean.png" in p.name
    gt_label = 0 if is_clean else 1
    
    mean_p = float(raw_p.mean())
    std_p = float(raw_p.std())
    lap_p = float(cv2.Laplacian(raw_p, cv2.CV_64F).var())
    
    fg_mask_p = (raw_p < 220).astype(np.uint8)
    fg_cov_p = float(np.sum(fg_mask_p) / (raw_p.shape[0] * raw_p.shape[1]) * 100.0)
    
    ridge_pixels_p = raw_p[raw_p < 220]
    ridge_mean_p = float(ridge_pixels_p.mean()) if len(ridge_pixels_p) > 0 else 0.0
    ridge_p5_p = float(np.percentile(ridge_pixels_p, 5)) if len(ridge_pixels_p) > 0 else 0.0
    
    bg_pixels_p = raw_p[raw_p >= 220]
    bg_mean_p = float(bg_pixels_p.mean()) if len(bg_pixels_p) > 0 else 255.0
    
    # Probability from df_all
    p_dist_val = float(df_all[df_all["filename"] == p.name]["prob_distorted"].iloc[0])
    
    phys_records.append({
        "filename": p.name,
        "domain": "Physical (SecuGen)",
        "label": gt_label,
        "class_name": "clean" if is_clean else "distorted",
        "prob_distorted": p_dist_val,
        "mean_intensity": mean_p,
        "std_intensity": std_p,
        "sharpness_laplacian": lap_p,
        "fg_coverage_pct": fg_cov_p,
        "ridge_mean_intensity": ridge_mean_p,
        "ridge_p5_intensity": ridge_p5_p,
        "bg_mean_intensity": bg_mean_p
    })

df_phys_res = pd.DataFrame(phys_records)

# Compare domains
domain_comp = []
for d_name, d_df in [("Synthetic Dataset", df_synth_res), ("Physical SecuGen Dataset", df_phys_res)]:
    d_clean = d_df[d_df["label"] == 0]
    d_dist = d_df[d_df["label"] == 1]
    
    domain_comp.append({
        "dataset_domain": d_name,
        "total_samples": len(d_df),
        "clean_samples": len(d_clean),
        "distorted_samples": len(d_dist),
        "clean_mean_prob": round(float(d_clean["prob_distorted"].mean()), 6),
        "distorted_mean_prob": round(float(d_dist["prob_distorted"].mean()), 6),
        "clean_prob_range": f"[{d_clean['prob_distorted'].min():.6f}, {d_clean['prob_distorted'].max():.6f}]",
        "distorted_prob_range": f"[{d_dist['prob_distorted'].min():.6f}, {d_dist['prob_distorted'].max():.6f}]",
        "mean_overall_intensity": round(float(d_df["mean_intensity"].mean()), 2),
        "mean_ridge_intensity": round(float(d_df["ridge_mean_intensity"].mean()), 2),
        "mean_ridge_p5_darkness": round(float(d_df["ridge_p5_intensity"].mean()), 2),
        "mean_bg_intensity": round(float(d_df["bg_mean_intensity"].mean()), 2),
        "mean_sharpness_laplacian": round(float(d_df["sharpness_laplacian"].mean()), 2),
        "mean_fg_coverage_pct": round(float(d_df["fg_coverage_pct"].mean()), 2)
    })

df_domain_comp = pd.DataFrame(domain_comp)
domain_csv_path = target_dir / "domain_gap_analysis.csv"
df_domain_comp.to_csv(domain_csv_path, index=False)
print(f"[OUTPUT] Saved domain_gap_analysis.csv to {domain_csv_path}")
print("\nDomain Gap Comparison Table:")
print(df_domain_comp.T.to_string())

# ==============================================================================
# 6. VISUALIZATIONS & PLOTS
# ==============================================================================
# Plot 1: OOF Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm_oof = np.array([[oof_metrics["tn"], oof_metrics["fp"]], [oof_metrics["fn"], oof_metrics["tp"]]])
im = ax.imshow(cm_oof, cmap="Blues", interpolation="nearest")
plt.colorbar(im)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted CLEAN", "Predicted DISTORTED"], fontsize=11, fontweight="bold")
ax.set_yticklabels(["Actual CLEAN", "Actual DISTORTED"], fontsize=11, fontweight="bold")

thresh_color = cm_oof.max() / 2.0
for i in range(2):
    for j in range(2):
        val = cm_oof[i, j]
        pct = (val / 42.0) * 100.0
        color = "white" if val > thresh_color else "black"
        ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

ax.set_title(f"Out-of-Fold Confusion Matrix (7-Fold GroupKFold, N=84)\nOverall Accuracy = {oof_metrics['accuracy']*100:.1f}% | Mean Cal Th = {mean_cv_metrics['mean_threshold']:.3f}", fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
oof_cm_path = target_dir / "oof_confusion_matrix.png"
plt.savefig(oof_cm_path, dpi=300)
plt.close()
print(f"[PLOT] Saved oof_confusion_matrix.png to {oof_cm_path}")

# Plot 2: OOF ROC Curve
fpr_pts, tpr_pts, _ = roc_curve(y_true_all, probs_all)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_pts, tpr_pts, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {oof_metrics['roc_auc']:.4f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Chance Level (AUC = 0.50)")
# Mark OOF operating point
ax.scatter([oof_metrics["fpr"]], [oof_metrics["recall"]], color="red", s=100, zorder=5, label=f"OOF Operating Point (FPR={oof_metrics['fpr']:.2f}, TPR={oof_metrics['recall']:.2f})")
# Mark threshold 0.50 point
ax.scatter([0.0], [0.0], color="purple", s=100, marker="^", zorder=5, label="Threshold 0.50 (FPR=0.00, TPR=0.00)")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
ax.set_title("Out-of-Fold ROC Curve on Physical Dataset", fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
oof_roc_path = target_dir / "oof_roc_curve.png"
plt.savefig(oof_roc_path, dpi=300)
plt.close()
print(f"[PLOT] Saved oof_roc_curve.png to {oof_roc_path}")

# Plot 3: Threshold Sensitivity Curve
fig, ax = plt.subplots(figsize=(8, 5))
sweep_thresholds = np.arange(0.005, 0.205, 0.005)
sw_prec, sw_rec, sw_spec, sw_f1, sw_bal = [], [], [], [], []

for t in sweep_thresholds:
    p_t = (probs_all >= t).astype(int)
    m = compute_metrics(y_true_all, p_t, probs_all)
    sw_prec.append(m["precision"] * 100)
    sw_rec.append(m["recall"] * 100)
    sw_spec.append(m["specificity"] * 100)
    sw_f1.append(m["f1_score"] * 100)
    sw_bal.append(m["balanced_accuracy"] * 100)

ax.plot(sweep_thresholds, sw_rec, label="Recall (Distorted Sensitivity)", color="#d62728", lw=2)
ax.plot(sweep_thresholds, sw_spec, label="Specificity (Clean)", color="#2ca02c", lw=2)
ax.plot(sweep_thresholds, sw_f1, label="F1-Score", color="#1f77b4", lw=2.5)
ax.plot(sweep_thresholds, sw_bal, label="Balanced Accuracy", color="#ff7f0e", linestyle="--", lw=2)
ax.plot(sweep_thresholds, sw_prec, label="Precision", color="#9467bd", linestyle=":", lw=2)

ax.axvline(mean_cv_metrics["mean_threshold"], color="black", linestyle="--", lw=1.5, label=f"CV Mean Threshold ({mean_cv_metrics['mean_threshold']:.3f})")
ax.set_xlim([0.005, 0.20])
ax.set_ylim([0, 105])
ax.set_xlabel("Decision Threshold", fontsize=11, fontweight="bold")
ax.set_ylabel("Metric Score (%)", fontsize=11, fontweight="bold")
ax.set_title("Performance Trade-offs vs Decision Threshold on Physical Dataset", fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="center right", fontsize=9)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
thresh_plot_path = target_dir / "threshold_sensitivity_curve.png"
plt.savefig(thresh_plot_path, dpi=300)
plt.close()
print(f"[PLOT] Saved threshold_sensitivity_curve.png to {thresh_plot_path}")

# Plot 4: Domain Gap Feature Distribution Comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Subplot 1: Probability Distribution
axes[0, 0].hist(df_synth_res[df_synth_res["label"] == 1]["prob_distorted"], bins=20, alpha=0.6, color="red", label="Synth Distorted")
axes[0, 0].hist(df_synth_res[df_synth_res["label"] == 0]["prob_distorted"], bins=20, alpha=0.6, color="blue", label="Synth Clean")
axes[0, 0].hist(df_phys_res[df_phys_res["label"] == 1]["prob_distorted"], bins=20, alpha=0.8, color="darkorange", label="Phys Distorted")
axes[0, 0].hist(df_phys_res[df_phys_res["label"] == 0]["prob_distorted"], bins=20, alpha=0.8, color="green", label="Phys Clean")
axes[0, 0].set_title("P(DISTORTED) Distribution", fontweight="bold")
axes[0, 0].set_xlabel("Probability")
axes[0, 0].legend(fontsize=8)

# Subplot 2: Sharpness Comparison
axes[0, 1].boxplot([
    df_synth_res["sharpness_laplacian"],
    df_phys_res["sharpness_laplacian"]
], tick_labels=["Synthetic", "Physical (SecuGen)"], patch_artist=True)
axes[0, 1].set_title("Laplacian Sharpness Variance", fontweight="bold")
axes[0, 1].set_yscale("log")

# Subplot 3: Ridge Darkness (5th percentile)
axes[1, 0].boxplot([
    df_synth_res["ridge_p5_intensity"],
    df_phys_res["ridge_p5_intensity"]
], tick_labels=["Synthetic", "Physical (SecuGen)"], patch_artist=True)
axes[1, 0].set_title("Ridge Darkness (5th Percentile Intensity)", fontweight="bold")

# Subplot 4: Foreground Area Coverage
axes[1, 1].boxplot([
    df_synth_res["fg_coverage_pct"],
    df_phys_res["fg_coverage_pct"]
], tick_labels=["Synthetic", "Physical (SecuGen)"], patch_artist=True)
axes[1, 1].set_title("Foreground Ridge Coverage (%)", fontweight="bold")

plt.tight_layout()
domain_plot_path = target_dir / "domain_gap_distributions.png"
plt.savefig(domain_plot_path, dpi=300)
plt.close()
print(f"[PLOT] Saved domain_gap_distributions.png to {domain_plot_path}")

print("\n" + "=" * 80)
print("EVALUATION & CALIBRATION EXECUTION COMPLETED!")
print("=" * 80)
