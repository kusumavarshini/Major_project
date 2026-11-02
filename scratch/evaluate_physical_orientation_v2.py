import os
import sys
import json
import shutil
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
print("FINGERSURE PHYSICAL EVALUATION: EXPERIMENT 2 (CLASSIFIER_ORIENTATION_V2)")
print("=" * 80)

# ==============================================================================
# 1. SETUP MODEL AND CHECKPOINT
# ==============================================================================
exp2_dir = PROJ_ROOT / "experiments" / "classifier_orientation_v2"
ckpt_dir = exp2_dir / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)

eval_dir = exp2_dir / "physical_evaluation"
eval_dir.mkdir(parents=True, exist_ok=True)

# Checkpoint resolution
best_model_v2 = exp2_dir / "best_model.pth"
best_model_downloaded = exp2_dir / "best_model (4).pth"

if not best_model_v2.exists():
    if best_model_downloaded.exists():
        print(f"[SETUP] Copying downloaded checkpoint '{best_model_downloaded.name}' to '{best_model_v2.name}'...")
        shutil.copy2(best_model_downloaded, best_model_v2)
    else:
        raise FileNotFoundError(f"Checkpoint not found in {exp2_dir}")

print(f"[SETUP] Using checkpoint: {best_model_v2} ({best_model_v2.stat().st_size:,} bytes)")

# Load checkpoint dict
ckpt = torch.load(best_model_v2, map_location="cpu", weights_only=False)
threshold_val = float(ckpt.get("threshold", 0.50))
best_epoch = ckpt.get("epoch", "N/A")
val_metrics = ckpt.get("val_metrics", {})

# Save threshold.json in checkpoints/ if not present
thresh_json_path = ckpt_dir / "threshold.json"
threshold_data = {
    "calibrated_threshold": threshold_val,
    "best_epoch": best_epoch,
    "validation_metrics": val_metrics,
    "source_checkpoint": str(best_model_v2.name),
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}
with open(thresh_json_path, "w", encoding="utf-8") as f:
    json.dump(threshold_data, f, indent=2)
print(f"[SETUP] Threshold JSON written to {thresh_json_path} (Threshold: {threshold_val})")

# Instantiate FingerprintCNN
model = FingerprintCNN(dropout=0.5)
state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
load_res = model.load_state_dict(state_dict, strict=True)
model.eval()
print(f"[MODEL] FingerprintCNN successfully loaded: {count_parameters(model):,} parameters")

# ==============================================================================
# 2. LOCATE PHYSICAL SECUGEN DATASET
# ==============================================================================
dataset_dir = PROJ_ROOT / "FingerSure_Dataset"
clean_dir = dataset_dir / "clean"
dist_dir = dataset_dir / "distorted"
metadata_csv = dataset_dir / "metadata.csv"

if not clean_dir.exists() or not dist_dir.exists():
    raise FileNotFoundError(f"FingerSure_Dataset clean or distorted directory missing in {dataset_dir}")

clean_files = sorted(list(clean_dir.glob("*.png")))
dist_files = sorted(list(dist_dir.glob("*.png")))

print(f"[DATASET] Discovered {len(clean_files)} CLEAN images and {len(dist_files)} DISTORTED images.")
assert len(clean_files) == 42, f"Expected 42 clean images, found {len(clean_files)}"
assert len(dist_files) == 42, f"Expected 42 distorted images, found {len(dist_files)}"
print("[DATASET] Total 84 physical images confirmed. Ground-truth unambiguously mapped: CLEAN=0, DISTORTED=1.")

# Load metadata for physical variations if available
meta_df = pd.read_csv(metadata_csv) if metadata_csv.exists() else None

# Build full sample list
samples = []
for p in clean_files:
    samples.append({
        "filepath": p,
        "filename": p.name,
        "identity": p.name.replace("_clean.png", ""),
        "ground_truth_label": 0,
        "ground_truth_class": "CLEAN"
    })

for p in dist_files:
    samples.append({
        "filepath": p,
        "filename": p.name,
        "identity": p.name.replace("_distorted.png", ""),
        "ground_truth_label": 1,
        "ground_truth_class": "DISTORTED"
    })

# ==============================================================================
# 3. PREPROCESSING & INFERENCE
# ==============================================================================
print(f"\n[INFERENCE] Evaluating 84 physical fingerprint images with locked threshold {threshold_val}...")

predictions = []

with torch.no_grad():
    for s in samples:
        img_path = s["filepath"]
        
        # Exact preprocessing:
        # 1. Load image
        raw_img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if raw_img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        
        h_orig, w_orig = raw_img.shape[:2]
        
        # Calculate image metrics for variation analysis
        mean_intensity = float(raw_img.mean())
        std_intensity = float(raw_img.std())
        laplacian_var = float(cv2.Laplacian(raw_img, cv2.CV_64F).var())
        
        # Segment foreground (ridges) vs optical background (white > 230)
        fg_mask = (raw_img < 220).astype(np.uint8)
        fg_coverage_pct = float(np.sum(fg_mask) / (h_orig * w_orig) * 100.0)
        
        # Estimate orientation/rotation angle via second-order central moments of foreground
        moments = cv2.moments(fg_mask)
        if moments["mu20"] != moments["mu02"]:
            angle_rad = 0.5 * np.arctan2(2 * moments["mu11"], (moments["mu20"] - moments["mu02"]))
            angle_deg = float(np.degrees(angle_rad))
        else:
            angle_deg = 0.0

        # 2. Resize to 224x224 via cv2.INTER_AREA
        img_224 = cv2.resize(raw_img, (224, 224), interpolation=cv2.INTER_AREA)
        
        # 3. Float32 normalization
        img_norm = img_224.astype(np.float32) / 255.0
        
        # 4. Tensor conversion: shape (1, 1, 224, 224)
        tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0)
        
        t0 = time.perf_counter()
        logit = model(tensor)
        inference_time_ms = (time.perf_counter() - t0) * 1000.0
        
        prob_distorted = float(torch.sigmoid(logit).item())
        prob_clean = 1.0 - prob_distorted
        
        pred_label = 1 if prob_distorted >= threshold_val else 0
        pred_class = "DISTORTED" if pred_label == 1 else "CLEAN"
        is_correct = (pred_label == s["ground_truth_label"])
        status = "CORRECT" if is_correct else "INCORRECT"
        
        # Determine likely physical variation category based on physical characteristics
        # (Rotated / Tilted / Sideways, Partial-contact, Uneven-contact, Normal)
        if s["ground_truth_label"] == 0:
            variation_category = "Normal Relaxed"
        else:
            # Physical characterization from optical properties
            if fg_coverage_pct < 28.0:
                variation_category = "Partial Contact"
            elif mean_intensity > 195.0 or std_intensity < 55.0:
                variation_category = "Uneven / Light Contact"
            elif abs(angle_deg) > 20.0:
                variation_category = "Rotated / Sideways"
            else:
                variation_category = "Tilted / Shear Distortion"

        predictions.append({
            "image_filename": s["filename"],
            "identity": s["identity"],
            "ground_truth_label": s["ground_truth_label"],
            "ground_truth_class": s["ground_truth_class"],
            "predicted_label": pred_label,
            "predicted_class": pred_class,
            "prob_distorted": round(prob_distorted, 6),
            "prob_clean": round(prob_clean, 6),
            "classification_result": status,
            "is_correct": is_correct,
            "inference_time_ms": round(inference_time_ms, 3),
            "physical_variation": variation_category,
            "mean_intensity": round(mean_intensity, 2),
            "sharpness_laplacian": round(laplacian_var, 2),
            "fg_coverage_pct": round(fg_coverage_pct, 2),
            "estimated_angle_deg": round(angle_deg, 2)
        })

df_preds = pd.DataFrame(predictions)

# Save predictions CSV
pred_csv_path = eval_dir / "predictions.csv"
df_preds[[
    "image_filename",
    "ground_truth_label",
    "predicted_label",
    "prob_distorted",
    "classification_result"
]].to_csv(pred_csv_path, index=False)
print(f"[OUTPUT] Detailed prediction CSV written to {pred_csv_path}")

# Also save full diagnostic predictions CSV
full_csv_path = eval_dir / "predictions_full_diagnostics.csv"
df_preds.to_csv(full_csv_path, index=False)

# ==============================================================================
# 4. COMPREHENSIVE METRICS CALCULATION
# ==============================================================================
y_true = df_preds["ground_truth_label"].to_numpy()
y_pred = df_preds["predicted_label"].to_numpy()
y_probs = df_preds["prob_distorted"].to_numpy()

total_samples = len(df_preds)
correct_count = int(df_preds["is_correct"].sum())
incorrect_count = total_samples - correct_count

# Confusion matrix:
# | Actual CLEAN | TN | FP |
# | Actual DISTORTED | FN | TP |
tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
bal_acc = 0.5 * (rec + spec)
roc_auc = roc_auc_score(y_true, y_probs)

fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

# Separate class subsets
clean_df = df_preds[df_preds["ground_truth_label"] == 0]
dist_df = df_preds[df_preds["ground_truth_label"] == 1]

clean_acc = float(clean_df["is_correct"].mean())
dist_acc = float(dist_df["is_correct"].mean())

clean_mean_p = float(clean_df["prob_distorted"].mean())
clean_std_p = float(clean_df["prob_distorted"].std())
dist_mean_p = float(dist_df["prob_distorted"].mean())
dist_std_p = float(dist_df["prob_distorted"].std())

print("\n" + "=" * 80)
print("EVALUATION RESULTS SUMMARY (Physical SecuGen Dataset - 84 Images)")
print("=" * 80)
print(f"Total Evaluated:      {total_samples} images (42 CLEAN, 42 DISTORTED)")
print(f"Correctly Classified: {correct_count} / {total_samples} ({acc*100:.2f}%)")
print(f"Incorrectly Classified:{incorrect_count} / {total_samples} ({(1-acc)*100:.2f}%)")
print("-" * 50)
print(f"Accuracy:             {acc*100:.2f}%")
print(f"Precision:            {prec*100:.2f}%")
print(f"Recall (Sensitivity): {rec*100:.2f}%")
print(f"F1-Score:             {f1*100:.2f}%")
print(f"Specificity:          {spec*100:.2f}%")
print(f"Balanced Accuracy:    {bal_acc*100:.2f}%")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"False Positive Rate:  {fpr*100:.2f}% ({fp}/42 CLEAN misclassified as DISTORTED)")
print(f"False Negative Rate:  {fnr*100:.2f}% ({fn}/42 DISTORTED misclassified as CLEAN)")
print("-" * 50)
print("Confusion Matrix:")
print("                   Predicted CLEAN | Predicted DISTORTED")
print(f"Actual CLEAN:      TN = {tn:2d}        | FP = {fp:2d}")
print(f"Actual DISTORTED:  FN = {fn:2d}        | TP = {tp:2d}")
print("-" * 50)
print(f"CLEAN Subgroup:     Accuracy = {clean_acc*100:.2f}% (TN={tn}, FP={fp}) | Mean P(Dist) = {clean_mean_p:.4f} ± {clean_std_p:.4f}")
print(f"DISTORTED Subgroup: Accuracy = {dist_acc*100:.2f}% (TP={tp}, FN={fn}) | Mean P(Dist) = {dist_mean_p:.4f} ± {dist_std_p:.4f}")

# Save summary metrics CSV
summary_metrics = [
    {"Metric": "Total Samples", "Value": total_samples},
    {"Metric": "Clean Samples", "Value": len(clean_df)},
    {"Metric": "Distorted Samples", "Value": len(dist_df)},
    {"Metric": "Locked Threshold", "Value": threshold_val},
    {"Metric": "Correctly Classified", "Value": correct_count},
    {"Metric": "Incorrectly Classified", "Value": incorrect_count},
    {"Metric": "Accuracy (%)", "Value": round(acc * 100, 2)},
    {"Metric": "Precision (%)", "Value": round(prec * 100, 2)},
    {"Metric": "Recall (%)", "Value": round(rec * 100, 2)},
    {"Metric": "F1-Score (%)", "Value": round(f1 * 100, 2)},
    {"Metric": "Specificity (%)", "Value": round(spec * 100, 2)},
    {"Metric": "Balanced Accuracy (%)", "Value": round(bal_acc * 100, 2)},
    {"Metric": "ROC-AUC", "Value": round(roc_auc, 4)},
    {"Metric": "False Positive Rate (%)", "Value": round(fpr * 100, 2)},
    {"Metric": "False Negative Rate (%)", "Value": round(fnr * 100, 2)},
    {"Metric": "True Negatives (TN)", "Value": int(tn)},
    {"Metric": "False Positives (FP)", "Value": int(fp)},
    {"Metric": "False Negatives (FN)", "Value": int(fn)},
    {"Metric": "True Positives (TP)", "Value": int(tp)},
    {"Metric": "Clean Accuracy (%)", "Value": round(clean_acc * 100, 2)},
    {"Metric": "Distorted Accuracy (%)", "Value": round(dist_acc * 100, 2)},
    {"Metric": "Clean Mean P(Distorted)", "Value": round(clean_mean_p, 4)},
    {"Metric": "Distorted Mean P(Distorted)", "Value": round(dist_mean_p, 4)},
]
summary_csv_path = eval_dir / "summary_metrics.csv"
pd.DataFrame(summary_metrics).to_csv(summary_csv_path, index=False)
print(f"[OUTPUT] Summary metrics CSV written to {summary_csv_path}")

# ==============================================================================
# 5. PHYSICAL VARIATION PERFORMANCE BREAKDOWN
# ==============================================================================
variation_groups = df_preds.groupby("physical_variation")
variation_stats = []

for var_name, g in variation_groups:
    n_total = len(g)
    n_corr = int(g["is_correct"].sum())
    n_inc = n_total - n_corr
    mean_p = float(g["prob_distorted"].mean())
    std_p = float(g["prob_distorted"].std()) if len(g) > 1 else 0.0
    acc_grp = float(n_corr / n_total)
    
    variation_stats.append({
        "variation_type": var_name,
        "total_samples": n_total,
        "correct": n_corr,
        "incorrect": n_inc,
        "accuracy_pct": round(acc_grp * 100, 2),
        "mean_prob_distorted": round(mean_p, 4),
        "std_prob_distorted": round(std_p, 4)
    })

df_vars = pd.DataFrame(variation_stats)
var_csv_path = eval_dir / "physical_variation_breakdown.csv"
df_vars.to_csv(var_csv_path, index=False)
print(f"\n[PHYSICAL VARIATIONS] Performance Breakdown:")
print(df_vars.to_string(index=False))

# Incorrectly classified images list
incorrect_df = df_preds[df_preds["classification_result"] == "INCORRECT"]
print(f"\n[INCORRECT IMAGES] Total: {len(incorrect_df)}")
for _, row in incorrect_df.iterrows():
    print(f"  - {row['image_filename']}: GT={row['ground_truth_class']} -> Pred={row['predicted_class']} (P(Dist)={row['prob_distorted']:.4f}) [{row['physical_variation']}]")

# ==============================================================================
# 6. JSON EVALUATION REPORT
# ==============================================================================
report_json_data = {
    "experiment": "classifier_orientation_v2",
    "evaluation_target": "physical_secugen_dataset_84_images",
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "checkpoint": str(best_model_v2.name),
    "locked_threshold": threshold_val,
    "dataset": {
        "source": "FingerSure_Dataset/",
        "total_images": total_samples,
        "clean_images": len(clean_df),
        "distorted_images": len(dist_df)
    },
    "metrics": {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "specificity": round(spec, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "roc_auc": round(roc_auc, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    },
    "subgroups": {
        "clean": {
            "total": len(clean_df),
            "correct_tn": int(tn),
            "incorrect_fp": int(fp),
            "accuracy": round(clean_acc, 4),
            "mean_prob_distorted": round(clean_mean_p, 4),
            "std_prob_distorted": round(clean_std_p, 4)
        },
        "distorted": {
            "total": len(dist_df),
            "correct_tp": int(tp),
            "incorrect_fn": int(fn),
            "accuracy": round(dist_acc, 4),
            "mean_prob_distorted": round(dist_mean_p, 4),
            "std_prob_distorted": round(dist_std_p, 4)
        }
    },
    "physical_variations": variation_stats,
    "incorrect_samples": [
        {
            "filename": r["image_filename"],
            "ground_truth": r["ground_truth_class"],
            "predicted": r["predicted_class"],
            "prob_distorted": r["prob_distorted"],
            "physical_variation": r["physical_variation"]
        }
        for _, r in incorrect_df.iterrows()
    ]
}

report_json_path = eval_dir / "evaluation_report.json"
with open(report_json_path, "w", encoding="utf-8") as f:
    json.dump(report_json_data, f, indent=2)
print(f"[OUTPUT] Full evaluation report JSON written to {report_json_path}")

# ==============================================================================
# 7. VISUALIZATIONS & GRAPHS
# ==============================================================================
# Graph 1: Confusion Matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm_data = np.array([[tn, fp], [fn, tp]])
im = ax.imshow(cm_data, cmap="Blues", interpolation="nearest")
plt.colorbar(im)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted CLEAN", "Predicted DISTORTED"], fontsize=11, fontweight="bold")
ax.set_yticklabels(["Actual CLEAN", "Actual DISTORTED"], fontsize=11, fontweight="bold")

thresh_color = cm_data.max() / 2.0
for i in range(2):
    for j in range(2):
        val = cm_data[i, j]
        pct = (val / 42.0) * 100.0
        color = "white" if val > thresh_color else "black"
        ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=14, fontweight="bold")

ax.set_title(f"Physical SecuGen Confusion Matrix (N=84)\nThreshold = {threshold_val} | Accuracy = {acc*100:.1f}%", fontsize=12, fontweight="bold", pad=12)
plt.tight_layout()
cm_plot_path = eval_dir / "confusion_matrix.png"
plt.savefig(cm_plot_path, dpi=300)
plt.close()
print(f"[PLOT] Confusion matrix plot saved to {cm_plot_path}")

# Graph 2: ROC Curve
fpr_pts, tpr_pts, thresholds = roc_curve(y_true, y_probs)
fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(fpr_pts, tpr_pts, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Random Guess (AUC = 0.50)")
# Mark the operating point
ax.scatter([fpr], [rec], color="red", s=100, zorder=5, label=f"Threshold {threshold_val} (FPR={fpr:.2f}, TPR={rec:.2f})")
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
ax.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
ax.set_title(f"ROC Curve on Physical SecuGen Dataset\nAUC = {roc_auc:.4f}", fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
roc_plot_path = eval_dir / "roc_curve.png"
plt.savefig(roc_plot_path, dpi=300)
plt.close()
print(f"[PLOT] ROC curve plot saved to {roc_plot_path}")

# Graph 3: Classification Metrics Bar Chart
fig, ax = plt.subplots(figsize=(8, 5))
metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "Specificity", "Balanced Acc"]
metric_vals = [acc * 100, prec * 100, rec * 100, f1 * 100, spec * 100, bal_acc * 100]
colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]

bars = ax.bar(metric_names, metric_vals, color=colors, width=0.55, edgecolor="black", linewidth=1.2)
ax.set_ylim(0, 110)
ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
ax.set_title(f"Classification Performance on Physical Fingerprints\nClassifier Orientation v2 (Threshold = {threshold_val})", fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="y", linestyle=":", alpha=0.6)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2.0, f"{yval:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
metrics_plot_path = eval_dir / "classification_metrics.png"
plt.savefig(metrics_plot_path, dpi=300)
plt.close()
print(f"[PLOT] Metrics bar chart saved to {metrics_plot_path}")

# Graph 4: Physical Variation Breakdown Plot
fig, ax = plt.subplots(figsize=(9, 5))
var_names = [v["variation_type"] for v in variation_stats]
var_accs = [v["accuracy_pct"] for v in variation_stats]
var_counts = [v["total_samples"] for v in variation_stats]
labels_with_n = [f"{n}\n(N={cnt})" for n, cnt in zip(var_names, var_counts)]

bars = ax.bar(labels_with_n, var_accs, color="#4c72b0", width=0.5, edgecolor="black", linewidth=1.2)
ax.set_ylim(0, 115)
ax.set_ylabel("Classification Accuracy (%)", fontsize=11, fontweight="bold")
ax.set_title("Performance Breakdown Across Physical Capture Variations", fontsize=12, fontweight="bold", pad=12)
ax.grid(axis="y", linestyle=":", alpha=0.6)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2.5, f"{yval:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
var_plot_path = eval_dir / "physical_variations_chart.png"
plt.savefig(var_plot_path, dpi=300)
plt.close()
print(f"[PLOT] Physical variations chart saved to {var_plot_path}")

print("\n" + "=" * 80)
print("EVALUATION EXECUTION COMPLETED SUCCESSFULLY!")
print("=" * 80)

