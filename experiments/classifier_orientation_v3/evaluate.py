"""
FingerprintCNN V3 Comprehensive Evaluation Module
=================================================
Evaluates the trained V3 checkpoint on:
1. Synthetic Validation & Held-Out Test Datasets (900 images each).
2. Physical SecuGen Dataset (84 images) using 7-Fold Participant Cross-Validation.
3. Comparative analysis against V2.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedGroupKFold
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

# Reconfigure stdout for utf-8 on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

SCRIPT_DIR = Path(__file__).resolve().parent
PROJ_ROOT = SCRIPT_DIR.parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

from model import FingerprintCNN, count_parameters
from experiments.classifier_orientation_v3.dataset import get_v3_dataloaders, resolve_path


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (rec + spec)

    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5

    fpr = float(fp / (tn + fp)) if (tn + fp) > 0 else 0.0
    fnr = float(fn / (tp + fn)) if (tp + fn) > 0 else 0.0

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "specificity": round(float(spec), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "roc_auc": round(float(auc), 4),
        "fpr": round(float(fpr), 4),
        "fnr": round(float(fnr), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def evaluate_synthetic(model: nn.Module, test_loader: DataLoader, locked_threshold: float, device: torch.device):
    """Evaluates V3 on the held-out synthetic test set."""
    model.eval()
    all_targets = []
    all_probs = []
    all_filenames = []
    all_stems = []
    all_vars = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits)

            all_targets.extend(targets.cpu().numpy().ravel())
            all_probs.extend(probs.cpu().numpy().ravel())
            all_filenames.extend(batch["filename"])
            all_stems.extend(batch["identity_stem"])
            all_vars.extend(batch["variation_type"])

    y_true = np.array(all_targets, dtype=int)
    y_probs = np.array(all_probs, dtype=float)
    y_pred = (y_probs >= locked_threshold).astype(int)

    metrics = compute_metrics(y_true, y_pred, y_probs)
    metrics["locked_threshold"] = locked_threshold

    # Prediction records
    preds_df = pd.DataFrame({
        "filename": all_filenames,
        "identity_stem": all_stems,
        "variation_type": all_vars,
        "ground_truth_label": y_true,
        "ground_truth_class": ["CLEAN" if y == 0 else "DISTORTED" for y in y_true],
        "predicted_label": y_pred,
        "predicted_class": ["CLEAN" if p == 0 else "DISTORTED" for p in y_pred],
        "prob_distorted": np.round(y_probs, 6),
        "prob_clean": np.round(1.0 - y_probs, 6),
        "is_correct": (y_pred == y_true),
        "classification_result": ["CORRECT" if (p == y) else "INCORRECT" for p, y in zip(y_pred, y_true)]
    })

    return metrics, preds_df


def evaluate_physical_secugen(model: nn.Module, device: torch.device):
    """Evaluates V3 on the 84 physical SecuGen images using 7-Fold StratifiedGroupKFold."""
    ds_dir = PROJ_ROOT / "FingerSure_Dataset"
    clean_files = sorted(list((ds_dir / "clean").glob("*.png")))
    dist_files = sorted(list((ds_dir / "distorted").glob("*.png")))
    all_files = clean_files + dist_files

    def parse_fn(name):
        base = name.replace("_clean.png", "").replace("_distorted.png", "")
        parts = base.split("_")
        finger = f"{parts[-2]}_{parts[-1]}"
        participant = "_".join(parts[:-2])
        return participant, finger

    samples = []
    with torch.no_grad():
        for p in all_files:
            part, finger = parse_fn(p.name)
            is_clean = "_clean.png" in p.name
            label = 0 if is_clean else 1

            raw = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            img_224 = cv2.resize(raw, (224, 224), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(img_224.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

            logit = model(tensor)
            prob_dist = float(torch.sigmoid(logit).item())

            samples.append({
                "filename": p.name,
                "participant_id": part,
                "finger": finger,
                "ground_truth_label": label,
                "ground_truth_class": "CLEAN" if label == 0 else "DISTORTED",
                "prob_distorted": round(prob_dist, 6),
                "prob_clean": round(1.0 - prob_dist, 6)
            })

    df_phys = pd.DataFrame(samples)
    groups = df_phys["participant_id"].to_numpy()
    y = df_phys["ground_truth_label"].to_numpy()
    probs = df_phys["prob_distorted"].to_numpy()
    X = np.arange(len(df_phys))

    sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)

    cv_records = []
    oof_preds = np.zeros(len(df_phys), dtype=int)
    oof_thresholds = np.zeros(len(df_phys), dtype=float)

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        cal_part = sorted(list(set(groups[train_idx])))
        val_part = sorted(list(set(groups[val_idx])))

        y_cal, p_cal = y[train_idx], probs[train_idx]
        y_val, p_val = y[val_idx], probs[val_idx]

        # Calibrate threshold strictly on calibration participants
        best_th = 0.50
        best_cal_f1 = -1.0
        for t in np.arange(0.005, 0.40, 0.001):
            t_val = round(float(t), 4)
            pred_cal = (p_cal >= t_val).astype(int)
            sc = f1_score(y_cal, pred_cal, zero_division=0)
            if sc > best_cal_f1:
                best_cal_f1 = sc
                best_th = t_val

        # Evaluate on unseen validation participants
        pred_val = (p_val >= best_th).astype(int)
        oof_preds[val_idx] = pred_val
        oof_thresholds[val_idx] = best_th

        m_val = compute_metrics(y_val, pred_val, p_val)
        cv_records.append({
            "fold": fold + 1,
            "calibration_threshold": best_th,
            "calibration_f1": round(best_cal_f1, 4),
            "val_accuracy": m_val["accuracy"],
            "val_f1": m_val["f1_score"],
            "val_recall": m_val["recall"],
            "val_specificity": m_val["specificity"],
            "val_precision": m_val["precision"],
            "val_auc": m_val["roc_auc"],
            "val_participants": ", ".join(val_part),
            "tp": m_val["tp"],
            "fp": m_val["fp"],
            "tn": m_val["tn"],
            "fn": m_val["fn"]
        })

    df_cv = pd.DataFrame(cv_records)

    # Pooled OOF metrics
    oof_metrics = compute_metrics(y, oof_preds, probs)
    oof_metrics["mean_calibrated_threshold"] = round(float(df_cv["calibration_threshold"].mean()), 4)
    oof_metrics["std_calibrated_threshold"] = round(float(df_cv["calibration_threshold"].std()), 4)

    df_phys["oof_calibrated_threshold"] = oof_thresholds
    df_phys["oof_predicted_label"] = oof_preds
    df_phys["oof_predicted_class"] = ["DISTORTED" if p == 1 else "CLEAN" for p in oof_preds]
    df_phys["oof_is_correct"] = (oof_preds == y)
    df_phys["oof_status"] = ["CORRECT" if c else "INCORRECT" for c in df_phys["oof_is_correct"]]

    return df_phys, df_cv, oof_metrics


def run_full_v3_evaluation(config_path: Path):
    """Coordinates full synthetic and physical evaluation and produces all artifacts."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = resolve_path(cfg["outputs"]["checkpoint_dir"])
    best_model_path = ckpt_dir / cfg["outputs"].get("best_model_filename", "best_model.pth")
    thresh_path = ckpt_dir / cfg["outputs"].get("threshold_filename", "threshold.json")

    with open(thresh_path, "r", encoding="utf-8") as f:
        thresh_info = json.load(f)
    locked_th = float(thresh_info.get("calibrated_threshold", 0.50))

    # Load model
    model = FingerprintCNN(dropout=cfg["model"].get("dropout", 0.5)).to(device)
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[EVAL] Loaded FingerprintCNN V3 from {best_model_path.name} (Locked Threshold: {locked_th:.4f})")

    # 1. Synthetic Held-Out Test Evaluation
    _, _, test_loader, _ = get_v3_dataloaders(config_path)
    synth_metrics, synth_preds_df = evaluate_synthetic(model, test_loader, locked_th, device)

    # Save synthetic test outputs
    test_res_path = SCRIPT_DIR / "test_results.json"
    with open(test_res_path, "w", encoding="utf-8") as f:
        json.dump(synth_metrics, f, indent=2)
    print(f"[OUTPUT] Saved test_results.json to {test_res_path}")

    test_pred_path = SCRIPT_DIR / "test_predictions.csv"
    synth_preds_df.to_csv(test_pred_path, index=False)
    print(f"[OUTPUT] Saved test_predictions.csv to {test_pred_path}")

    test_cm_path = SCRIPT_DIR / "test_confusion_matrix.json"
    with open(test_cm_path, "w", encoding="utf-8") as f:
        json.dump({
            "tp": synth_metrics["tp"],
            "fp": synth_metrics["fp"],
            "tn": synth_metrics["tn"],
            "fn": synth_metrics["fn"]
        }, f, indent=2)

    # Plot Confusion Matrix (Synthetic)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = np.array([[synth_metrics["tn"], synth_metrics["fp"]], [synth_metrics["fn"], synth_metrics["tp"]]])
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted CLEAN", "Predicted DISTORTED"], fontsize=11, fontweight="bold")
    ax.set_yticklabels(["Actual CLEAN", "Actual DISTORTED"], fontsize=11, fontweight="bold")

    thresh_c = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > thresh_c else "black"
            ax.text(j, i, f"{val}\n({val/len(synth_preds_df)*100:.1f}%)", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    ax.set_title(f"Synthetic Test Confusion Matrix (N=900)\nThreshold = {locked_th:.4f} | Accuracy = {synth_metrics['accuracy']*100:.2f}%", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # Plot ROC Curve (Synthetic)
    fpr_pts, tpr_pts, _ = roc_curve(synth_preds_df["ground_truth_label"], synth_preds_df["prob_distorted"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_pts, tpr_pts, color="#1f77b4", lw=2.5, label=f"V3 ROC Curve (AUC = {synth_metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    ax.scatter([synth_metrics["fpr"]], [synth_metrics["recall"]], color="red", s=100, zorder=5, label=f"Threshold {locked_th:.2f} (TPR={synth_metrics['recall']:.2f})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title("ROC Curve on Synthetic Held-Out Test Set (N=900)", fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(SCRIPT_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # 2. Physical SecuGen Evaluation
    phys_dir = SCRIPT_DIR / "physical_evaluation"
    phys_dir.mkdir(parents=True, exist_ok=True)

    df_phys, df_cv, oof_metrics = evaluate_physical_secugen(model, device)

    df_phys.to_csv(phys_dir / "predictions.csv", index=False)
    df_phys.to_csv(phys_dir / "out_of_fold_predictions.csv", index=False)
    df_cv.to_csv(phys_dir / "cv_fold_results.csv", index=False)

    with open(phys_dir / "physical_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "oof_metrics": oof_metrics,
            "fold_summary": df_cv.to_dict(orient="records")
        }, f, indent=2)

    # Plot OOF Confusion Matrix (Physical)
    fig, ax = plt.subplots(figsize=(6, 5))
    cm_p = np.array([[oof_metrics["tn"], oof_metrics["fp"]], [oof_metrics["fn"], oof_metrics["tp"]]])
    im = ax.imshow(cm_p, cmap="Blues", interpolation="nearest")
    plt.colorbar(im)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted CLEAN", "Predicted DISTORTED"], fontsize=11, fontweight="bold")
    ax.set_yticklabels(["Actual CLEAN", "Actual DISTORTED"], fontsize=11, fontweight="bold")

    thresh_cp = cm_p.max() / 2.0
    for i in range(2):
        for j in range(2):
            val = cm_p[i, j]
            color = "white" if val > thresh_cp else "black"
            ax.text(j, i, f"{val}\n({val/42*100:.1f}%)", ha="center", va="center", color=color, fontsize=13, fontweight="bold")

    ax.set_title(f"Physical SecuGen Out-of-Fold Matrix (N=84)\nOOF Accuracy = {oof_metrics['accuracy']*100:.1f}% | OOF F1 = {oof_metrics['f1_score']*100:.1f}%", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(phys_dir / "oof_confusion_matrix.png", dpi=300)
    plt.close()

    # Plot OOF ROC Curve (Physical)
    fpr_p, tpr_p, _ = roc_curve(df_phys["ground_truth_label"], df_phys["prob_distorted"])
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_p, tpr_p, color="#1f77b4", lw=2.5, label=f"V3 Physical ROC (AUC = {oof_metrics['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5)
    ax.scatter([oof_metrics["fpr"]], [oof_metrics["recall"]], color="red", s=100, zorder=5, label=f"OOF Point (TPR={oof_metrics['recall']:.2f}, FPR={oof_metrics['fpr']:.2f})")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title(f"Out-of-Fold ROC Curve on Physical SecuGen Captures\nAUC = {oof_metrics['roc_auc']:.4f}", fontsize=11, fontweight="bold", pad=12)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(phys_dir / "oof_roc_curve.png", dpi=300)
    plt.close()

    print("[EVAL COMPLETE] All synthetic and physical evaluation artifacts successfully generated!")
    return synth_metrics, oof_metrics

