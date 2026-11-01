"""
Fingerprint Distortion Binary Classifier Training Pipeline (Experiment V3)
===========================================================================
Model: FingerprintCNN (389,057 parameters)
Loss:  BCEWithLogitsLoss (pos_weight=1.0)
Opt:   Adam (lr=1e-4, weight_decay=1e-4)
Data:  data/orientation_capture_variations_v1/ (4,200 Train / 900 Val / 900 Test)
Domain Adaptation: Online Symmetric Photometric & Asymmetric Structural Augmentations
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
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
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


def set_seed(seed: int = 42):
    """Ensure complete deterministic reproducibility."""
    random_seed = seed
    import random
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.50) -> Dict[str, float]:
    """Computes standard biometric classification metrics at threshold."""
    y_pred = (y_probs >= threshold).astype(int)
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

    return {
        "threshold": round(threshold, 4),
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "specificity": round(float(spec), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "roc_auc": round(float(auc), 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }


def find_optimal_validation_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    min_thresh: float = 0.01,
    max_thresh: float = 0.95,
    step: float = 0.01
) -> Tuple[float, Dict[str, float]]:
    """Grid-searches optimal threshold strictly on validation predictions."""
    best_th = 0.50
    best_m = evaluate_metrics(y_true, y_probs, threshold=0.50)
    best_score = best_m["f1_score"]

    for t in np.arange(min_thresh, max_thresh + 1e-5, step):
        t_val = round(float(t), 4)
        m = evaluate_metrics(y_true, y_probs, threshold=t_val)
        if m["f1_score"] > best_score:
            best_score = m["f1_score"]
            best_th = t_val
            best_m = m

    return best_th, best_m


def run_inference_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    """Inference over dataloader without grad."""
    model.eval()
    total_loss = 0.0
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)
            probs = torch.sigmoid(logits)

            total_loss += loss.item() * inputs.size(0)
            all_targets.extend(targets.cpu().numpy().ravel())
            all_probs.extend(probs.cpu().numpy().ravel())

    mean_loss = total_loss / len(loader.dataset)
    return mean_loss, np.array(all_targets), np.array(all_probs)


def train_v3_pipeline(config_path: Path):
    """Executes full training for Experiment V3."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    seed = cfg["dataset"].get("seed", 42)
    set_seed(seed)
    device = get_device()
    print(f"[DEVICE] Training on: {device}")

    out_cfg = cfg["outputs"]
    ckpt_dir = resolve_path(out_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = ckpt_dir / out_cfg.get("best_model_filename", "best_model.pth")
    last_model_path = ckpt_dir / "last_model.pth"
    thresh_path = ckpt_dir / out_cfg.get("threshold_filename", "threshold.json")
    hist_path = ckpt_dir / out_cfg.get("history_filename", "training_history.json")
    loss_plot_path = SCRIPT_DIR / "training_validation_loss.png"

    # DataLoaders
    train_l, val_l, test_l, _ = get_v3_dataloaders(config_path)
    print(f"[DATA] Train: {len(train_l.dataset)} | Val: {len(val_l.dataset)} | Test: {len(test_l.dataset)}")

    # Model
    model = FingerprintCNN(dropout=cfg["model"].get("dropout", 0.5)).to(device)
    print(f"[MODEL] FingerprintCNN initialized: {count_parameters(model):,} parameters")

    train_cfg = cfg["training"]
    lr = train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 1e-4)
    max_epochs = train_cfg.get("max_epochs", 30)
    patience = train_cfg.get("early_stopping_patience", 9)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([train_cfg.get("pos_weight", 1.0)]).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    print("=" * 80)
    print(f"STARTING V3 TRAINING: Max Epochs={max_epochs} | Early Stopping Patience={patience}")
    print("=" * 80)

    best_val_f1 = -1.0
    best_val_metrics = {}
    best_threshold = 0.50
    best_epoch = 0
    patience_counter = 0

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_f1": [],
        "val_accuracy": [],
        "val_roc_auc": [],
        "val_threshold": []
    }

    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_l:
            inputs = batch["image"].to(device)
            targets = batch["label"].to(device).unsqueeze(1)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_l.dataset)

        # Validation evaluation
        val_loss, y_true_val, y_probs_val = run_inference_epoch(model, val_l, criterion, device)

        # Discover optimal validation threshold (strictly on validation set)
        opt_thresh, val_m = find_optimal_validation_threshold(
            y_true_val, y_probs_val,
            min_thresh=cfg["training"]["threshold_tuning"].get("min_threshold", 0.01),
            max_thresh=cfg["training"]["threshold_tuning"].get("max_threshold", 0.95),
            step=cfg["training"]["threshold_tuning"].get("step", 0.01)
        )

        val_f1 = val_m["f1_score"]
        val_acc = val_m["accuracy"]
        val_auc = val_m["roc_auc"]

        history["epoch"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["val_accuracy"].append(round(val_acc, 4))
        history["val_roc_auc"].append(round(val_auc, 4))
        history["val_threshold"].append(round(opt_thresh, 4))

        print(f"Epoch {epoch:2d}/{max_epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} (Thresh={opt_thresh:.2f}) | Val Acc: {val_acc:.4f} | Val AUC: {val_auc:.4f}")

        # Checkpoint if validation F1 improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_metrics = val_m
            best_threshold = opt_thresh
            best_epoch = epoch
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": best_val_f1,
                "threshold": best_threshold,
                "val_metrics": best_val_metrics,
                "config": cfg
            }, best_model_path)
            print(f"  --> Saved BEST model to {best_model_path.name} (Val F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] Validation F1 has not improved for {patience} consecutive epochs.")
                break

    # Save last model
    torch.save({
        "epoch": len(history["epoch"]),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": cfg
    }, last_model_path)
    print(f"[CHECKPOINT] Saved last model to {last_model_path.name}")

    total_time = time.time() - t_start
    print(f"\n[TRAINING COMPLETE] Finished in {total_time:.2f}s. Best Epoch: {best_epoch} (Val F1: {best_val_f1:.4f} at threshold {best_threshold:.4f})")

    # Save threshold.json
    thresh_data = {
        "calibrated_threshold": best_threshold,
        "best_epoch": best_epoch,
        "validation_metrics": best_val_metrics,
        "training_time_seconds": round(total_time, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(thresh_path, "w", encoding="utf-8") as f:
        json.dump(thresh_data, f, indent=2)
    print(f"[OUTPUT] Saved threshold.json to {thresh_path}")

    # Save training_history.json
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[OUTPUT] Saved training_history.json to {hist_path}")

    # Plot training and validation loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = history["epoch"]
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="#d62728", lw=2)
    ax1.scatter([best_epoch], [history["val_loss"][best_epoch - 1]], color="gold", s=80, edgecolors="black", zorder=5, label=f"Best Epoch ({best_epoch})")
    ax1.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax1.set_ylabel("BCE Loss", fontsize=11, fontweight="bold")
    ax1.set_title("Training & Validation Loss Curves", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.plot(epochs_range, history["val_f1"], label="Val F1-Score", color="#2ca02c", lw=2)
    ax2.plot(epochs_range, history["val_accuracy"], label="Val Accuracy", color="#ff7f0e", linestyle="--", lw=2)
    ax2.scatter([best_epoch], [best_val_f1], color="gold", s=80, edgecolors="black", zorder=5, label=f"Best F1: {best_val_f1:.4f}")
    ax2.set_xlabel("Epoch", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Score", fontsize=11, fontweight="bold")
    ax2.set_title("Validation Performance Metrics", fontsize=12, fontweight="bold")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("FingerprintCNN V3 Training History (Domain-Adaptive Experiment)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(loss_plot_path, dpi=300)
    plt.close()
    print(f"[PLOT] Saved training_validation_loss.png to {loss_plot_path}")

    return {
        "status": "TRAINING_COMPLETE",
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "total_time": total_time
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FingerprintCNN V3 Training Pipeline")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    parser.add_argument("--train-now", action="store_true", help="Trigger full training")

    args = parser.parse_args()
    cfg_file = resolve_path(args.config) if Path(args.config).is_file() else SCRIPT_DIR / "config.json"

    if args.train_now:
        train_v3_pipeline(cfg_file)
    else:
        print("Run with --train-now to start training.")

