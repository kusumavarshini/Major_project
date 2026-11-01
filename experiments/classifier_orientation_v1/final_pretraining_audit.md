# FingerSure – Final Pre-Training Audit Report: Classifier Orientation v1

**Date:** 2026-11-01  
**Target Pipeline:** `experiments/classifier_orientation_v1/`  
**Model Architecture:** `FingerprintCNN` (389,057 trainable parameters)  
**Dataset:** `data/orientation_capture_variations_v1/` (6,000 images, 1,500 identities)  
**Audit Status:** **ALL 8 PRE-TRAINING CHECKS PASSED (100% COMPLIANCE)**  
**Training Execution Status:** **HALTED / AWAITING EXPLICIT USER APPROVAL (0 Epochs Run)**

---

## Executive Summary

A comprehensive pre-training verification audit of the `experiments/classifier_orientation_v1/` pipeline was conducted. All eight criteria established for architectural validity, data integrity, leakage prevention, loss and hyperparameter configuration, production compatibility, and environment safety have been verified deterministically.

**Zero training has been executed.** The checkpoint directory `experiments/classifier_orientation_v1/checkpoints/` remains empty. All production checkpoints (`classifier_fvc_adapted/best_model.pth`, `ddrnet_fvc_finetuned/best_model.pth`) and production source files (`authentication.py`, `database.py`, `backend_api.py`) remain completely untouched.

| Check # | Audit Item | Required Specification | Measured / Audited Result | Status |
| :---: | :--- | :--- | :--- | :---: |
| **1** | Preprocessing Parity | Pixel-by-pixel match between training and inference | Max absolute difference = `0.00000000` across all test images | **PASS** |
| **2** | Binary Class Mapping | CLEAN = 0, DISTORTED = 1 | `clean`: 0, `distorted`: 1, sigmoid represents $P(\text{DISTORTED})$ | **PASS** |
| **3** | Class Imbalance Handling | Handle 1:3 ratio appropriately | 1,050:3,150 (1:3.0) with validation threshold tuning over $[0.05, 0.95]$ | **PASS** |
| **4** | Hyperparameter Integrity | Adam, LR 1e-4, WD 1e-4, BS 32, Epochs 25, Patience 6, Seed 42 | Configured and matched across `config.json` and `train.py` | **PASS** |
| **5** | Best Model & Threshold Selection | Gated strictly by validation F1; threshold tuned on validation data | `val_f1 > best_val_f1` gates checkpointing; grid search on `val_l` | **PASS** |
| **6** | Held-Out Test Set Isolation | Test set isolated from training and threshold tuning | Post-training evaluation only with locked threshold; 0 stem leakage | **PASS** |
| **7** | Production Checkpoint Integrity | Production classifier and DDRNet untouched | Classifier (1,570,229 B) and DDRNet (180,395,816 B) SHA256 verified | **PASS** |
| **8** | Production Model Loading | Loadable by existing `FingerprintCNN` in `authentication.py` | `load_state_dict(strict=True)` 30/30 keys matched, logits `(1, 1)` | **PASS** |

---

## Detailed Audit Results

### Check 1: Pixel-by-Pixel Preprocessing Parity

#### Objective
Verify that the image preprocessing pipeline implemented in `experiments/classifier_orientation_v1/dataset.py` produces tensors that are mathematically identical (down to $0.0$ floating point difference) to the production inference preprocessing in `authentication.py:classify_distortion`.

#### Pipeline Comparison
* **Training Pipeline (`dataset.py`)**:
  1. `cv2.imread(path, cv2.IMREAD_GRAYSCALE)`
  2. If resolution $\neq (224, 224)$, resize via `cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)`.
  3. Cast to `float32` and normalize: `img.astype(np.float32) / 255.0`.
  4. Convert to PyTorch Tensor: `torch.from_numpy(img_norm).unsqueeze(0)` $\to$ Shape `(1, 224, 224)`.
* **Production Pipeline (`authentication.py:154-173`)**:
  1. Grayscale check: `if image.ndim == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`.
  2. Resize: `cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)`.
  3. Cast to `float32` and normalize: `image_float = image_224.astype(np.float32) / 255.0`.
  4. Convert to PyTorch Tensor: `torch.from_numpy(image_float).unsqueeze(0).unsqueeze(0)` $\to$ Shape `(1, 1, 224, 224)`.

#### Quantitative Verification
Six representative images were evaluated through both preprocessing routines:
1. `FM001_C1_clean.png` (Synthetic Clean, raw shape $224 \times 224$):
   * Max Absolute Difference: `0.00000000` | Mean Absolute Difference: `0.00000000` | Exact Match: **True**
2. `FM001_C1_orientation.png` (Synthetic Orientation Variation, raw shape $224 \times 224$):
   * Max Absolute Difference: `0.00000000` | Mean Absolute Difference: `0.00000000` | Exact Match: **True**
3. `FM001_C1_partial.png` (Synthetic Partial Capture, raw shape $224 \times 224$):
   * Max Absolute Difference: `0.00000000` | Mean Absolute Difference: `0.00000000` | Exact Match: **True**
4. `FM001_C1_uneven.png` (Synthetic Uneven Contact, raw shape $224 \times 224$):
   * Max Absolute Difference: `0.00000000` | Mean Absolute Difference: `0.00000000` | Exact Match: **True**
5. `A_LEFT_THUMB_clean.png` (Physical SecuGen Clean, raw shape $400 \times 300$):
   * Max Absolute Difference: `0.00000000` | Mean Absolute Difference: `0.00000000` | Exact Match: **True**
6. `A_LEFT_THUMB_distorted.png` (Physical SecuGen Distorted, raw shape $400 \times 300$):
   * Max Absolute Difference: `0.00000000` | Mean Absolute Difference: `0.00000000` | Exact Match: **True**

**Verdict: PASS**

---

### Check 2: Binary Class Mapping (CLEAN = 0, DISTORTED = 1)

#### Objective
Confirm that the label assignment across the configuration, split manifests, model outputs, and production inference follows the strict contract: `CLEAN = 0` and `DISTORTED = 1`.

#### Verification
1. **Configuration (`config.json`)**:
   ```json
   "classes": {
     "clean": 0,
     "distorted": 1,
     "distorted_components": [
       "orientation_variation",
       "partial_capture",
       "uneven_contact"
     ]
   }
   ```
2. **Split Manifest Audit (`train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`)**:
   * `train`: Clean rows = 1,050 (unique labels: `[0]`, variations: `['clean']`), Distorted rows = 3,150 (unique labels: `[1]`, variations: `['orientation_variation', 'partial_capture', 'uneven_contact']`).
   * `val`: Clean rows = 225 (unique labels: `[0]`), Distorted rows = 675 (unique labels: `[1]`).
   * `test`: Clean rows = 225 (unique labels: `[0]`), Distorted rows = 675 (unique labels: `[1]`).
   * Zero unmapped, null, or inverted labels exist.
3. **Logits and Probability Semantics**:
   * Loss: `nn.BCEWithLogitsLoss()`. The target label $y \in \{0.0, 1.0\}$.
   * Model output: Logit $z$.
   * $\sigma(z) = \frac{1}{1 + e^{-z}} = P(\text{DISTORTED})$.
   * $1 - \sigma(z) = P(\text{CLEAN})$.
   * In `authentication.py:182-198`:
     ```python
     prob_dist = float(torch.sigmoid(logits).item())
     prob_clean = float(1.0 - prob_dist)
     is_distorted = bool(prob_dist >= threshold)
     pred_class = "DISTORTED" if is_distorted else "CLEAN"
     ```
   * Full semantic and numerical agreement confirmed.

**Verdict: PASS**

---

### Check 3: Class Imbalance Handling (1:3 Ratio)

#### Objective
Confirm that the 1:3 class imbalance (1 clean : 3 distorted impressions per identity) is handled systematically without creating systematic bias against the clean class.

#### Verification & Analysis
1. **Class Distribution**:
   * Total dataset: 6,000 images across 1,500 identities.
   * `train`: 1,050 clean (25.0%) vs 3,150 distorted (75.0%) $\to$ Exact 1:3.0 ratio.
   * `val`: 225 clean (25.0%) vs 675 distorted (75.0%) $\to$ Exact 1:3.0 ratio.
   * `test`: 225 clean (25.0%) vs 675 distorted (75.0%) $\to$ Exact 1:3.0 ratio.
2. **Strategy Configured**:
   * **Loss Function Configuration**: `pos_weight = 1.0` in `config.json` (`training.pos_weight`). In `BCEWithLogitsLoss(pos_weight=pos_weight)`, `pos_weight` weights the positive class (class 1, DISTORTED). Because distorted is the majority class (75%), setting `pos_weight > 1.0` would undesirably amplify the majority class. Leaving `pos_weight = 1.0` maintains natural unscaled loss gradients.
   * **Decision Boundary Calibration**: Rather than artificially skewing training gradients, `train.py` implements **post-hoc empirical validation threshold optimization**:
     * Grid-searches candidate thresholds $t \in [0.05, 0.95]$ with step $0.01$.
     * Evaluates F1 score, specificity, balanced accuracy, precision, and recall on the validation set.
     * Selects the threshold maximizing validation F1 score (`f1_score`).
     * Compensates directly for the 75% positive prior, ensuring clean fingerprints are not falsely flagged as distorted.

**Verdict: PASS**

---

### Check 4: Hyperparameters Specification

#### Objective
Verify that all training hyperparameters are set to the exact requested specifications across `config.json` and `train.py`.

#### Measured Parameter Audit
| Parameter | Requested Value | `config.json` Value | `train.py` Value | Result |
| :--- | :---: | :---: | :---: | :---: |
| **Optimizer** | `Adam` | `"Adam"` | `torch.optim.Adam` | Matched |
| **Learning Rate** | $0.0001$ ($10^{-4}$) | `0.0001` | `lr = 0.0001` | Matched |
| **Weight Decay** | $0.0001$ ($10^{-4}$) | `0.0001` | `weight_decay = 0.0001` | Matched |
| **Batch Size** | $32$ | `32` | `batch_size = 32` | Matched |
| **Maximum Epochs** | $25$ | `25` | `max_epochs = 25` | Matched |
| **Early Stopping Patience** | $6$ | `6` | `patience = 6` | Matched |
| **Random Seed** | $42$ | `42` | `set_seed(42)` | Matched |

#### Determinism Audit
`set_seed(42)` in `train.py` sets:
* `np.random.seed(42)`
* `torch.manual_seed(42)`
* `torch.cuda.manual_seed_all(42)`
* `torch.backends.cudnn.deterministic = True`
* `torch.backends.cudnn.benchmark = False`

**Verdict: PASS**

---

### Check 5: Best Model Selection & Validation Threshold Tuning

#### Objective
Confirm that best-model selection uses validation F1 and that the classification threshold is selected strictly from validation data.

#### Code Audit in `train.py`
1. **Validation Metric Evaluation (lines 349-365)**:
   ```python
   val_loss, y_true_val, y_probs_val = run_inference_epoch(model, val_l, criterion, device)
   opt_thresh, val_metrics = find_optimal_threshold(
       y_true=y_true_val,
       y_probs=y_probs_val,
       min_thresh=th_cfg.get("min_threshold", 0.05),
       max_thresh=th_cfg.get("max_threshold", 0.95),
       step=th_cfg.get("step", 0.01),
       target_metric="f1_score"
   )
   val_f1 = val_metrics["f1_score"]
   ```
2. **Gated Checkpoint Saving (lines 376-393)**:
   ```python
   if val_f1 > best_val_f1:
       best_val_f1 = val_f1
       best_val_metrics = val_metrics
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
   ```
3. **Threshold Serialization**:
   * The calibrated threshold `best_threshold` is exported to `threshold.json`.
   * Model selection is strictly gated by `val_f1 > best_val_f1`.
   * Neither test data nor training loss is used to select the threshold or best checkpoint.

**Verdict: PASS**

---

### Check 6: Held-Out Test Set Strict Isolation

#### Objective
Confirm that the held-out test set is completely isolated and never used for threshold selection, model selection, or gradient updates.

#### Verification
1. **Temporal Isolation in `train.py`**:
   * The test loader `test_l` is only referenced at line 429, strictly **after** the training loop (`for epoch in range(1, max_epochs + 1):`) has ended and early stopping has resolved.
   * Line 425 reloads `best_model.pth`.
   * Line 430 executes `evaluate_predictions(y_true_test, y_probs_test, threshold=best_threshold)`. The threshold applied is strictly the locked `best_threshold` determined from the validation set.
2. **Zero-Leakage Identity Split**:
   * Identity partitioning was performed at the subject stem level (e.g., `FM001_C1`), ensuring all four impressions (clean, orientation, partial, uneven) of any identity are restricted to exactly one partition:
     * Train Stems: 1,050 identities (4,200 images)
     * Validation Stems: 225 identities (900 images)
     * Test Stems: 225 identities (900 images)
   * Set Intersections:
     * $\text{Train} \cap \text{Val} = \emptyset$ (0 stems)
     * $\text{Train} \cap \text{Test} = \emptyset$ (0 stems)
     * $\text{Val} \cap \text{Test} = \emptyset$ (0 stems)
   * Leakage Count: **0 / 1,500 identities**.

**Verdict: PASS**

---

### Check 7: Production Checkpoints Integrity Audit

#### Objective
Verify that all existing production model checkpoints, DDRNet weights, and production source code remain untouched and unmodified.

#### Checkpoint Integrity Inspection
1. **Production Classifier Checkpoint**:
   * Path: `experiments/classifier_fvc_adapted/best_model.pth`
   * File Size: `1,570,229` bytes
   * SHA-256 Hash: `f01f301ad3c7eb50ca9fcf53c150c9bfe4129548aa1fb4e815e96645579f1df9`
   * Status: **UNTOUCHED / UNMODIFIED**
2. **Production DDRNet Distortion Rectification Checkpoint**:
   * Path: `experiments/ddrnet_fvc_finetuned/best_model.pth`
   * File Size: `180,395,816` bytes
   * SHA-256 Hash: `92270708eb5cb883b1bfbc7c1d7637841c19b6716ee454e99f0f9b17c6031737`
   * Status: **UNTOUCHED / UNMODIFIED**
3. **Pipeline Isolation**:
   * `experiments/classifier_orientation_v1/checkpoints/`: **0 checkpoint files present** (clean state).
   * Production core files (`authentication.py`, `database.py`, `matcher.py`, `backend_api.py`): **UNTOUCHED / UNMODIFIED**.

**Verdict: PASS**

---

### Check 8: FingerprintCNN Inference Implementation Compatibility

#### Objective
Confirm that weights produced by `classifier_orientation_v1/train.py` can be seamlessly loaded by the existing `FingerprintCNN` inference engine in `authentication.py`.

#### Architectural Inspection & Load Simulation
1. **Architecture Match**:
   * Both use `FingerprintCNN` defined in `model.py`.
   * Trainable parameters: exactly **389,057**.
   * Layer breakdown:
     * `features.0`: `Conv2d(1, 32, kernel_size=3, padding=1)`
     * `features.1`: `BatchNorm2d(32)`
     * `features.4`: `Conv2d(32, 64, kernel_size=3, padding=1)`
     * `features.5`: `BatchNorm2d(64)`
     * `features.8`: `Conv2d(64, 128, kernel_size=3, padding=1)`
     * `features.9`: `BatchNorm2d(128)`
     * `features.12`: `Conv2d(128, 256, kernel_size=3, padding=1)`
     * `features.13`: `BatchNorm2d(256)`
     * `pool`: `AdaptiveAvgPool2d(output_size=1)`
     * `dropout`: `Dropout(p=0.5)`
     * `fc`: `Linear(in_features=256, out_features=1, bias=True)`
2. **Loader Verification**:
   * `authentication.py:load_classifier()` expects:
     ```python
     state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
     model.load_state_dict(state_dict)
     ```
   * Simulated checkpoint saved in `train.py` format was tested against a clean `FingerprintCNN()` instance:
     * Total keys evaluated: **30 / 30**
     * Missing keys: **0**
     * Unexpected keys: **0**
     * Load status: `strict=True` **PASSED with 100% key match**.
3. **Inference Execution Verification**:
   * Forward pass with input tensor `(1, 1, 224, 224)` produced logit output of shape `torch.Size([1, 1])`.
   * Sigmoid transformation yielded valid probabilities $P(\text{DISTORTED}) \in [0.0, 1.0]$ and $P(\text{CLEAN}) = 1.0 - P(\text{DISTORTED})$.
   * Zero code changes are required in `authentication.py` other than pointing `CLASSIFIER_CHECKPOINT` and `CLASSIFIER_THRESHOLD` to the newly trained outputs.

**Verdict: PASS**

---

## Conclusion & Readiness Assessment

The pre-training audit of `experiments/classifier_orientation_v1/` has concluded with **PASS** across all eight checks.

* **Data integrity**: 6,000 images, 1,500 identities, 0 identity leakage.
* **Model architecture**: Native 389,057-parameter `FingerprintCNN`.
* **Preprocessing**: 100% pixel-level parity with production inference.
* **Separation**: Complete isolation of the held-out test set; model and threshold selection strictly governed by validation F1.
* **Safety**: All production models and database pipelines remain completely untouched.

**The training script is in a locked, ready state and awaits explicit user authorization to begin.**

