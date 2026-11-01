# FingerprintCNN Distortion Classifier (Orientation v1 Pipeline)

**Pipeline Path:** [`experiments/classifier_orientation_v1/`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v1/)  
**Model Architecture:** `FingerprintCNN` (exactly 389,057 parameters)  
**Dataset:** [`data/orientation_capture_variations_v1/`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/data/orientation_capture_variations_v1/) (6,000 images across 1,500 identities)  
**Task:** Binary Fingerprint Distortion Classification (CLEAN = 0, DISTORTED = 1)  

---

## 1. Pipeline Overview & Objectives

This pipeline establishes a clean, decoupled training workflow for the FingerSure CNN distortion classifier. It trains `FingerprintCNN` on the newly generated **Orientation and Physical Capture Variations Dataset (v1)** rather than artificial mathematical displacement grids or spline warping fields.

### Key Objectives:
1. **Physical Sensor Generalization**: Train on real-world capture variations (rotation angles $\pm 8^\circ$ to $\pm 44^\circ$, organic partial contact, and non-uniform optical pressure fading) modeled after the **SecuGen Hamster Pro 20** scanner.
2. **Elimination of Bilinear Blur Shortcut**: Use bicubic-resampled orientations and contrast-attenuated ridges so the model learns genuine orientation and contact degradation rather than subpixel interpolation blur.
3. **Identity-Level Data Leakage Prevention**: 70:15:15 partitioning performed strictly on original fingerprint identity stems. All variations of the same finger remain within the same split partition.
4. **Complete DDRNet Decoupling**: Fully verified that CNN classifier inference and DDRNet geometric rectification operate as independent stages. Updating the classifier requires **zero changes to DDRNet**.

---

## 2. Model Architecture & Preprocessing

### FingerprintCNN Architecture
The model reuses the established 4-layer convolutional architecture from [`model.py`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/model.py):

```
Input: (B, 1, 224, 224) Grayscale float32 in [0.0, 1.0]
  │
  ├─ Conv2D(1 → 32, k=3, p=1)   → BatchNorm2D → ReLU → MaxPool2D(2, 2)  → (B, 32, 112, 112)
  ├─ Conv2D(32 → 64, k=3, p=1)  → BatchNorm2D → ReLU → MaxPool2D(2, 2)  → (B, 64, 56, 56)
  ├─ Conv2D(64 → 128, k=3, p=1) → BatchNorm2D → ReLU → MaxPool2D(2, 2)  → (B, 128, 28, 28)
  ├─ Conv2D(128 → 256, k=3, p=1)→ BatchNorm2D → ReLU                    → (B, 256, 28, 28)
  │
  ├─ AdaptiveAvgPool2D((1, 1))                                          → (B, 256, 1, 1)
  ├─ Dropout(p=0.5)                                                     → (B, 256)
  └─ Linear(256 → 1)                                                    → (B, 1) Logit

Total Trainable Parameters: 389,057
```

### Preprocessing Compatibility Matrix

| Pipeline Stage | Input Format | Resolution | Interpolation | Normalization | Tensor Dimensions |
|---|---|---|---|---|---|
| **CNN Training** | Grayscale PNG | $224 \times 224$ | Identity / `INTER_AREA` | Float32 $\in [0.0, 1.0]$ (`/ 255.0`) | `(B, 1, 224, 224)` |
| **CNN Production Inference** ([`authentication.py:160-173`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/authentication.py#L160-L173)) | Grayscale | Any (e.g. $400 \times 300$) | `cv2.INTER_AREA` $\rightarrow 224 \times 224$ | Float32 $\in [0.0, 1.0]$ (`/ 255.0`) | `(1, 1, 224, 224)` |
| **DDRNet Production Inference** ([`authentication.py:334-380`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/authentication.py#L334-L380)) | Original Grayscale | Native resolution ($orig\_h, orig\_w$) | `cv2.INTER_AREA` $\rightarrow 224 \times 224$ (temp) | Inverted: `(255.0 - gray) / 255.0` | `x`: `(1, 1, 224, 224)`, `mask`: `(1, 1, 14, 14)` |

> [!NOTE]
> **DDRNet Compatibility Verified**: DDRNet operates independently via `rectify_with_ddrnet(original_image)`. The CNN tensor is never passed to DDRNet. DDRNet generates a $14 \times 14$ displacement field, upsamples it to the original resolution, and remaps the **original full-resolution input image**. Replacing the CNN classifier introduces zero side effects to DDRNet.

---

## 3. Dataset Composition & Split Guarantees

- **Dataset Root:** `data/orientation_capture_variations_v1/`
- **Total Unique Identities:** 1,500 identities (`FM001_C1` through `FM100_M5`).
- **Class Mapping:**
  - Class 0: `clean` (1 impression per identity)
  - Class 1: `distorted` (3 impressions per identity: orientation, partial-contact, uneven-contact)
- **Identity-Level Partitioning:**
  - **Train Split (70%)**: 1,050 identities $\rightarrow$ 1,050 clean + 3,150 distorted = **4,200 images** (132 batches)
  - **Validation Split (15%)**: 225 identities $\rightarrow$ 225 clean + 675 distorted = **900 images** (29 batches)
  - **Held-Out Test Split (15%)**: 225 identities $\rightarrow$ 225 clean + 675 distorted = **900 images** (29 batches)
  - **Total Images:** **6,000 images**
- **Data Leakage Guarantees:**
  - Strict disjoint sets: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
  - Partition manifests saved in [`splits/`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v1/splits/).

---

## 4. Pipeline Configuration & Training Settings

Configured in [`config.json`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v1/config.json):
- **Batch Size:** 32
- **Optimizer:** `Adam` (learning rate: $1 \times 10^{-4}$, weight decay: $1 \times 10^{-4}$)
- **Loss Function:** `BCEWithLogitsLoss`
- **Max Epochs:** 25
- **Early Stopping:** 6 epochs patience based on validation F1 score
- **Threshold Calibration:** Grid search on validation set across $[0.05, 0.95]$ with step $0.01$ (91 candidate thresholds) maximizing F1-score
- **Evaluation Metrics:** Loss, Accuracy, Precision, Recall, F1-Score, Specificity, Balanced Accuracy, ROC-AUC, and Confusion Matrix

---

## 5. File Structure

```text
experiments/classifier_orientation_v1/
├── config.json               # Full pipeline hyperparameters and paths
├── dataset.py               # PyTorch Dataset, DataLoaders, and zero-leakage split manager
├── train.py                 # Training, validation, metrics, threshold search, and dry-run CLI
├── README.md                # Technical documentation and compatibility specifications
├── splits/                  # Generated partition manifests
│   ├── train_manifest.csv   # 4,200 rows (1,050 identities)
│   ├── val_manifest.csv     # 900 rows (225 identities)
│   └── test_manifest.csv    # 900 rows (225 identities)
└── checkpoints/             # Output directory (populated upon approved training)
    ├── best_model.pth       # Highest validation F1 checkpoint
    ├── threshold.json       # Calibrated decision threshold and validation metrics
    ├── training_history.json# Per-epoch loss, accuracy, and F1 logs
    ├── test_results.json    # Final evaluation on locked held-out test split
    └── test_confusion_matrix.json
```

---

## 6. How to Run

### Pre-Flight Dry Run (No Training, Safe)
Validates dataset indexing, split partitioning, batch loading, forward pass, and gradient backprop:
```powershell
python experiments/classifier_orientation_v1/train.py --dry-run
```

### Full Training Execution (Requires User Approval)
```powershell
python experiments/classifier_orientation_v1/train.py --train-now
```

---

## 7. Approval Gate & Strict Restrictions

Per instructions:
- **Pipeline creation and dry-run validation are COMPLETE.**
- **No training epochs have been executed.**
- **No checkpoints have been modified or overwritten.**
- **Awaiting user explicit approval before executing training.**

