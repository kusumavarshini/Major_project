# FingerSure – Experiment Plan: Classifier Orientation V3

**Date:** 2026-11-01  
**Target Experiment:** `experiments/classifier_orientation_v3/`  
**Model Architecture:** `FingerprintCNN` (4 convolutional blocks, 389,057 trainable parameters)  
**Objective:** Train an improved binary fingerprint distortion classifier that bridges the synthetic-to-physical domain gap through symmetric photometric adaptation and asymmetric structural capture simulation, achieving robust generalization on real SecuGen Hamster Pro 20 optical captures.

---

## 1. Safety & Boundary Constraints

1. **Zero Overwrite of Existing Assets:**  
   All previous experiment directories (`classifier_orientation_v1`, `classifier_orientation_v2`), model checkpoints (`best_model.pth`, `best_model (4).pth`), DDRNet weights, SourceAFIS galleries, and production backend files (`authentication.py`, `database.py`) are strictly preserved and unmodified.
2. **Strict Physical Dataset Isolation:**  
   The 84 physical SecuGen images in [`FingerSure_Dataset/`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/) **must NEVER be included** in the training or validation sets. They are reserved strictly for post-training participant-level cross-validation and real-world benchmarking.
3. **Training Execution Status:**  
   This document and associated configuration files establish the blueprint. **Training will not commence until explicit user authorization is granted.**

---

## 2. Model Architecture & Pipeline Specifications

Following the quantitative findings of the domain audit, the core model architecture is retained to ensure comparability with V2 and seamless integration with production `authentication.py`:

| Parameter | Specification | Justification |
| :--- | :---: | :--- |
| **Model Class** | `FingerprintCNN` | Custom 4-layer CNN defined in [`model.py`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/model.py) |
| **Input Shape** | `(1, 224, 224)` | Grayscale single-channel float32 normalized to $[0.0, 1.0]$ |
| **Parameter Count** | **389,057** | $1 \to 32 \to 64 \to 128 \to 256 \to \text{Pool} \to \text{Drop}(0.5) \to \text{FC}(1)$ |
| **Output Activation** | Single Logit $z$ | Probability $P(\text{DISTORTED}) = \sigma(z) = 1 / (1 + e^{-z})$ |
| **Target Classes** | Binary: $0$ and $1$ | $\text{CLEAN} = 0$, $\text{DISTORTED} = 1$ |
| **Loss Function** | `BCEWithLogitsLoss` | `pos_weight = 1.0` (unscaled natural gradient loss) |
| **Optimizer** | `Adam` | Learning rate $1 \times 10^{-4}$, Weight decay $1 \times 10^{-4}$ |
| **Batch Size** | `32` | Standard minibatch size |
| **Max Epochs** | `25` | With early stopping patience = 6 |
| **Random Seed** | `42` | Deterministic reproducibility across PyTorch, NumPy, and random |

---

## 3. Dataset Splitting & Leakage Prevention Protocol

### Synthetic Dataset (`data/orientation_capture_variations_v1/`)
- **Total Samples:** 6,000 images across 1,500 unique identities.
- **Components:** 1,500 clean, 1,500 orientation variation, 1,500 partial capture, 1,500 uneven contact.
- **Partitioning Strategy:** Strict subject identity-level 70 : 15 : 15 splitting:
  - **Train Partition (70%):** 1,050 identities $\to$ **4,200 images** (1,050 clean, 3,150 distorted).
  - **Validation Partition (15%):** 225 identities $\to$ **900 images** (225 clean, 675 distorted).
  - **Held-Out Test Partition (15%):** 225 identities $\to$ **900 images** (225 clean, 675 distorted).
- **Identity Leakage Check:**
  $$\text{Train}_{\text{identities}} \cap \text{Val}_{\text{identities}} = \emptyset, \quad \text{Train}_{\text{identities}} \cap \text{Test}_{\text{identities}} = \emptyset, \quad \text{Val}_{\text{identities}} \cap \text{Test}_{\text{identities}} = \emptyset$$

---

## 4. Online Domain-Adaptive Augmentation Pipeline

During training, an online augmentation transform will be applied to minibatches:

```python
class DomainAdaptiveAugmentation:
    def __init__(self, p_photometric=0.6, p_geometric=0.5):
        self.p_photo = p_photometric
        self.p_geom = p_geometric

    def __call__(self, img_tensor, label):
        # 1. Symmetric Photometric Domain Adaptation (Applied to BOTH Clean and Distorted)
        if random.random() < self.p_photo:
            # TIR Ridge Darkening: scale ridge intensity downward towards 15
            img_tensor = apply_tir_darkening(img_tensor, gamma=random.uniform(1.3, 2.0))
            # Laplacian Sharpness Modulation: boost high-frequency ridge edges
            img_tensor = apply_unsharp_mask(img_tensor, amount=random.uniform(0.6, 1.5))
            # Sensor noise injection
            img_tensor = add_gaussian_sensor_noise(img_tensor, std=random.uniform(0.005, 0.015))

        # 2. Aspect-Ratio Vertical Squeeze (Simulate 300x400 -> 224x224 squeeze)
        if random.random() < 0.4:
            img_tensor = apply_anisotropic_squeeze(img_tensor, sy=random.uniform(0.75, 1.05))

        # 3. Asymmetric Morphological Augmentation
        if label == 0:  # CLEAN
            # Mild natural alignment jitter only ([-5 deg, +5 deg])
            img_tensor = apply_affine_rotation(img_tensor, angle=random.uniform(-5, 5))
        else:           # DISTORTED
            # Enhanced angular tilt / heavy sideways roll ([-35 deg, -12 deg] U [+12 deg, +35 deg])
            if random.random() < self.p_geom:
                img_tensor = apply_affine_rotation(img_tensor, angle=random.choice([-1, 1]) * random.uniform(12, 35))
            # Pressure falloff / uneven contact mask
            if random.random() < 0.3:
                img_tensor = apply_elliptical_contact_mask(img_tensor)

        return img_tensor
```

---

## 5. Threshold Selection & Model Checkpointing

1. **Validation Threshold Tuning:**  
   - At the conclusion of each epoch, the model is evaluated on the 900 unseen synthetic validation images.
   - An optimal operating threshold $t_{\text{opt}} \in [0.01, 0.95]$ is discovered via grid search targeting validation F1-score (`f1_score`).
2. **Best Model Checkpointing:**  
   - A checkpoint is saved to `experiments/classifier_orientation_v3/checkpoints/best_model.pth` strictly when validation F1 improves (`val_f1 > best_val_f1`).
   - The corresponding calibrated threshold is saved to `threshold.json`.
3. **Held-Out Synthetic Test Evaluation:**  
   - Once training terminates (via early stopping or 25 epochs), the 900 held-out synthetic test images are evaluated with the locked threshold to report in-distribution performance.

---

## 6. Real-World Evaluation Protocol (SecuGen Physical Dataset)

Following model training, evaluation will be executed on the physical SecuGen dataset using the exact same participant-level framework established in V2:

1. **Participant-Grouped Cross-Validation (StratifiedGroupKFold, 7 Folds):**  
   - 21 participants, 84 impressions.
   - Zero participant overlap between calibration and validation folds.
   - Measures out-of-fold accuracy, precision, recall, F1-score, specificity, and ROC-AUC.
2. **Target Benchmark Comparison:**  
   - Direct comparison against V2 baseline metrics:
     - Target Physical ROC-AUC: $> 0.8500$ (V2 was $0.7517$).
     - Target Physical Out-of-Fold Accuracy: $> 75.00\%$ (V2 was $65.48\%$).
     - Target Threshold 0.50 Distortion Recall: $> 50.00\%$ (V2 was $0.00\%$).
   - Confirming whether domain adaptation successfully shifts the physical prediction probabilities into the $[0.40, 0.90]$ range for distorted prints.

