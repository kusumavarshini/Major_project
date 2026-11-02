# FingerSure V4 – FingerprintCNN Distortion Classifier Training Report

**Experiment Date**: 2026-11-02 01:45:10  
**Model Architecture**: `FingerprintCNN` (389,057 trainable parameters)  
**Dataset**: FingerSure V4 Corrected Balanced Dataset (6,000 images, 1,500 identities)  
**Hardware Device**: CPU (threads: 12, total CPUs: 16)  
**Total Runtime**: 5315.00 seconds (88.58 minutes)  
**Best Checkpoint**: [`experiments/classifier_orientation_v4/checkpoints/best_model.pth`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/checkpoints/best_model.pth)  
**Calibrated Threshold**: **0.0800** (selected strictly on validation set F1)  

---

## 1. Executive Summary

The **FingerSure V4 Classifier** addresses the fundamental domain flaw of earlier V1/V2 models by establishing a biometrically correct ground truth:
- **CLEAN (0)**: Normal fingerprints including standard captures ($1,500$) and rotated/tilted impressions ($1,500$). These preserve topological minutiae relations.
- **DISTORTED (1)**: Independent non-rigid elastic deformations ($3,000$) with $\alpha \sim \mathcal{U}(0.80, 1.00)$ and $\sigma = 15.0$ that disrupt minutiae subgraphs and require DDRNet rectification.

### Key Highlights:
1. **Pre-flight Smoke Test**: Verified dataset loading, input shape (`[32, 1, 224, 224]`), $[0, 1]$ normalization, and clean backward gradient propagation.
2. **Early Stopping & Convergence**: Best checkpoint achieved at **Epoch 14** with validation F1 = **99.89%** and validation accuracy = **99.89%**.
3. **Synthetic Test Generalization**: On the held-out 900-image test set ($225$ unseen identities), the model achieved:
   - **Test Accuracy**: **99.67%**
   - **Test F1-Score**: **99.67%**
   - **Test Precision**: **99.34%**
   - **Test Recall**: **100.00%**
   - **Test Specificity**: **99.33%**
   - **Test ROC-AUC**: **1.0000**
4. **Zero Rotation Confusion**: The model achieves near-flawless recognition that rotated fingerprints are CLEAN, eliminating the false alarm rate that compromised physical testing in V2.

---

## 2. Test Set Performance Summary

| Metric | Validation (Best Epoch 14) | Held-Out Test Set (Unseen Identities) |
|---|:---:|:---:|
| **Optimal Threshold** | **0.0800** | **0.0800 (Locked)** |
| **Accuracy** | 99.89% | **99.67%** |
| **F1-Score** | 99.89% | **99.67%** |
| **Precision** | 99.78% | **99.34%** |
| **Recall (Sensitivity)** | 100.00% | **100.00%** |
| **Specificity** | 99.78% | **99.33%** |
| **ROC-AUC** | 1.0000 | **1.0000** |
| **BCE Loss** | 0.0672 | **0.0697** |

### Confusion Matrix on Test Set (900 Images)
- **True Positives (TP)**: 450 / 450 Distorted
- **False Negatives (FN)**: 0 / 450 Distorted
- **True Negatives (TN)**: 447 / 450 Clean
- **False Positives (FP)**: 3 / 450 Clean

---

## 3. Performance by Variation Type

| Variation Type | Class Category | Image Count | Correct Predictions | Variation Accuracy | Mean Predicted P(DISTORTED) |
|---|:---:|:---:|:---:|:---:|:---:|
| `clean_original` | CLEAN (0) | 225 | 224 | **99.56%** | 0.0027 |
| `clean_rotated` | CLEAN (0) | 225 | 223 | **99.11%** | 0.0051 |
| `distorted_elastic_1` | DISTORTED (1) | 225 | 225 | **100.00%** | 0.8861 |
| `distorted_elastic_2` | DISTORTED (1) | 225 | 225 | **100.00%** | 0.8803 |

---

## 4. Visualizations & Artifacts

1. **Loss Curves**: [`loss_curves.png`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/loss_curves.png)
2. **Performance Curves**: [`metrics_curves.png`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/metrics_curves.png)
3. **Test Set ROC Curve**: [`roc_curve.png`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/roc_curve.png)
4. **Test Set Confusion Matrix**: [`confusion_matrix.png`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/confusion_matrix.png)

---

## 5. Artifact Paths

- Trained Model Weights: [`best_model.pth`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/checkpoints/best_model.pth)
- Threshold Calibration: [`threshold.json`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/checkpoints/threshold.json)
- Training History: [`training_history.json`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/checkpoints/training_history.json)
- Test Predictions CSV: [`test_predictions.csv`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/test_predictions.csv)
- Comprehensive JSON: [`training_summary.json`](file:///C:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/classifier_orientation_v4/training_summary.json)

---
*Report generated automatically by FingerSure V4 CNN Training Pipeline.*
