# FingerSure – Augmentation Design Specification: Classifier V3 Domain Adaptation

**Document Date:** 2026-11-01  
**Target Experiment:** `experiments/classifier_orientation_v3/`  
**Purpose:** Formulate a principled, photometrically accurate data augmentation pipeline to close the synthetic-to-physical domain gap between training data (`data/orientation_capture_variations_v1/`) and optical SecuGen captures ([`FingerSure_Dataset/`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/FingerSure_Dataset/)).

---

## 1. Augmentation Philosophy & Core Safety Principles

To ensure that augmentations resolve the domain gap rather than introducing new shortcut artifacts or degrading recognition accuracy:

1. **Symmetric Photometric Adaptation (Domain Shift):**  
   Sensory and optical differences (e.g. Total Internal Reflection ridge darkness, optical glass background saturation, Laplacian edge sharpness, sensor contrast) are sensor-level phenomena that affect **both CLEAN and DISTORTED** physical impressions. Therefore, photometric augmentations must be applied **symmetrically to both classes** so the CNN does not learn to equate darkness or sharpness with a classification label.
2. **Asymmetric Structural Augmentation (Class Identity):**  
   Morphological and spatial transformations that alter capture geometry (severe rotation $>15^\circ$, sideways displacement, partial border clipping, localized pressure falloff) represent true physical capture distortion and must **strictly remain exclusive to the DISTORTED class** (or be used as component generators for class 1).
3. **Preservation of Ridge Continuity:**  
   No transformation may break, dissolve, or hallucinate minutiae or ridges. Elastic deformation must remain distinct from random affine jitter, and morphological filters must preserve fingerprint topology.
4. **Binary Label Integrity:**  
   The binary classification objective ($\text{CLEAN} = 0$, $\text{DISTORTED} = 1$) is strictly preserved.

---

## 2. Detailed Augmentation Catalogue

Below is the complete specification of proposed training-only augmentations designed from the quantitative findings of the domain audit:

---

### Augmentation 1: Optical TIR Ridge Darkening & Dynamic Range Stretching

* **Observed Physical Characteristic Addressed:**  
  Physical SecuGen impressions exhibit an average 5th-percentile ridge darkness of **$12.14$** (near solid black), whereas synthetic training images average **$115.83$** (soft gray). Dynamic range on physical glass is $252.7$ vs $163.7$ in synthetic data.
* **Mathematical Implementation:**  
  Piecewise non-linear remapping applied to pixels below background threshold ($I < 220$):
  $$I_{\text{new}} = \begin{cases} 
  I_{\text{bg}} & \text{if } I \ge I_{\text{bg\_thresh}} \\
  I \cdot \left(\frac{I - I_{\min}}{I_{\text{bg\_thresh}} - I_{\min}}\right)^{\gamma_{\text{ridge}}} \cdot \alpha_{\text{scale}} & \text{if } I < I_{\text{bg\_thresh}}
  \end{cases}$$
  where $\gamma_{\text{ridge}} \in [1.3, 2.2]$ (stretching midtones downward) and $\alpha_{\text{scale}}$ pulls minimum ridge intensity into $[0, 30]$.
* **Target Class Assignment:**  
  **BOTH CLASSES (Symmetric: 50% probability on Clean, 50% on Distorted).**
* **Preservation of Structure & Label:**  
  The mapping is strictly monotonic. Ridge peaks, valleys, and minutiae locations remain mathematically unchanged. Because both clean and distorted samples experience this darkening, the CNN cannot associate dark ridges with distortion; instead, it is forced to inspect ridge flow and orientation.

---

### Augmentation 2: Laplacian Edge Gradient & Sharpness Modulation

* **Observed Physical Characteristic Addressed:**  
  Physical SecuGen captures exhibit a Laplacian variance of **$10,400 - 11,500$** ($28\times$ higher than synthetic data at $380 - 400$) due to optical lens focusing and TIR surface transitions.
* **Mathematical Implementation:**  
  Unsharp masking filter combined with bounded Laplacian edge boosting:
  $$I_{\text{sharp}} = \text{clip}\left(I + \beta \cdot (I - G_\sigma * I), 0, 255\right)$$
  where $G_\sigma$ is a Gaussian blur kernel with $\sigma \in [0.8, 1.5]$ and boosting coefficient $\beta \in [0.5, 1.8]$.
* **Target Class Assignment:**  
  **BOTH CLASSES (Symmetric: 40% probability on Clean, 40% on Distorted).**
* **Preservation of Structure & Label:**  
  Enhances high-frequency ridge boundaries without introducing synthetic ringing or phase shift. Exposing the CNN to high-sharpness training images prevents physical prints from overwhelming early convolutional feature maps.

---

### Augmentation 3: Sensor Background Uniformity & Glass Bleach

* **Observed Physical Characteristic Addressed:**  
  Physical SecuGen background is pure optical glass ($238.5 \pm 2.2$) with near-zero texture variance outside the contact zone. Synthetic training images occasionally contain paper grain or scanning artifacts in the margins.
* **Mathematical Implementation:**  
  Thresholded background clipping:
  $$I_{\text{clean\_bg}} = \begin{cases} 
  \mathcal{N}(240, 2.0) & \text{if } I > 218 \\
  I & \text{otherwise}
  \end{cases}$$
* **Target Class Assignment:**  
  **BOTH CLASSES (Symmetric: 30% probability).**
* **Preservation of Structure & Label:**  
  Eliminates spurious peripheral noise, ensuring the network does not use border gradients to discriminate classes.

---

### Augmentation 4: Aspect-Ratio Squeeze & Contact Scale Simulation

* **Observed Physical Characteristic Addressed:**  
  SecuGen Hamster Pro 20 captures have a native $300 \times 400$ resolution (aspect ratio $0.75$). Resizing to $224 \times 224$ via `INTER_AREA` compresses the vertical axis by $0.75\times$, causing physical ridge spacing to compress vertically by $25\%$. Furthermore, human thumb contact occupies $52.4\%$ of the frame vs $30.0\%$ in synthetic data.
* **Mathematical Implementation:**  
  Random anisotropic scaling along the vertical axis by factor $s_y \in [0.75, 1.10]$ combined with random isotropic zoom $s_{\text{zoom}} \in [0.95, 1.35]$ centered on the fingerprint core.
* **Target Class Assignment:**  
  **BOTH CLASSES (Symmetric: 40% probability on Clean, 40% on Distorted).**
* **Preservation of Structure & Label:**  
  Enables the convolutional filters to learn spatial invariance across both natural $1:1$ ridge frequencies and the vertically squeezed ridge frequencies created by `INTER_AREA` downsampling.

---

### Augmentation 5: Mild Optical Sensor Additive Gaussian & Shot Noise

* **Observed Physical Characteristic Addressed:**  
  SecuGen optical CMOS sensors introduce subtle sensor shot noise in low-light ridge troughs ($I < 50$).
* **Mathematical Implementation:**  
  Additive zero-mean Gaussian noise $\eta \sim \mathcal{N}(0, \sigma_{\text{sensor}}^2)$ with $\sigma_{\text{sensor}} \in [1.5, 4.0]$ clipped to valid uint8 range $[0, 255]$.
* **Target Class Assignment:**  
  **BOTH CLASSES (Symmetric: 25% probability).**
* **Preservation of Structure & Label:**  
  $\sigma_{\text{sensor}} \le 4.0$ is small relative to the valley-to-ridge dynamic range ($\Delta I \approx 150$), acting as a regularizer without corrupting minutiae.

---

### Augmentation 6: In-Plane Rotation & Angular Tilt Augmentation

* **Observed Physical Characteristic Addressed:**  
  In the physical dataset, 33 of the 42 distorted samples involve substantial in-plane finger rotation, sideways alignment, or angular tilt against the scanner guide (angles up to $\pm 35^\circ$).
* **Mathematical Implementation:**  
  Affine rotation matrix centered at the fingerprint centroid:
  $$\mathbf{R}_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}, \quad \theta \in [-35^\circ, -12^\circ] \cup [+12^\circ, +35^\circ]$$
  For clean prints, only mild natural alignment jitter is allowed: $\theta_{\text{clean}} \in [-5^\circ, +5^\circ]$.
* **Target Class Assignment:**  
  **ASYMMETRIC:**
  - **Clean Class:** Restricted strictly to $\theta \in [-5^\circ, +5^\circ]$ (natural upright placement).
  - **Distorted Class:** Heavy rotation $\theta \in [12^\circ, 35^\circ]$ or $[-35^\circ, -12^\circ]$.
* **Preservation of Structure & Label:**  
  Fingerprints tilted beyond $12^\circ$ violate standard biometric capture protocols and force feature extraction errors in unrectified matching; designating $|\theta| > 12^\circ$ as DISTORTED reflects real-world operational guidelines.

---

### Augmentation 7: Non-Uniform Pressure & Elliptical Contact Falloff

* **Observed Physical Characteristic Addressed:**  
  Physical impressions with light or uneven touch (e.g. `manal`, `Mangalamma`, `manya`, `Lakshmi`) exhibit uneven contact boundaries where ridge contrast drops smoothly toward the platen perimeter.
* **Mathematical Implementation:**  
  Multiplication by an off-center elliptical smooth mask $M(x, y)$:
  $$M(x, y) = \exp\left(-\left(\frac{(x - x_0)^2}{2\sigma_x^2} + \frac{(y - y_0)^2}{2\sigma_y^2}\right)\right)$$
  $$I_{\text{uneven}}(x, y) = I(x, y) \cdot M(x, y) + 238 \cdot (1 - M(x, y))$$
* **Target Class Assignment:**  
  **DISTORTED CLASS ONLY (Class 1).**
* **Preservation of Structure & Label:**  
  Simulates genuine uneven skin contact where the user pressed only one quadrant of the thumb.

---

## 3. Transformations That Must Be Strictly Avoided

| Transformation | Why It Must Be Avoided | Risk to Biometric Recognition |
| :--- | :--- | :--- |
| **Heavy Gaussian Blur ($\sigma > 2.5$)** | Reintroduces the exact Classifier V3 shortcut where blur was learned as a proxy for distortion. | Inverts classifier predictions on clean, sharp images. |
| **Random Cutout / Black Patch Masking** | Creates artificial black square boundaries that never occur on an optical TIR sensor. | CNN learns to detect rectangular occlusion boundaries. |
| **Binarization / Hard Otsu Thresholding** | Eliminates grayscale ridge gradients and valley luminance depth. | Destroys subtle skin elasticity signals and subpixel ridge flow. |
| **Extreme Elastic Distortion Grids ($> 15$ px)** | Random non-physical warping generates self-intersecting ridges that violate human skin mechanics. | Degrades recognition and corrupts DDRNet rectification targets. |
| **Class-Asymmetric Darkening** | Darkening only clean or only distorted prints. | Causes the classifier to classify by average luminance rather than deformation. |

---

## 4. Implementation Priority & Staging

1. **Phase 1 Priority (Core Photometric Adaptation):**
   - Augmentation 1 (TIR Ridge Darkening)
   - Augmentation 2 (Laplacian Sharpness Modulation)
   - Augmentation 4 (Aspect-Ratio Vertical Squeeze)
   *Rationale:* Directly addresses the $28\times$ sharpness and $8\times$ ridge darkness disparity identified in the audit.
2. **Phase 2 Priority (Morphological Robustness):**
   - Augmentation 6 (Rotation / Tilt Boundaries)
   - Augmentation 7 (Elliptical Contact Falloff)
   - Augmentation 5 (Sensor Noise)
   *Rationale:* Refines the decision boundary between subtle tilt and heavy orientation distortion.

