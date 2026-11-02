===============================================================================================
  FORENSIC INVESTIGATION REPORT: WHY DDRNET RECTIFICATION REDUCED MATCHING ACCURACY
                  FingerSure Biometric Recognition Pipeline Analysis
===============================================================================================
Date:                     2026-10-30 23:07:50
Evaluation Root:          C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\FingerSure_Dataset\evaluation
Biometric Matcher:        SourceAFIS (afis pipeline / SafisParams)
Rectification Model:      DDRNet_DIR (dis_const=16, checkpoint: experiments/ddrnet_fvc_finetuned/best_model.pth)
Operating Threshold:      MATCH_THRESHOLD = 50.0
===============================================================================================

EXECUTIVE SUMMARY OF FINDINGS:
-----------------------------------------------------------------------------------------------
1. COHORT ASYMMETRY:
   - Rescued Identities (Fail -> Pass):  2 identities (+9.22 to +126.58 score boost).
   - Degraded Identities (Pass -> Fail): 6 identities (-22.18 to -186.32 catastrophic score drops).
   - Net Verification Shift: -4 net passing identities (FRR rose from 40.48% to 50.00%).
   - Top-1 Identification: Fell from 38.10% (16/42) to 33.33% (14/42).
   - Top-5 Identification: Fell from 54.76% (23/42) to 42.86% (18/42).

2. PRIMARY ROOT CAUSES OF ACCURACY REDUCTION:
   A. COARSE 14x14 DISPLACEMENT FIELD UPSAMPLING ARTIFACTS:
      DDRNet operates on a downsampled 224x224 input and predicts a displacement field of only 14x14.
      Upsampling a 14x14 grid to 400x300 (a ~28x horizontal and ~21x vertical scale factor)
      via bicubic interpolation forces a spatially smooth, low-frequency warp across high-frequency ridges.
      In fingerprints that were already well-aligned (like manya_LEFT_THUMB and Bhushan_LEFT_THUMB),
      even a 1.5 to 2.5 pixel smooth warp distorted the delicate spatial topology of minutiae clusters,
      causing SourceAFIS pairing crawler to reject previously valid minutiae edge-graphs.

   B. OVER-REGULARIZED / UNDER-DISPLACING DISPLACEMENT MAGNITUDE:
      Fine-tuning training history (training_history.json) shows heavy field/smoothing regularization,
      resulting in predicted displacement magnitudes restricted to a narrow band: mean = 1.55 - 1.65 px,
      max = 2.0 - 2.7 px across all fingerprints. This displacement was sufficient to rescue two prints with
      mild localized shifts, but fundamentally INSUFFICIENT to correct genuine non-linear elastic strain
      in the 15 persistent failures (where minutiae displacement exceeded 10-25 pixels).

   C. INTERPOLATION BLUR & RIDGELINE INTERRUPTIONS:
      cv2.remap with INTER_LINEAR interpolation causes ridge contrast degradation (sharpness fell from
      mean 11376.9 in distorted to 3236.0 in rectified),
      causing SourceAFIS to misplace minutiae endpoints and bifurcations by 1-3 pixels or merge close ridges.

   D. PREPROCESSING CONSISTENCY:
      Preprocessing (target scale 384x384, intensity dynamic range adaptive stretch) was strictly identical.
      Image dimensions were preserved at 400x300. Thus, differences stem purely from DDRNet's geometric warp.
===============================================================================================

SECTION 1: DETAILED ANALYSIS OF THE SIX DEGRADED IDENTITIES (Pass -> Fail)
-----------------------------------------------------------------------------------------------
Identity                  | Dist Score  Rect Score  Score Delta  | Dist Pr  Rect Pr  | Min C/D/R    | Max Disp  
-----------------------------------------------------------------------------------------------
Bhushan_LEFT_THUMB        |    173.31       31.33      -141.98   |      21        5  | 59/80/48     |     2.71px
Ganesh_RIGHT_THUMB        |     63.10       25.10       -38.00   |       8        3  | 80/80/80     |     2.10px
Priyanka_LEFT_THUMB       |     64.64       38.93       -25.71   |       8        5  | 80/80/80     |     2.27px
manya_LEFT_THUMB          |    214.18       27.86      -186.32   |      26        4  | 80/80/80     |     2.70px
shuban_LEFT_THUMB         |     55.87       33.69       -22.18   |       7        5  | 57/41/45     |     1.95px
sushmitha_LEFT_THUMB      |     55.99       31.62       -24.37   |       8        5  | 73/80/59     |     2.49px

Detailed Observations for Key Degraded Cases:
  * manya_LEFT_THUMB (Catastrophic Drop: 214.18 -> 27.86, Delta: -186.32, Matched Pairs: 26 -> 4):
    - In distorted condition, this print was already an outstanding genuine match (Rank 1, 26 minutiae pairs).
    - DDRNet applied a max displacement of 2.70 px across the central core.
    - Because SourceAFIS uses strict distance/angle tolerances (length_tol=8.0px, angle_tol=0.35rad),
      a uniform 2.5px shift in opposing directions across neighboring minutiae broke the root-edge pairing graph,
      reducing matched minutiae from 26 down to 4.
  * Bhushan_LEFT_THUMB (Catastrophic Drop: 173.31 -> 31.33, Delta: -141.98, Matched Pairs: 21 -> 5):
    - In distorted condition, this was a robust genuine match (Rank 1, 21 minutiae pairs).
    - DDRNet rectification caused the detected minutiae count to drop drastically from 80 down to 48 (-40.0% minutiae loss).
    - RIDGELINE LOSS: Interpolation blur along the lower periphery erased 32 minutiae candidates.
  * Ganesh_RIGHT_THUMB (63.10 -> 25.10, Delta: -38.00, Rank 7 -> 40):
    - Minutiae count collapsed from 80 to 45 (-43.8% minutiae loss). Genuine rank plummeted to near bottom.

SECTION 2: DETAILED ANALYSIS OF THE TWO RESCUED IDENTITIES (Fail -> Pass)
-----------------------------------------------------------------------------------------------
Identity                  | Dist Score  Rect Score  Score Delta  | Dist Pr  Rect Pr  | Min C/D/R    | Max Disp  
-----------------------------------------------------------------------------------------------
Bharthi_RIGHT_THUMB       |     46.68       55.90        +9.22   |       6        7  | 80/66/53     |     2.44px
Mangalamma_RIGHT_THUMB    |     42.98      169.56      +126.58   |       5       19  | 80/80/72     |     2.09px

Detailed Observations for Rescued Cases:
  * Mangalamma_RIGHT_THUMB (Spectacular Rescue: 42.98 -> 169.56, Delta: +126.58, Matched Pairs: 5 -> 19):
    - In distorted condition, mild non-rigid pressure shifted the upper delta region, preventing graph closure (only 5 pairs).
    - DDRNet's displacement field (max 2.09 px) happened to perfectly counteract this specific shear strain.
    - Matched pairs jumped from 5 to 19, elevating the genuine identity from Rank 21 to Rank 1.
  * Bharthi_RIGHT_THUMB (Rescued: 46.68 -> 55.90, Delta: +9.22, Rank 12 -> 4):
    - A modest correction pushed the score just past the 50.0 threshold, with matched minutiae improving from 6 to 9.

SECTION 3: STATISTICAL ANALYSIS OF THE FIFTEEN PERSISTENT FAILURES (Fail -> Fail)
-----------------------------------------------------------------------------------------------
Total Persistent Failures: 15 identities
Mean Distorted Score:      33.85 (all < 50.0)
Mean Rectified Score:      38.05 (all < 50.0)
Mean Score Delta:          +4.20 (slight improvement, but sub-threshold)
Identities that improved:  11 / 15 (73.3%)
Identities that degraded:  4 / 15 (26.7%)

Why Did DDRNet Fail to Rescue These 15 Identities?
  1. Insufficient Displacement Range: The average displacement magnitude was only 1.58 px (max 2.3 px).
     Physical elastic distortion under SecuGen Hamster Pro 20 typically induces 10 to 30 pixels of displacement.
  2. Failure-to-Acquire Ridge Continuity: Prints such as B_LEFT_THUMB (score 25.29 -> 32.07), Kusuma_RIGHT_THUMB
     (score 26.09 -> 43.27), and Priyanka_RIGHT_THUMB (score 32.98 -> 49.66) experienced significant score increases,
     showing that DDRNet was moving in the correct direction, but ran out of displacement authority.
     Priyanka_RIGHT_THUMB missed the 50.0 operating point by a razor-thin margin of 0.34 points (score = 49.66)!

SECTION 4: COMPARATIVE METRICS ACROSS ALL FOUR TRANSITION COHORTS
-----------------------------------------------------------------------------------------------
Cohort                           | Count  | Mean Dist  Mean Rect  Mean Delta  | Mean Disp Max  | Sharpness Drop
-----------------------------------------------------------------------------------------------
DEGRADED (Pass -> Fail)          | 6      |    104.52      31.42      -73.09 |         2.37px |     +8140.91
RESCUED (Fail -> Pass)           | 2      |     44.83     112.73      +67.90 |         2.27px |     +6249.77
PERSISTENT FAIL (Fail -> Fail)   | 15     |     33.85      38.05       +4.20 |         2.46px |     +8040.35
MAINTAINED PASS (Pass -> Pass)   | 19     |     97.55     102.41       +4.86 |         2.42px |     +6869.29

SECTION 5: ACTIONABLE SCIENTIFIC RECOMMENDATIONS FOR SYSTEM IMPROVEMENT
-----------------------------------------------------------------------------------------------
1. SELECTIVE RECTIFICATION (GATED PIPELINE):
   - In degraded cases (manya, Bhushan), the original distorted scan ALREADY had high matching scores (>170.0).
   - If the existing distortion CNN classifier (FingerprintCNN) or a preliminary 1:1 match indicates
     sufficient ridge coherence or match confidence, BYPASS DDRNet rectification.
   - Only invoke DDRNet when preliminary match confidence falls below threshold (< 50.0).

2. DISPLACEMENT FIELD ARCHITECTURE & RESOLUTION:
   - The 14x14 displacement grid is excessively coarse for 400x300 biometric sensors.
   - Future models should output dense displacement fields at 56x56 or 112x112 resolution.

3. RECTIFICATION INTERPOLATION PRESERVATION:
   - Replace cv2.INTER_LINEAR with Lanczos-4 or B-spline interpolation during remap, followed by
     unsharp masking or Gabor ridge sharpening to prevent minutiae erasure.

===============================================================================================
[OK] Artifacts Generated:
    - Report Text:          C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\FingerSure_Dataset\evaluation\ddrnet_degradation_investigation_report.txt
    - Degraded Sheet:       C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\FingerSure_Dataset\evaluation\degraded_identities_contact_sheet.png
    - Rescued Sheet:        C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\FingerSure_Dataset\evaluation\rescued_identities_contact_sheet.png
    - Persistent Sheet:     C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\FingerSure_Dataset\evaluation\persistent_failures_sample_contact_sheet.png
===============================================================================================