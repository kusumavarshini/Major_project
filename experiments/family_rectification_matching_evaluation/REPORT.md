# DDRNet Fingerprint Rectification: Matching Accuracy Evaluation Report

**Project:** Fingerprint Distortion Rectification for Accurate Recognition  
**Evaluation Target:** 180 Eligible Test Queries (Impressions 2–5) from the Family Fingerprint Dataset  
**Clean Gallery:** 300 Enrolled Individual Identities (Impression 1, MySQL `fingerprint_authentication`)  
**Matching Algorithm:** SourceAFIS Invariant Edge Crawling (`matcher.py` / `MATCHER.identify_1toN`)  
**Operating Decision Threshold:** `50.0` (`database.MATCH_THRESHOLD`)  
**Date:** 2026-10-27  

---

## 1. Executive Summary

This evaluation measures whether DDRNet non-rigid geometric rectification improves SourceAFIS biometric identification of distorted family fingerprints.

- **Query Set:** Exactly **180 test queries** from impressions 2–5, corresponding to the 45 test persons across 15 test families (`FAMILY-13`, `FAMILY-14`, `FAMILY-15`, `FAMILY-20`, `FAMILY-23`, `FAMILY-36`, `FAMILY-37`, `FAMILY-42`, `FAMILY-48`, `FAMILY-54`, `FAMILY-65`, `FAMILY-66`, `FAMILY-67`, `FAMILY-78`, `FAMILY-9`).
- **Gallery Set:** Exactly **300 enrolled clean individual templates** (clean Impression 1 of all 300 persons in the database).
- **Zero Self-Matching:** Every query is physically disjoint from its clean enrolled gallery template (queries are impressions 2–5; gallery is impression 1).
- **Ground Truth:** Biometric identity is evaluated strictly at the **individual person level** (`identity_code = family_id/member_type`). Family and member categories are not treated as identities.

---

## 2. Before vs. After Rectification Performance Comparison

| Metric | Condition 1: BEFORE (Original Distorted) | Condition 2: AFTER (DDRNet Rectified) | Absolute Impact |
| :--- | :---: | :---: | :---: |
| **Total Test Queries** | **180** | **180** | — |
| **Clean Gallery Size (Identities)** | **300** | **300** | — |
| **Combined Top-1 Accuracy (All Queries)** | **0.56%** (1/180) | **0.0%** (0/180) | **-0.56%** |
| **Combined Top-1 Accuracy (Valid Queries)** | **0.58%** (1/171) | **0.0%** (0/169) | **-0.58%** |
| **Combined Top-5 Accuracy (All Queries)** | **0.56%** (1/180) | **1.11%** (2/180) | **+0.55%** |
| **Combined Top-5 Accuracy (Valid Queries)** | **0.58%** (1/171) | **1.18%** (2/169) | **+0.60%** |
| **False Rejection Rate (FRR @ Thresh 50.0)** | **88.33%** (159/180) | **90.56%** (163/180) | **+2.23%** |
| **False Acceptance Rate (FAR @ Thresh 50.0)** | **13.3185%** (7168/53820) | **9.6191%** (5177/53820) | **-3.6994%** |
| **Mean Genuine Match Score** | **33.11** | **31.06** | **-2.05** |
| **Median Genuine Match Score** | **33.19** | **30.58** | **-2.61** |
| **Valid Query Coverage (Minutiae $\ge 2$)** | **171/180** (1 FTA) | **169/180** (4 FTA) | **-2 queries** |
| **Per-Query Rank Improvement** | — | **87 improved, 87 degraded, 6 unchanged** | — |

---

## 3. Results by Distortion Type

### 3.1 Bending Distortion (60 Queries)
- **Before Rectification:**
  - Top-1 Accuracy: **0.0%** (0/60 all, 0.0% valid)
  - Top-5 Accuracy: **0.0%** (0/60 all, 0.0% valid)
  - FRR: **85.0%** | FAR: **17.1795%** (Trials: 60 genuine, 17940 impostor)
- **After Rectification:**
  - Top-1 Accuracy: **0.0%** (0/60 all, 0.0% valid)
  - Top-5 Accuracy: **1.67%** (1/60 all, 1.79% valid)
  - FRR: **90.0%** | FAR: **10.3567%** (Trials: 60 genuine, 17940 impostor)
  - Rank Shifts: **27 improved, 29 degraded, 4 unchanged**

### 3.2 Elastic Distortion (59 Queries)
- **Before Rectification:**
  - Top-1 Accuracy: **1.69%** (1/59 all, 1.82% valid)
  - Top-5 Accuracy: **1.69%** (1/59 all, 1.82% valid)
  - FRR: **89.83%** | FAR: **10.2999%** (Trials: 59 genuine, 17641 impostor)
- **After Rectification:**
  - Top-1 Accuracy: **0.0%** (0/59 all, 0.0% valid)
  - Top-5 Accuracy: **0.0%** (0/59 all, 0.0% valid)
  - FRR: **89.83%** | FAR: **7.3635%** (Trials: 59 genuine, 17641 impostor)
  - Rank Shifts: **29 improved, 29 degraded, 1 unchanged**

### 3.3 Local Distortion (61 Queries)
- **Before Rectification:**
  - Top-1 Accuracy: **0.0%** (0/61 all, 0.0% valid)
  - Top-5 Accuracy: **0.0%** (0/61 all, 0.0% valid)
  - FRR: **90.16%** | FAR: **12.4404%** (Trials: 61 genuine, 18239 impostor)
- **After Rectification:**
  - Top-1 Accuracy: **0.0%** (0/61 all, 0.0% valid)
  - Top-5 Accuracy: **1.64%** (1/61 all, 1.69% valid)
  - FRR: **91.8%** | FAR: **11.0752%** (Trials: 61 genuine, 18239 impostor)
  - Rank Shifts: **31 improved, 29 degraded, 1 unchanged**

---

## 4. Verification Protocol and Metric Formulas

### 4.1 Identification Accuracy (Top-1 and Top-5)
- **Top-1 Accuracy:**
  $$\text{Top-1} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{genuine\_rank}_i = 1)}{N} \times 100\%$$
- **Top-5 Accuracy:**
  $$\text{Top-5} = \frac{\sum_{i=1}^{N} \mathbb{I}(\text{genuine\_rank}_i \le 5)}{N} \times 100\%$$

### 4.2 Verification Rates (FAR and FRR)
The verification protocol evaluates each query against the decision threshold $T = 50.0$:
- **Genuine Trials ($N_{g} = 180$):**
  Each probe is compared to the clean enrolled template of the same person.
  $$\text{FRR}(T) = \frac{\sum_{i=1}^{N_g} \mathbb{I}(\text{genuine\_score}_i < T)}{N_g} \times 100\%$$
- **Impostor Trials ($N_{imp} = 180 \times 299 = 53,820$ comparisons):**
  Each probe is compared against the clean enrolled templates of all other 299 persons.
  $$\text{FAR}(T) = \frac{\sum_{i=1}^{N_g} \sum_{j \ne \text{target}} \mathbb{I}(\text{impostor\_score}_{i,j} \ge T)}{N_{imp}} \times 100\%$$

---

## 5. Key Findings and Technical Insights

1. **Identification Accuracy Impact:**
   - **Top-1 Accuracy:** DDRNet rectification **did not improve Top-1 accuracy** (0.56% before vs. 0.00% after). In fact, the only genuine Top-1 match in the distorted set (`FAMILY-13/FATHER`, impression 3) was degraded to rank 44 after rectification due to altered minutiae coordinates.
   - **Top-5 Accuracy:** Marginally shifted from 0.56% (1/180) to 1.11% (2/180), corresponding to +1 additional candidate barely entering the top-5 window, while Top-1 remained at 0%.
   - **Rank Shift Distribution:** Across all 180 test queries, **87 queries improved** in genuine rank, **87 queries degraded**, and **6 queries were unchanged**. This symmetrical 50/50 split demonstrates that the rectification transformation acted essentially as a random spatial perturbation on the minutiae graph topology relative to the clean gallery reference.

2. **Verification Rate Impact (FAR vs. FRR):**
   - **FRR at Threshold 50.0:** Slightly worsened from **88.33%** (159 false rejections) to **90.56%** (163 false rejections). Mean genuine match score dropped from 33.11 to 31.06.
   - **FAR at Threshold 50.0:** Decreased from **13.3185%** (7,168 false accepts out of 53,820 impostor comparisons) to **9.6191%** (5,177 false accepts). This drop in FAR occurred primarily because rectified images had slightly lower overall minutiae density and lower raw scores across all candidates, not because genuine discriminability increased.

3. **Template Extraction and Minutiae Quality:**
   - Minutiae extraction failures (FTA < 2 minutiae) increased from 1 query before rectification to 4 queries after rectification (`FM023_C2`, `FM023_C4`, `FM023_C5`, `FM048_C4`).
   - Image resampling during neural rectification introduced subtle edge blur and ridge smoothing, which occasionally disrupted true ridge bifurcations/endings or shifted minutiae orientations outside the SourceAFIS angular tolerance ($\pm 0.39$ rad).

---

## 6. Artifacts and Reproduction Command

The evaluation was executed using the standalone script:
```powershell
.venv\Scripts\python -u experiments/family_rectification_matching_evaluation/run_evaluation.py
```

All generated files are strictly preserved in:
- [`REPORT.md`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/family_rectification_matching_evaluation/REPORT.md)
- [`summary.csv`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/family_rectification_matching_evaluation/summary.csv)
- [`per_query_results.csv`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/family_rectification_matching_evaluation/per_query_results.csv)
- [`metrics.json`](file:///c:/Users/kusum/OneDrive/Desktop/trial/Fingerprint-distortion-detection-for-accurate-recognition/experiments/family_rectification_matching_evaluation/metrics.json)

