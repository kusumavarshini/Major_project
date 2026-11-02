import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import afis.algorithms.safis as safis
from matcher import SourceAFISMatcher, MATCHER

enrolled_path = Path("experiments/enrollment_images/enrolled_00011_7166610e.png")
enrolled_img = cv2.imread(str(enrolled_path), cv2.IMREAD_GRAYSCALE)

matcher = SourceAFISMatcher()
matcher.params = safis.SafisParams(
    max_neighbors=16,
    max_edge_length=320.0,
    length_tol=14.0,
    angle_tol=0.45,
    max_roots=50,
)

tpl_enrolled = matcher.extract_template(enrolled_img)

recent_scans = sorted(Path("experiments/authentication_outputs").glob("*_orig.png"), key=lambda p: p.stat().st_mtime)

for probe_path in recent_scans[-4:]:
    img = cv2.imread(str(probe_path), cv2.IMREAD_GRAYSCALE)
    
    # 1. Without CLAHE
    t1 = matcher.extract_template(img)
    m1 = matcher.match_1to1(t1, tpl_enrolled)
    
    # 2. With CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img)
    t2 = matcher.extract_template(img_clahe)
    m2 = matcher.match_1to1(t2, tpl_enrolled)
    
    print(f"Scan {probe_path.name}:")
    print(f"  Raw:   Minutiae={t1['minutiae_count']}, Match vs 00011 Score={m1.raw_score:.1f}, Pairs={m1.matched_minutiae_count}")
    print(f"  CLAHE: Minutiae={t2['minutiae_count']}, Match vs 00011 Score={m2.raw_score:.1f}, Pairs={m2.matched_minutiae_count}")
