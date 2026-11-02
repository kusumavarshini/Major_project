import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import afis.algorithms.safis as safis
from matcher import SourceAFISMatcher, MATCHER
from authentication import classify_distortion, rectify_with_ddrnet
from mysql_auth import get_all_gallery_templates

enrolled_path = Path("experiments/enrollment_images/enrolled_00011_7166610e.png")
enrolled_img = cv2.imread(str(enrolled_path), cv2.IMREAD_GRAYSCALE)

# Create an improved matcher with refined parameters
improved_matcher = SourceAFISMatcher()
improved_matcher.params = safis.SafisParams(
    max_neighbors=16,
    max_edge_length=320.0,
    length_tol=14.0,
    angle_tol=0.45,
    max_roots=40,
)

gallery = get_all_gallery_templates()

# Extract enrolled template with both matchers
enrolled_tpl_default = MATCHER.extract_template(enrolled_img)
enrolled_tpl_improved = improved_matcher.extract_template(enrolled_img)

print("=" * 70)
print(f"ENROLLED '00011' Minutiae: Default={enrolled_tpl_default['minutiae_count']}, Improved={enrolled_tpl_improved['minutiae_count']}")
print("=" * 70)

recent_scans = sorted(Path("experiments/authentication_outputs").glob("*_orig.png"), key=lambda p: p.stat().st_mtime)

for probe_path in recent_scans[-5:]:
    probe_img = cv2.imread(str(probe_path), cv2.IMREAD_GRAYSCALE)
    cls_res = classify_distortion(probe_img)
    rect_img = rectify_with_ddrnet(probe_img)
    
    # 1. Default matcher
    p_def_raw = MATCHER.extract_template(probe_img)
    p_def_rect = MATCHER.extract_template(rect_img)
    m_def_raw = MATCHER.match_1to1(p_def_raw, enrolled_tpl_default)
    m_def_rect = MATCHER.match_1to1(p_def_rect, enrolled_tpl_default)
    
    # 2. Improved matcher
    p_imp_raw = improved_matcher.extract_template(probe_img)
    p_imp_rect = improved_matcher.extract_template(rect_img)
    m_imp_raw = improved_matcher.match_1to1(p_imp_raw, enrolled_tpl_improved)
    m_imp_rect = improved_matcher.match_1to1(p_imp_rect, enrolled_tpl_improved)
    
    # 3. 1:N Search with Improved Matcher
    id_imp_raw = improved_matcher.identify_1toN(p_imp_raw, gallery, top_n=3)
    id_imp_rect = improved_matcher.identify_1toN(p_imp_rect, gallery, top_n=3)
    
    print(f"\n--- Scan: {probe_path.name} ---")
    print(f"Distortion: dist_prob={cls_res['prob_distorted']:.3f} ({cls_res['predicted_class']})")
    print("DEFAULT 1:1 Match vs 00011:")
    print(f"  Raw:  Score={m_def_raw.raw_score:.1f}, Pairs={m_def_raw.matched_minutiae_count}")
    print(f"  Rect: Score={m_def_rect.raw_score:.1f}, Pairs={m_def_rect.matched_minutiae_count}")
    print("IMPROVED 1:1 Match vs 00011:")
    print(f"  Raw:  Score={m_imp_raw.raw_score:.1f}, Pairs={m_imp_raw.matched_minutiae_count}")
    print(f"  Rect: Score={m_imp_rect.raw_score:.1f}, Pairs={m_imp_rect.matched_minutiae_count}")
    print("IMPROVED 1:N Best Candidate (Dual/Max):")
    top_raw = id_imp_raw['top_5'][0] if id_imp_raw['top_5'] else None
    top_rect = id_imp_rect['top_5'][0] if id_imp_rect['top_5'] else None
    print(f"  Top Raw:  {top_raw['subject_id'] if top_raw else 'None'} (Score={top_raw['raw_score'] if top_raw else 0})")
    print(f"  Top Rect: {top_rect['subject_id'] if top_rect else 'None'} (Score={top_rect['raw_score'] if top_rect else 0})")
