import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from matcher import MATCHER
from authentication import classify_distortion, rectify_with_ddrnet
from mysql_auth import get_all_gallery_templates

enrolled_path = Path("experiments/enrollment_images/enrolled_00011_7166610e.png")
enrolled_img = cv2.imread(str(enrolled_path), cv2.IMREAD_GRAYSCALE)

enrolled_tpl = MATCHER.extract_template(enrolled_img)
print(f"Enrolled '00011' minutiae count: {enrolled_tpl['minutiae_count']}")

# Get recent scan
recent_scans = sorted(Path("experiments/authentication_outputs").glob("*_orig.png"), key=lambda p: p.stat().st_mtime)
print(f"Found {len(recent_scans)} recent scans.")

for probe_path in recent_scans[-4:]:
    probe_img = cv2.imread(str(probe_path), cv2.IMREAD_GRAYSCALE)
    cls_res = classify_distortion(probe_img)
    rect_img = rectify_with_ddrnet(probe_img)
    
    probe_raw_tpl = MATCHER.extract_template(probe_img)
    probe_rect_tpl = MATCHER.extract_template(rect_img)
    
    # Match raw probe vs enrolled
    res_raw = MATCHER.match_1to1(probe_raw_tpl, enrolled_tpl)
    # Match rect probe vs enrolled
    res_rect = MATCHER.match_1to1(probe_rect_tpl, enrolled_tpl)
    
    # Also run 1:N on both
    gallery = get_all_gallery_templates()
    id_raw = MATCHER.identify_1toN(probe_raw_tpl, gallery, top_n=3)
    id_rect = MATCHER.identify_1toN(probe_rect_tpl, gallery, top_n=3)
    
    print("=" * 60)
    print(f"Scan: {probe_path.name}")
    print(f"  Distortion: prob_dist={cls_res['prob_distorted']:.3f}, class={cls_res['predicted_class']}")
    print(f"  Probe Raw Minutiae: {probe_raw_tpl['minutiae_count']} | Rect Minutiae: {probe_rect_tpl['minutiae_count']}")
    print(f"  1:1 vs Enrolled '00011':")
    print(f"    Raw Probe Match:  Score={res_raw.raw_score}, Matched Minutiae={res_raw.matched_minutiae_count}, Match={res_raw.match}")
    print(f"    Rect Probe Match: Score={res_rect.raw_score}, Matched Minutiae={res_rect.matched_minutiae_count}, Match={res_rect.match}")
    print(f"  1:N Search (Raw Probe):")
    for i, c in enumerate(id_raw.get('top_5', [])[:3]):
        print(f"    #{i+1}: Subject={c['subject_id']}, Score={c['raw_score']}, Minutiae={c['matched_minutiae']}")
    print(f"  1:N Search (Rect Probe):")
    for i, c in enumerate(id_rect.get('top_5', [])[:3]):
        print(f"    #{i+1}: Subject={c['subject_id']}, Score={c['raw_score']}, Minutiae={c['matched_minutiae']}")
