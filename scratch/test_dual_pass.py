import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import afis.algorithms.safis as safis
from matcher import SourceAFISMatcher
from authentication import rectify_with_ddrnet
from mysql_auth import get_all_gallery_templates

gallery = get_all_gallery_templates()

matcher = SourceAFISMatcher()
matcher.params = safis.SafisParams(
    max_neighbors=16,
    max_edge_length=320.0,
    length_tol=14.0,
    angle_tol=0.45,
    max_roots=50,
)

# Test on 3 recent probe scans
recent_scans = sorted(Path("experiments/authentication_outputs").glob("*_orig.png"), key=lambda p: p.stat().st_mtime)

for probe_path in recent_scans[-3:]:
    probe_img = cv2.imread(str(probe_path), cv2.IMREAD_GRAYSCALE)
    rect_img = rectify_with_ddrnet(probe_img)
    
    tpl_raw = matcher.extract_template(probe_img)
    tpl_rect = matcher.extract_template(rect_img)
    
    # 1:N on raw and rect
    res_raw = matcher.identify_1toN(tpl_raw, gallery, top_n=5)
    res_rect = matcher.identify_1toN(tpl_rect, gallery, top_n=5)
    
    # Combined / Dual Match: For each candidate, score = max(raw, rect)
    combined_scores = {}
    for c in res_raw['candidates']:
        combined_scores[c['subject_id']] = max(combined_scores.get(c['subject_id'], 0), c['raw_score'])
    for c in res_rect['candidates']:
        combined_scores[c['subject_id']] = max(combined_scores.get(c['subject_id'], 0), c['raw_score'])
        
    sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("=" * 60)
    print(f"Probe: {probe_path.name}")
    print("Top 5 Combined Dual-Pass Matches:")
    for rank, (subj, sc) in enumerate(sorted_combined[:5], 1):
        print(f"  #{rank}: Subject '{subj}' -> Score = {sc:.1f}")
