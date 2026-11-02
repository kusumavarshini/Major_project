import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import afis.algorithms.safis as safis
import afis.algorithms.segmentation as seg
import afis.algorithms.orientation as ori
import afis.algorithms.frequency as freq
import afis.algorithms.enhancement as enh
import afis.algorithms.skeleton as skel
import afis.algorithms.minutiae as minu
from matcher import SourceAFISMatcher, MATCHER
from authentication import rectify_with_ddrnet
from mysql_auth import get_all_gallery_templates

def extract_template_improved(image, max_minutiae=100, use_clahe=True):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    TARGET_BIOMETRIC_SIZE = (384, 384)
    if (image.shape[1], image.shape[0]) != TARGET_BIOMETRIC_SIZE:
        proc_img = cv2.resize(image, TARGET_BIOMETRIC_SIZE, interpolation=cv2.INTER_CUBIC)
    else:
        proc_img = image.copy()
        
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        proc_img = clahe.apply(proc_img)
        
    normed = (proc_img.astype(np.float64) - float(np.mean(proc_img))) / (float(np.std(proc_img)) + 1e-6)
    mask, _ = seg.segment(normed, block_size=16)
    orientation = ori.estimate_orientation(normed, block_size=16)
    frequency = freq.estimate_frequency(normed, orientation, block_size=16)
    frequency = freq.fill_invalid(frequency, mask)
    enhanced = enh.enhance(normed, orientation, frequency, mask=mask, block_size=16, angle_steps=8)
    binary = skel.binarize(enhanced, mask=mask, block_size=16)
    skeleton = skel.skeletonize(binary)
    skeleton = skel.clean_skeleton(skeleton)
    raw_minutiae = minu.extract_minutiae(skeleton, orientation, mask=mask, border_margin=10)
    pruned = minu.prune_false_minutiae(raw_minutiae, skeleton, mask=mask, min_distance=7.0, max_minutiae=max_minutiae)
    minutiae_arr = minu.minutiae_to_array(pruned)
    return {
        "minutiae": minutiae_arr,
        "minutiae_count": int(minutiae_arr.shape[0]),
    }

tuned_params = safis.SafisParams(
    max_neighbors=16,
    max_edge_length=320.0,
    length_tol=14.0,
    angle_tol=0.45,
    max_roots=50,
)

enrolled_path = Path("experiments/enrollment_images/enrolled_00011_7166610e.png")
enrolled_img = cv2.imread(str(enrolled_path), cv2.IMREAD_GRAYSCALE)

tpl_enrolled_old = MATCHER.extract_template(enrolled_img)
tpl_enrolled_new = extract_template_improved(enrolled_img)
print(f"Enrolled '00011' minutiae: Old={tpl_enrolled_old['minutiae_count']}, Improved={tpl_enrolled_new['minutiae_count']}")

recent_scans = sorted(Path("experiments/authentication_outputs").glob("*_orig.png"), key=lambda p: p.stat().st_mtime)

for probe_path in recent_scans[-4:]:
    probe_img = cv2.imread(str(probe_path), cv2.IMREAD_GRAYSCALE)
    rect_img = rectify_with_ddrnet(probe_img)
    
    # Old extraction + old matcher params
    t_raw_old = MATCHER.extract_template(probe_img)
    t_rect_old = MATCHER.extract_template(rect_img)
    s_raw_old, _, p_raw_old = safis.match_minutiae(t_raw_old['minutiae'], tpl_enrolled_old['minutiae'], params=MATCHER.params)
    s_rect_old, _, p_rect_old = safis.match_minutiae(t_rect_old['minutiae'], tpl_enrolled_old['minutiae'], params=MATCHER.params)
    
    # Improved extraction + tuned params
    t_raw_new = extract_template_improved(probe_img)
    t_rect_new = extract_template_improved(rect_img)
    s_raw_new, _, p_raw_new = safis.match_minutiae(t_raw_new['minutiae'], tpl_enrolled_new['minutiae'], params=tuned_params)
    s_rect_new, _, p_rect_new = safis.match_minutiae(t_rect_new['minutiae'], tpl_enrolled_new['minutiae'], params=tuned_params)
    
    print("-" * 65)
    print(f"Scan {probe_path.name}:")
    print(f"  [RAW PROBE vs 00011]  Old Score: {s_raw_old:.1f} ({len(p_raw_old)} pairs)  ==>  Tuned Score: {s_raw_new:.1f} ({len(p_raw_new)} pairs)")
    print(f"  [RECT PROBE vs 00011] Old Score: {s_rect_old:.1f} ({len(p_rect_old)} pairs)  ==>  Tuned Score: {s_rect_new:.1f} ({len(p_rect_new)} pairs)")
    best_score_old = max(s_raw_old, s_rect_old)
    best_score_new = max(s_raw_new, s_rect_new)
    print(f"  --> Max Dual-Representation Score: {best_score_old:.1f} ==> {best_score_new:.1f}")
