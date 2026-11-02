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
from mysql_auth import get_all_gallery_templates
from authentication import rectify_with_ddrnet

def extract_template(image, use_clahe=True, max_minutiae=90):
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
    pruned = minu.prune_false_minutiae(raw_minutiae, skeleton, mask=mask, min_distance=8.0, max_minutiae=max_minutiae)
    minutiae_arr = minu.minutiae_to_array(pruned)
    return minutiae_arr

gallery = get_all_gallery_templates()
enrolled_00011 = [g for g in gallery if g["subject_id"] == "00011"][0]["template"]["minutiae"]
impostors = [g for g in gallery if g["subject_id"].startswith("GALLERY-")][:50]

scans = sorted(Path("experiments/authentication_outputs").glob("*_orig.png"), key=lambda p: p.stat().st_mtime)[-4:]

configs = [
    ("Default (k=8, l_tol=8, a_tol=0.35, r=15)", safis.SafisParams(max_neighbors=8, max_edge_length=300.0, length_tol=8.0, angle_tol=0.35, max_roots=15)),
    ("Config A (k=10, l_tol=10, a_tol=0.38, r=30)", safis.SafisParams(max_neighbors=10, max_edge_length=300.0, length_tol=10.0, angle_tol=0.38, max_roots=30)),
    ("Config B (k=12, l_tol=11, a_tol=0.40, r=35)", safis.SafisParams(max_neighbors=12, max_edge_length=320.0, length_tol=11.0, angle_tol=0.40, max_roots=35)),
    ("Config C (k=14, l_tol=12, a_tol=0.42, r=40)", safis.SafisParams(max_neighbors=14, max_edge_length=320.0, length_tol=12.0, angle_tol=0.42, max_roots=40)),
    ("Config D (k=16, l_tol=14, a_tol=0.45, r=50)", safis.SafisParams(max_neighbors=16, max_edge_length=320.0, length_tol=14.0, angle_tol=0.45, max_roots=50)),
]

for name, params in configs:
    print("=" * 70)
    print(f"Testing {name}:")
    genuine_scores = []
    max_impostor_scores = []
    
    for scan_path in scans:
        img = cv2.imread(str(scan_path), cv2.IMREAD_GRAYSCALE)
        rect_img = rectify_with_ddrnet(img)
        m_raw = extract_template(img)
        m_rect = extract_template(rect_img)
        
        # Match genuine vs 00011 (dual rep)
        _, s_raw, _ = safis.match_minutiae(m_raw, enrolled_00011, params=params)
        _, s_rect, _ = safis.match_minutiae(m_rect, enrolled_00011, params=params)
        g_score = max(s_raw, s_rect)
        genuine_scores.append(g_score)
        
        # Match vs impostors
        imp_scores = []
        for imp in impostors:
            _, is_raw, _ = safis.match_minutiae(m_raw, imp["template"]["minutiae"], params=params)
            _, is_rect, _ = safis.match_minutiae(m_rect, imp["template"]["minutiae"], params=params)
            imp_scores.append(max(is_raw, is_rect))
        max_imp = max(imp_scores) if imp_scores else 0
        max_impostor_scores.append(max_imp)
        
    print(f"  Genuine Scans vs 00011: {genuine_scores} (min={min(genuine_scores):.1f}, avg={np.mean(genuine_scores):.1f})")
    print(f"  Max Impostor Scores:     {max_impostor_scores} (max={max(max_impostor_scores):.1f}, avg={np.mean(max_impostor_scores):.1f})")
    margin = min(genuine_scores) - max(max_impostor_scores)
    print(f"  Separation Margin:       {margin:.1f}")
