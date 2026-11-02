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

gallery = get_all_gallery_templates()
print(f"Loaded {len(gallery)} gallery templates from MySQL.")

recent_scans = sorted(Path("experiments/authentication_outputs").glob("*_orig.png"), key=lambda p: p.stat().st_mtime)

for probe_path in recent_scans[-4:]:
    probe_img = cv2.imread(str(probe_path), cv2.IMREAD_GRAYSCALE)
    rect_img = rectify_with_ddrnet(probe_img)
    
    t_raw = extract_template_improved(probe_img)
    t_rect = extract_template_improved(rect_img)
    
    # Match against gallery with raw and rect
    results = {}
    for cand in gallery:
        subj = cand["subject_id"]
        c_minu = cand["template"]["minutiae"]
        
        _, raw_s_raw, p_raw = safis.match_minutiae(t_raw["minutiae"], c_minu, params=tuned_params)
        _, raw_s_rect, p_rect = safis.match_minutiae(t_rect["minutiae"], c_minu, params=tuned_params)
        
        best_score = max(raw_s_raw, raw_s_rect)
        best_pairs = max(len(p_raw), len(p_rect))
        which = "RAW" if raw_s_raw >= raw_s_rect else "RECT"
        
        if subj not in results or best_score > results[subj]["score"]:
            results[subj] = {
                "subject_id": subj,
                "score": best_score,
                "pairs": best_pairs,
                "rep": which,
                "raw_score": raw_s_raw,
                "rect_score": raw_s_rect,
            }
            
    ranked = sorted(results.values(), key=lambda x: x["score"], reverse=True)
    print("=" * 65)
    print(f"Scan {probe_path.name}:")
    for i, c in enumerate(ranked[:5]):
        print(f"  #{i+1}: Subject={c['subject_id']:12s} Score={c['score']:6.1f} (Pairs={c['pairs']:2d}, Rep={c['rep']}) [Raw={c['raw_score']:5.1f}, Rect={c['rect_score']:5.1f}]")
