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
from mysql_auth import get_all_gallery_templates, clear_gallery_cache

# 1. Clear gallery cache so fresh edges are built
clear_gallery_cache()
gallery = get_all_gallery_templates(force_refresh=True)

# 2. Configure Tuned SafisParams
params = safis.SafisParams(
    max_neighbors=12,
    max_edge_length=320.0,
    length_tol=11.0,
    angle_tol=0.40,
    max_roots=35,
)

def extract_template(image, max_minutiae=100):
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    TARGET_BIOMETRIC_SIZE = (384, 384)
    if (image.shape[1], image.shape[0]) != TARGET_BIOMETRIC_SIZE:
        proc_img = cv2.resize(image, TARGET_BIOMETRIC_SIZE, interpolation=cv2.INTER_CUBIC)
    else:
        proc_img = image.copy()
        
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

# 3. Test on the exact user scan: search_1791916338110
orig_path = Path('experiments/authentication_outputs/search_1791916338110_orig.png')
rect_path = Path('experiments/authentication_outputs/search_1791916338110_rect.png')

orig_img = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)
rect_img = cv2.imread(str(rect_path), cv2.IMREAD_GRAYSCALE)

t_raw = extract_template(orig_img)
t_rect = extract_template(rect_img)

print(f"Extracted minutiae: Raw={t_raw['minutiae_count']}, Rect={t_rect['minutiae_count']}")

# Build edge indices for probe templates
ea_raw = safis.build_edges(t_raw["minutiae"], params)
ea_rect = safis.build_edges(t_rect["minutiae"], params)

p_adj_raw = {}
for e in ea_raw:
    p_adj_raw.setdefault(e.ref, []).append(e)
p_adj_rect = {}
for e in ea_rect:
    p_adj_rect.setdefault(e.ref, []).append(e)

def match_probe(ea, p_adj, cand_tpl):
    c_minu = cand_tpl.get("minutiae")
    if c_minu is None or len(c_minu) < 2 or not ea:
        return 0.0, 0
    c_edges = safis.build_edges(c_minu, params)
    c_buckets = {}
    c_adj = {}
    for e in c_edges:
        k = int(round(e.length / params.length_tol))
        c_buckets.setdefault(k, []).append(e)
        c_adj.setdefault(e.ref, []).append(e)
    roots = []
    for pe in ea:
        key = int(round(pe.length / params.length_tol))
        for k in (key - 1, key, key + 1):
            for ce in c_buckets.get(k, []):
                if safis._edges_compatible(pe, ce, params):
                    close = 1.0 - (
                        abs(pe.length - ce.length) / params.length_tol
                        + safis._angle_diff(pe.ref_angle, ce.ref_angle) / params.angle_tol
                        + safis._angle_diff(pe.neighbor_angle, ce.neighbor_angle) / params.angle_tol
                    ) / 3.0
                    roots.append((close, pe, ce))
    if not roots:
        return 0.0, 0
    roots.sort(key=lambda t: t[0], reverse=True)
    roots = roots[:params.max_roots]
    best_n = 0
    best_pairs_cnt = 0
    from matcher import _fast_crawl
    for _c, pe, ce in roots:
        pairing, pairs = _fast_crawl(p_adj, c_adj, pe.ref, ce.ref, pe.neighbor, ce.neighbor, params)
        if len(pairing) > best_n:
            best_n = len(pairing)
            best_pairs_cnt = len(pairs)
    raw = float(best_n) * 5.0
    if best_n >= 2:
        raw += float(best_pairs_cnt) * 1.5
    return raw, best_pairs_cnt

# Dual-Representation Search
scores = []
for cand in gallery:
    subj = cand["subject_id"]
    cand_tpl = cand["template"]
    raw_s, raw_p = match_probe(ea_raw, p_adj_raw, cand_tpl)
    rect_s, rect_p = match_probe(ea_rect, p_adj_rect, cand_tpl)
    best_s = max(raw_s, rect_s)
    best_p = max(raw_p, rect_p)
    which = "RECT" if rect_s > raw_s else "RAW"
    scores.append({
        "subject_id": subj,
        "score": best_s,
        "pairs": best_p,
        "rep": which,
        "raw_score": raw_s,
        "rect_score": rect_s,
    })

scores.sort(key=lambda x: x["score"], reverse=True)
print("\n========== DUAL-REPRESENTATION GALLERY SEARCH RESULTS ==========")
for i, c in enumerate(scores[:5]):
    print(f"#{i+1}: Subject={c['subject_id']:12s} Score={c['score']:6.1f} (Pairs={c['pairs']:2d}, Rep={c['rep']}) [Raw={c['raw_score']:5.1f}, Rect={c['rect_score']:5.1f}]")

THRESHOLD = 50.0
top = scores[0]
match_found = top["score"] >= THRESHOLD
print(f"\nFinal Decision with Threshold {THRESHOLD}: {'MATCH_FOUND' if match_found else 'NO_MATCH_FOUND'}")
print(f"Winner: {top['subject_id']} with score {top['score']:.1f}")
