import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from matcher import MATCHER
import afis.algorithms.safis as safis
import afis.algorithms.segmentation as seg
from scipy import ndimage as ndi

def clean_border_minutiae(img, margin=16):
    proc_img = cv2.resize(img, (384, 384))
    normed = (proc_img.astype(np.float64) - float(np.mean(proc_img))) / (float(np.std(proc_img)) + 1e-6)
    mask, _ = seg.segment(normed, block_size=16)
    dist = ndi.distance_transform_edt(mask)
    tpl = MATCHER.extract_template(img)
    m = tpl['minutiae']
    valid = []
    for row in m:
        x, y = int(round(row[0])), int(round(row[1]))
        if 0 <= y < dist.shape[0] and 0 <= x < dist.shape[1] and dist[y, x] >= margin:
            valid.append(row)
    tpl['minutiae'] = np.array(valid)
    tpl['minutiae_count'] = len(valid)
    return tpl

q1_img = cv2.imread('data/Latent_fvc/1.png', cv2.IMREAD_GRAYSCALE)
c1_img = cv2.imread('data/clean_fvc_new/1.png', cv2.IMREAD_GRAYSCALE)
c83_img = cv2.imread('data/clean_fvc_new/83.png', cv2.IMREAD_GRAYSCALE)

for margin in [6, 12, 16, 20]:
    t_q1 = clean_border_minutiae(q1_img, margin)
    t_c1 = clean_border_minutiae(c1_img, margin)
    t_c83 = clean_border_minutiae(c83_img, margin)
    
    res_gen = MATCHER.match_1to1(t_q1, t_c1)
    res_imp = MATCHER.match_1to1(t_q1, t_c83)
    mq1 = t_q1["minutiae_count"]
    mc1 = t_c1["minutiae_count"]
    mc83 = t_c83["minutiae_count"]
    print(f"Margin {margin:>2}px: Q1 minu={mq1}, C1 minu={mc1}, C83 minu={mc83} | Genuine Score={res_gen.score:>6.2f} (pairs={res_gen.matched_minutiae_count}) vs Impostor C83 Score={res_imp.score:>6.2f} (pairs={res_imp.matched_minutiae_count})")
