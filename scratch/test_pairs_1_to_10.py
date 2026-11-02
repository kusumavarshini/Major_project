import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from matcher import MATCHER

print(f"{'Pair':<6} | {'Clean Minu':<10} | {'Latent Minu':<11} | {'Score':<8} | {'Matched':<7} | {'Match?':<6}")
print("-" * 60)

for k in range(1, 11):
    c_path = PROJECT_ROOT / "data" / "clean_fvc_new" / f"{k}.png"
    l_path = PROJECT_ROOT / "data" / "Latent_fvc" / f"{k}.png"
    
    img_c = cv2.imread(str(c_path), cv2.IMREAD_GRAYSCALE)
    img_l = cv2.imread(str(l_path), cv2.IMREAD_GRAYSCALE)
    
    tpl_c = MATCHER.extract_template(img_c)
    tpl_l = MATCHER.extract_template(img_l)
    
    res = MATCHER.match_1to1(tpl_l, tpl_c)
    mc = tpl_c["minutiae_count"]
    ml = tpl_l["minutiae_count"]
    print(f"{k:<6} | {mc:<10} | {ml:<11} | {res.score:<8.2f} | {res.matched_minutiae_count:<7} | {res.match}")
