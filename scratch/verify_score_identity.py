import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from mysql_auth import get_all_gallery_templates
from matcher import MATCHER
import afis.algorithms.safis as safis

gallery = get_all_gallery_templates()
img = cv2.imread("data/clean_fvc_new/1.png", cv2.IMREAD_GRAYSCALE)
probe_tpl = MATCHER.extract_template(img)

# 1. Standard identify_1toN
res_std = MATCHER.identify_1toN(probe_tpl, gallery, top_n=5)
top_std = [(c["subject_id"], c["raw_score"], c["matched_minutiae"]) for c in res_std["candidates"]]

print("Standard Top 5:")
for r in top_std:
    print(" ", r)
