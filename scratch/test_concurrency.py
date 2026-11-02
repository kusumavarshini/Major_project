import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from mysql_auth import get_all_gallery_templates
from matcher import MATCHER
import afis.algorithms.safis as safis

gallery = get_all_gallery_templates()
img = cv2.imread("data/clean_fvc_new/1.png", cv2.IMREAD_GRAYSCALE)
probe_tpl = MATCHER.extract_template(img)

# Sequential
t0 = time.time()
res_seq = MATCHER.identify_1toN(probe_tpl, gallery, top_n=5)
t_seq = (time.time() - t0) * 1000.0

print(f"Sequential: {t_seq:.2f} ms")
