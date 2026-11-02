import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import cv2
import numpy as np
from authentication import identify_fingerprint
from matcher import MATCHER

img_path = "data/clean_fvc_new/1.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    img_path = "data/clean_fvc/1.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is not None:
    print(f"Testing with image: {img_path}, shape: {img.shape}")
    t0 = time.time()
    res = identify_fingerprint(img, top_n=5, threshold=100.0)
    total_time = (time.time() - t0) * 1000.0
    print("\n--- Pipeline Profiling Results ---")
    print(f"Total measured time: {total_time:.2f} ms")
    print(f"Reported processing time: {res.get('processing_time_ms')} ms")
    print(f"Stage times: {res.get('stage_times')}")
    print(f"Total searched: {res.get('total_gallery_count')}")
    print(f"Match found: {res.get('match_found')}")
    print(f"Best match: {res.get('best_match')}")
    print("Candidates:")
    for c in res.get("candidates", [])[:5]:
        print(f"  {c.get('subject_id')}: score={c.get('score')}, matched_minutiae={c.get('matched_minutiae')}")
else:
    print("No test image found.")
