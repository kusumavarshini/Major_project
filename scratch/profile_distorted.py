import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import cv2
import numpy as np
from authentication import identify_fingerprint

img_path = "data/latent_fvc/1.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is not None:
    print(f"Testing with distorted image: {img_path}, shape: {img.shape}")
    t0 = time.time()
    res = identify_fingerprint(img, top_n=5, threshold=100.0)
    total_time = (time.time() - t0) * 1000.0
    print("\n--- Pipeline Profiling Results (Distorted) ---")
    print(f"Total measured time: {total_time:.2f} ms")
    print(f"Stage times: {res.get('stage_times')}")
    print(f"Match found: {res.get('match_found')}")
    print(f"Confidence: {res.get('confidence_level')}")
    print(f"Rejection reason: {res.get('rejection_reason')}")
    print("Candidates:")
    for c in res.get("candidates", [])[:5]:
        print(f"  {c.get('subject_id')}: score={c.get('score')}, matched_minutiae={c.get('matched_minutiae')}")
