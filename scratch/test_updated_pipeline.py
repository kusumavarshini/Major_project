import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from authentication import identify_fingerprint

print("==================================================")
print("TEST 1: GENUINE CLEAN QUERY (data/clean_fvc_new/1.png)")
print("==================================================")
img1 = cv2.imread("data/clean_fvc_new/1.png", cv2.IMREAD_GRAYSCALE)
res1 = identify_fingerprint(img1, top_n=5, threshold=100.0)
print(f"Decision: {res1['decision']}")
print(f"Match found: {res1['match_found']}")
print(f"Best Candidate (Accepted Identity): {res1['best_candidate']['subject_id'] if res1['best_candidate'] else None}")
print(f"Score: {res1['best_candidate']['raw_score'] if res1['best_candidate'] else None}")
print(f"Operating Threshold: {res1['operating_threshold']}")
print(f"Decision Reason: {res1['decision_reason']}")
print("Top 5 candidates:")
for c in res1['top_5']:
    print(f"  {c['subject_id']}: score={c['raw_score']}, match={c['match']}")

print("\n==================================================")
print("TEST 2: UNENROLLED / DISTORTED QUERY (data/latent_fvc/1.png)")
print("==================================================")
img2 = cv2.imread("data/latent_fvc/1.png", cv2.IMREAD_GRAYSCALE)
res2 = identify_fingerprint(img2, top_n=5, threshold=100.0)
print(f"Decision: {res2['decision']}")
print(f"Match found: {res2['match_found']}")
print(f"Best Candidate (Accepted Identity): {res2['best_candidate']['subject_id'] if res2['best_candidate'] else None}")
print(f"Nearest Candidate: {res2['nearest_candidate']['subject_id'] if res2['nearest_candidate'] else None} (score={res2['nearest_candidate']['raw_score'] if res2['nearest_candidate'] else None})")
print(f"Operating Threshold: {res2['operating_threshold']}")
print(f"Decision Reason: {res2['decision_reason']}")
print("Top 5 candidates (still visible as nearest candidates):")
for c in res2['top_5']:
    print(f"  {c['subject_id']}: score={c['raw_score']}, match={c['match']}")
