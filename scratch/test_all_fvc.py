import sys
sys.path.append(".")
from pathlib import Path
import cv2
import numpy as np
from matcher import SourceAFISMatcher

matcher = SourceAFISMatcher()
img_dir = Path("data/clean_fvc")

files = sorted(list(img_dir.glob("*.png")), key=lambda p: int(p.stem) if p.stem.isdigit() else 999999)

results = []
failures = []
minutiae_counts = []

for f in files:
    fid = int(f.stem)
    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    if img is None:
        failures.append((fid, f.name, "Failed to read image"))
        continue
    try:
        tpl = matcher.extract_template(img)
        cnt = tpl.get("minutiae_count", 0)
        if cnt > 0:
            results.append((fid, cnt))
            minutiae_counts.append(cnt)
        else:
            failures.append((fid, f.name, "0 minutiae extracted"))
    except Exception as e:
        failures.append((fid, f.name, str(e)))

print(f"Processed: {len(files)}")
print(f"Successful: {len(results)}")
print(f"Failures: {len(failures)}")
if failures:
    print("Failures details:", failures)
if minutiae_counts:
    print(f"Min minutiae: {min(minutiae_counts)}")
    print(f"Max minutiae: {max(minutiae_counts)}")
    print(f"Avg minutiae: {sum(minutiae_counts) / len(minutiae_counts):.2f}")
