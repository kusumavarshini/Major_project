"""
Test parallel matching worker in a standalone script.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJ_ROOT))

import cv2
from matcher import MATCHER
import afis.algorithms.safis as safis

GLOBAL_GALLERY = []

def init_worker(gallery_data):
    global GLOBAL_GALLERY
    GLOBAL_GALLERY = gallery_data

def process_one_query(q):
    global GLOBAL_GALLERY
    img = cv2.imread(q["path"], cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"filename": q["filename"], "error": "Cannot read"}
    tpl = MATCHER.extract_template(img)
    
    # Match against global gallery
    scores = []
    for g in GLOBAL_GALLERY:
        res = MATCHER.match_1to1(tpl, g["template"])
        scores.append((g["identity"], res.score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return {"filename": q["filename"], "top_id": scores[0][0], "top_score": scores[0][1]}

def main():
    clean_dir = PROJ_ROOT / "data" / "split" / "test"
    sample_clean = list(clean_dir.glob("**/*.png"))[:10]
    gallery = []
    for p in sample_clean:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        tpl = MATCHER.extract_template(img)
        rel = p.relative_to(clean_dir)
        fam, mem = rel.parts[0], rel.parts[1]
        gallery.append({
            "identity": f"{fam}/{mem}",
            "template": tpl
        })
    print(f"Gallery loaded: {len(gallery)} templates.")

    queries = [{"path": str(p), "filename": p.name} for p in sample_clean[:4]]
    t0 = time.time()
    with Pool(processes=min(cpu_count(), 4), initializer=init_worker, initargs=(gallery,)) as pool:
        results = pool.map(process_one_query, queries)
    print(f"Parallel evaluation completed in {time.time() - t0:.2f}s:")
    for r in results:
        print(" ", r)

if __name__ == "__main__":
    main()
