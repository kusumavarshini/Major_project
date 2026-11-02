import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
from matcher import MATCHER
from mysql_auth import get_all_gallery_templates

gallery = get_all_gallery_templates(force_refresh=True)
print(f"Gallery size: {len(gallery)}")

for q_id in [1, 2, 4, 6, 10]:
    q_path = PROJECT_ROOT / "data" / "Latent_fvc" / f"{q_id}.png"
    img = cv2.imread(str(q_path), cv2.IMREAD_GRAYSCALE)
    tpl = MATCHER.extract_template(img)
    res = MATCHER.identify_1toN(tpl, gallery, top_n=len(gallery))
    
    cands = res["candidates"]
    rank = None
    genuine_score = 0.0
    for idx, c in enumerate(cands, start=1):
        if str(c.get("fingerprint_id")) == str(q_id) or c.get("subject_id") == f"GALLERY-{q_id}":
            rank = idx
            genuine_score = float(c.get("raw_score", 0.0))
            break
            
    top1 = cands[0] if cands else {}
    t1_id = top1.get("fingerprint_id")
    t1_score = top1.get("raw_score")
    print(f"Query {q_id:>2}: Genuine ID={q_id} -> Rank={rank}, Score={genuine_score:.2f} | Top-1: ID={t1_id}, Score={t1_score:.2f}")
