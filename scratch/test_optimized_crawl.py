import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import math
import numpy as np
import afis.algorithms.safis as safis
from mysql_auth import get_all_gallery_templates
from matcher import MATCHER

gallery = get_all_gallery_templates()
print(f"Loaded {len(gallery)} templates")

# Extract probe
import cv2
img = cv2.imread("data/clean_fvc_new/1.png", cv2.IMREAD_GRAYSCALE)
probe_tpl = MATCHER.extract_template(img)
probe_minu = probe_tpl["minutiae"]

# Optimized crawl function that takes pre-built p_adj and c_adj
def fast_crawl(p_adj, c_adj, root_p, root_c, root_pn, root_cn, params):
    pairing = {root_p: root_c, root_pn: root_cn}
    paired_c = {root_c, root_cn}
    pairs = [(root_p, root_c, 1.0), (root_pn, root_cn, 1.0)]

    changed = True
    while changed:
        changed = False
        for p_idx, c_idx in list(pairing.items()):
            for pe in p_adj.get(p_idx, ()):
                if pe.neighbor in pairing:
                    continue
                for ce in c_adj.get(c_idx, ()):
                    if ce.neighbor in paired_c:
                        continue
                    if not safis._edges_compatible(pe, ce, params):
                        continue
                    pairing[pe.neighbor] = ce.neighbor
                    paired_c.add(ce.neighbor)
                    pairs.append((pe.neighbor, ce.neighbor, 1.0))
                    changed = True
                    break
    return pairing, pairs

# Benchmark original identify_1toN
t0 = time.time()
res_orig = MATCHER.identify_1toN(probe_tpl, gallery, top_n=5)
t_orig = (time.time() - t0) * 1000.0
print(f"Original identify_1toN: {t_orig:.2f} ms")

# Benchmark with precomputed p_adj and c_adj
params = MATCHER.params
length_tol = float(params.length_tol) or 1e-6
angle_tol = params.angle_tol
max_roots = params.max_roots

# Precompute c_adj for all gallery templates
for cand_record in gallery:
    cand_tpl = cand_record.get("template", cand_record)
    if "_adj" not in cand_tpl:
        c_edges = cand_tpl.get("_edges", [])
        c_adj = {}
        for e in c_edges:
            c_adj.setdefault(e.ref, []).append(e)
        cand_tpl["_adj"] = c_adj

# Fast run
t0 = time.time()
ea = safis.build_edges(probe_minu, params)
p_adj = {}
for e in ea:
    p_adj.setdefault(e.ref, []).append(e)

candidate_results = []
for cand_record in gallery:
    cand_tpl = cand_record.get("template", cand_record)
    cand_edges = cand_tpl.get("_edges", [])
    cand_buckets = cand_tpl.get("_buckets", {})
    c_adj = cand_tpl.get("_adj", {})

    if not cand_edges or not ea:
        raw_score = 0.0
        pairs_count = 0
    else:
        roots = []
        for pe in ea:
            key = int(round(pe.length / length_tol))
            for k in (key - 1, key, key + 1):
                b = cand_buckets.get(k)
                if not b:
                    continue
                for ce in b:
                    if safis._edges_compatible(pe, ce, params):
                        close = 1.0 - (
                            abs(pe.length - ce.length) / length_tol
                            + safis._angle_diff(pe.ref_angle, ce.ref_angle) / angle_tol
                            + safis._angle_diff(pe.neighbor_angle, ce.neighbor_angle) / angle_tol
                        ) / 3.0
                        roots.append((close, pe, ce))

        if not roots:
            raw_score = 0.0
            pairs_count = 0
        else:
            roots.sort(key=lambda t: t[0], reverse=True)
            roots = roots[:max_roots]
            best_n = 0
            best_pairs_cnt = 0
            for _c, pe, ce in roots:
                pairing, pairs = fast_crawl(p_adj, c_adj, pe.ref, ce.ref, pe.neighbor, ce.neighbor, params)
                if len(pairing) > best_n:
                    best_n = len(pairing)
                    best_pairs_cnt = len(pairs)
                    if best_n >= 25:
                        break
            raw_score = float(best_n) * 5.0
            if best_n >= 2:
                raw_score += float(best_pairs_cnt) * 1.5
            pairs_count = best_pairs_cnt

    candidate_results.append({
        "fingerprint_id": cand_record.get("fingerprint_id"),
        "subject_id": cand_record.get("subject_id"),
        "raw_score": round(raw_score, 2),
        "matched_minutiae": pairs_count,
    })

candidate_results.sort(key=lambda x: (x["raw_score"], x["matched_minutiae"]), reverse=True)
t_fast = (time.time() - t0) * 1000.0
print(f"Optimized identify_1toN: {t_fast:.2f} ms")
print(f"Speedup: {t_orig / t_fast:.2f}x")
print(f"Top result: {candidate_results[0]}")
