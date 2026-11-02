import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import cv2
import numpy as np
import afis.algorithms.safis as safis
from mysql_auth import get_all_gallery_templates
from matcher import MATCHER

gallery = get_all_gallery_templates()
img = cv2.imread("data/clean_fvc_new/1.png", cv2.IMREAD_GRAYSCALE)
probe_tpl = MATCHER.extract_template(img)
probe_minu = probe_tpl["minutiae"]

params = MATCHER.params
length_tol = float(params.length_tol) or 1e-6
angle_tol = params.angle_tol
max_roots = params.max_roots

# Pre-build candidate adjacencies
for cand_record in gallery:
    cand_tpl = cand_record.get("template", cand_record)
    cand_edges = cand_tpl.get("_edges")
    if cand_edges and "_adj" not in cand_tpl:
        c_adj = {}
        for e in cand_edges:
            c_adj.setdefault(e.ref, []).append(e)
        cand_tpl["_adj"] = c_adj

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

# Run both on all 319 templates and compare every score
ea = safis.build_edges(probe_minu, params)
p_adj = {}
for e in ea:
    p_adj.setdefault(e.ref, []).append(e)

diffs = 0
for cand_record in gallery:
    cand_tpl = cand_record.get("template", cand_record)
    cand_edges = cand_tpl.get("_edges", [])
    cand_buckets = cand_tpl.get("_buckets", {})
    c_adj = cand_tpl.get("_adj", {})

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
        continue
    roots.sort(key=lambda t: t[0], reverse=True)
    roots = roots[:max_roots]

    # Orig crawl
    best_n_orig = 0
    for _c, pe, ce in roots:
        p, _ = safis._crawl(ea, cand_edges, pe.ref, ce.ref, pe.neighbor, ce.neighbor, params)
        if len(p) > best_n_orig:
            best_n_orig = len(p)

    # Fast crawl
    best_n_fast = 0
    for _c, pe, ce in roots:
        p, _ = fast_crawl(p_adj, c_adj, pe.ref, ce.ref, pe.neighbor, ce.neighbor, params)
        if len(p) > best_n_fast:
            best_n_fast = len(p)

    if best_n_orig != best_n_fast:
        diffs += 1
        print(f"Diff on {cand_record.get('subject_id')}: orig={best_n_orig}, fast={best_n_fast}")

print(f"Total differences across all {len(gallery)} templates: {diffs}")
