import os
import sys
import time
import pickle
import mysql.connector
from pathlib import Path
from multiprocessing import Pool, cpu_count
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from matcher import MATCHER
import afis.algorithms.safis as safis

GLOBAL_GALLERY = []

def init_worker(gallery_data):
    global GLOBAL_GALLERY
    GLOBAL_GALLERY = gallery_data

def process_probe(probe):
    global GLOBAL_GALLERY
    target_identity = probe['identity_code']
    p_tpl = probe['template']
    p_minu = p_tpl.get('minutiae')
    is_fta = (p_minu is None or len(p_minu) < 2)

    res = MATCHER.identify_1toN(p_tpl, GLOBAL_GALLERY, top_n=5)
    candidates = res.get('candidates', [])

    ranks = {c['subject_id']: idx + 1 for idx, c in enumerate(candidates)}
    rk = ranks.get(target_identity, None)

    top1_cand = candidates[0] if candidates else None
    pred_top1 = top1_cand['subject_id'] if top1_cand else None
    pred_top1_score = float(top1_cand['raw_score']) if top1_cand else 0.0

    target_cand = next((c for c in candidates if c['subject_id'] == target_identity), None)
    genuine_score = float(target_cand['raw_score']) if target_cand else 0.0

    is_top1 = (rk == 1)
    is_top5 = (rk is not None and rk <= 5)

    return {
        'probe_fingerprint_id': probe['fingerprint_id'],
        'person_id': probe['person_id'],
        'identity_code': target_identity,
        'family_id': probe['family_id'],
        'member_type': probe['member_type'],
        'impression_id': probe['impression_id'],
        'minutiae_count': probe['actual_minutiae_count'],
        'is_fta': is_fta,
        'predicted_top1': pred_top1,
        'predicted_top1_score': pred_top1_score,
        'genuine_rank': rk,
        'genuine_score': genuine_score,
        'is_top1': is_top1,
        'is_top5': is_top5,
    }

if __name__ == '__main__':
    print(f"Testing multiprocessing with {cpu_count()} cores...", flush=True)
    conn = mysql.connector.connect(
        host='localhost', port=3306, user='root', password='1234', database='fingerprint_authentication'
    )
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT f.fingerprint_id, p.person_id, p.identity_code, p.family_id, p.member_type, 
               f.impression_id, f.minutiae_template, f.minutiae_count 
        FROM fingerprints f 
        JOIN persons p ON f.person_id = p.person_id 
        WHERE f.impression_id IN (1, 2) 
        ORDER BY p.person_id, f.impression_id 
        LIMIT 20
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    def pre_index(cand_tpl, params):
        c_minu = cand_tpl.get('minutiae')
        if c_minu is None or len(c_minu) < 2:
            cand_tpl['_edges'] = []
            cand_tpl['_buckets'] = {}
            cand_tpl['_adj'] = {}
            return
        edges = safis.build_edges(c_minu, params)
        buckets = {}
        adj = {}
        lt = float(params.length_tol) or 1e-6
        for e in edges:
            k = int(round(e.length / lt))
            buckets.setdefault(k, []).append(e)
            adj.setdefault(e.ref, []).append(e)
        cand_tpl['_edges'] = edges
        cand_tpl['_buckets'] = buckets
        cand_tpl['_adj'] = adj
        cand_tpl['_cache_params_key'] = (
            params.max_neighbors,
            params.max_edge_length,
            params.length_tol,
            params.angle_tol,
        )

    gallery = []
    probes = []
    for r in rows:
        t_obj = pickle.loads(r['minutiae_template'])
        minu = getattr(t_obj, 'minutiae', None)
        tpl = {'minutiae': minu, 'minutiae_count': len(minu) if minu is not None else 0}
        item = {
            'fingerprint_id': r['fingerprint_id'],
            'person_id': r['person_id'],
            'identity_code': r['identity_code'],
            'family_id': r['family_id'],
            'member_type': r['member_type'],
            'impression_id': r['impression_id'],
            'actual_minutiae_count': len(minu) if minu is not None else 0,
            'template': tpl,
        }
        if r['impression_id'] == 1:
            pre_index(tpl, MATCHER.params)
            gallery.append({
                'fingerprint_id': r['fingerprint_id'],
                'person_id': r['person_id'],
                'subject_id': r['identity_code'],
                'template': tpl,
            })
        elif r['impression_id'] == 2:
            probes.append(item)

    print(f"Gallery items: {len(gallery)}, Probe items: {len(probes)}", flush=True)

    t0 = time.time()
    with Pool(processes=4, initializer=init_worker, initargs=(gallery,)) as pool:
        results = pool.map(process_probe, probes)
    t1 = time.time()
    print(f"Processed {len(results)} probes in {t1-t0:.2f}s (Avg {(t1-t0)/len(results):.2f}s per probe)", flush=True)
    for r in results[:5]:
        print(f"  Query {r['identity_code']}: Rank={r['genuine_rank']}, Score={r['genuine_score']}, Top1={r['is_top1']}, Top5={r['is_top5']}", flush=True)
