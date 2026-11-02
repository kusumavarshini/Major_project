import time
import pickle
import mysql.connector
import os
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")
from matcher import MATCHER
import afis.algorithms.safis as safis

conn = mysql.connector.connect(
    host='localhost', port=3306, user='root', password='1234', database='fingerprint_authentication'
)
cur = conn.cursor(dictionary=True)
cur.execute("""
    SELECT f.fingerprint_id, p.person_id, p.identity_code, f.impression_id, f.minutiae_template, f.minutiae_count 
    FROM fingerprints f 
    JOIN persons p ON f.person_id = p.person_id 
    WHERE f.impression_id IN (1, 2) 
    ORDER BY p.person_id, f.impression_id 
    LIMIT 600
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
    tpl = {'minutiae': t_obj.minutiae, 'minutiae_count': int(r['minutiae_count'] or 0)}
    if r['impression_id'] == 1:
        pre_index(tpl, MATCHER.params)
        gallery.append({'fingerprint_id': r['fingerprint_id'], 'person_id': r['person_id'], 'subject_id': r['identity_code'], 'template': tpl})
    elif r['impression_id'] == 2:
        probes.append({'fingerprint_id': r['fingerprint_id'], 'person_id': r['person_id'], 'subject_id': r['identity_code'], 'template': tpl})

print(f'Gallery size: {len(gallery)}, Probes size: {len(probes)}')

for i in range(5):
    p = probes[i]
    t0 = time.time()
    res = MATCHER.identify_1toN(p['template'], gallery, top_n=5)
    t1 = time.time()
    target = p['subject_id']
    ranks = {c['subject_id']: idx + 1 for idx, c in enumerate(res['candidates'])}
    rk = ranks.get(target, 999)
    top_cand = res['candidates'][0] if res['candidates'] else None
    print(f"Probe {i+1} ({target}): time={(t1-t0)*1000:.1f}ms, target_rank={rk}, top1={top_cand['subject_id']} (score={top_cand['raw_score']})")
