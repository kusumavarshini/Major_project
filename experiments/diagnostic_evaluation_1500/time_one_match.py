import time
import pickle
import mysql.connector
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

gallery = []
for r in rows:
    if r['impression_id'] == 1:
        t_obj = pickle.loads(r['minutiae_template'])
        tpl = {'minutiae': t_obj.minutiae, 'minutiae_count': int(r['minutiae_count'] or 0)}
        gallery.append({'fingerprint_id': r['fingerprint_id'], 'person_id': r['person_id'], 'subject_id': r['identity_code'], 'template': tpl})

probes = [r for r in rows if r['impression_id'] == 2]

p = probes[0]
p_obj = pickle.loads(p['minutiae_template'])
p_tpl = {'minutiae': p_obj.minutiae, 'minutiae_count': int(p['minutiae_count'] or 0)}

print(f"Starting 1:N match for probe {p['identity_code']} ({len(p_tpl['minutiae'])} minutiae) against {len(gallery)} gallery templates...", flush=True)
t0 = time.time()
res = MATCHER.identify_1toN(p_tpl, gallery, top_n=5)
t1 = time.time()
print(f"Done! Elapsed: {t1-t0:.2f} s.", flush=True)
top = res['candidates'][0] if res['candidates'] else None
if top:
    print(f"Top candidate: {top['subject_id']}, score: {top['raw_score']}", flush=True)
target_rank = next((idx+1 for idx, c in enumerate(res['candidates']) if c['subject_id'] == p['identity_code']), None)
print(f"Target identity: {p['identity_code']}, genuine rank: {target_rank}", flush=True)
