import pickle
import mysql.connector
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from matcher import MATCHER

conn = mysql.connector.connect(
    host='localhost', port=3306, user='root', password='1234', database='fingerprint_authentication'
)
cur = conn.cursor(dictionary=True)
cur.execute("""
    SELECT impression_id, minutiae_count, minutiae_template 
    FROM fingerprints 
    WHERE person_id=3 
    ORDER BY impression_id
""")
rows = cur.fetchall()
cur.close()
conn.close()

tpls = []
for r in rows:
    obj = pickle.loads(r['minutiae_template'])
    tpls.append({
        'impression_id': r['impression_id'], 
        'minutiae': obj.minutiae, 
        'minutiae_count': len(obj.minutiae)
    })

print("=== Person ID 3 (FAMILY-1/CHILD) Impressions Minutiae Counts ===")
for t in tpls:
    print(f"Impression {t['impression_id']}: {t['minutiae_count']} minutiae")

print("\n=== Genuine Match Scores vs Impression 1 ===")
for i in range(1, len(tpls)):
    res = MATCHER.match_1to1(tpls[i], tpls[0])
    print(f"Impression {tpls[i]['impression_id']} vs Impression 1: score={res.score}, raw_score={res.raw_score}, match={res.match}, matched_minutiae={res.matched_minutiae_count}")
