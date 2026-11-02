import os
import pickle
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv('MYSQL_HOST', 'localhost'),
    port=int(os.getenv('MYSQL_PORT', 3306)),
    user=os.getenv('MYSQL_USER', 'root'),
    password=os.getenv('MYSQL_PASSWORD', '1234'),
    database='fingerprint_authentication'
)

cur = conn.cursor(dictionary=True)
cur.execute("""
    SELECT impression_id, COUNT(*) as cnt 
    FROM fingerprints 
    WHERE minutiae_count IS NULL OR minutiae_count = 0 
    GROUP BY impression_id 
    ORDER BY impression_id
""")
print("=== Zero-Minutiae Count by Impression ===")
for r in cur.fetchall():
    print(f"Impression {r['impression_id']}: {r['cnt']} zero-minutiae records")

cur.execute("""
    SELECT COUNT(DISTINCT person_id) as persons_affected 
    FROM fingerprints 
    WHERE minutiae_count IS NULL OR minutiae_count = 0
""")
print("Distinct persons affected:", cur.fetchone()['persons_affected'])

# Check image paths of these 78 records to see why minutiae count was 0
cur.execute("""
    SELECT fingerprint_id, person_id, impression_id, image_path 
    FROM fingerprints 
    WHERE minutiae_count IS NULL OR minutiae_count = 0 
    LIMIT 10
""")
print("\nSample records:")
for r in cur.fetchall():
    print(f"ID={r['fingerprint_id']}, Person={r['person_id']}, Imp={r['impression_id']}, Path={r['image_path']}")

# Check total records in database
cur.execute("SELECT COUNT(*) as total FROM fingerprints")
print("\nTotal fingerprints in db:", cur.fetchone()['total'])

# Test deserialization and structure of ALL 1500 templates
cur.execute("SELECT fingerprint_id, minutiae_count, minutiae_template FROM fingerprints")
all_rows = cur.fetchall()
valid_unpickled = 0
zero_minutiae = 0
has_minutiae_array = 0
minutiae_counts = []

for r in all_rows:
    raw = r['minutiae_template']
    if raw is None:
        continue
    try:
        obj = pickle.loads(raw)
        valid_unpickled += 1
        m = getattr(obj, 'minutiae', None)
        if m is not None:
            has_minutiae_array += 1
            l = len(m)
            minutiae_counts.append(l)
            if l == 0:
                zero_minutiae += 1
    except Exception as e:
        pass

print(f"\nTotal scanned: {len(all_rows)}")
print(f"Successfully unpickled: {valid_unpickled}")
print(f"Has .minutiae attribute: {has_minutiae_array}")
print(f"Records with len(minutiae) == 0: {zero_minutiae}")
print(f"Records with len(minutiae) > 0: {len(minutiae_counts) - zero_minutiae}")
if minutiae_counts:
    non_zero = [c for c in minutiae_counts if c > 0]
    print(f"Non-zero minutiae stats: min={min(non_zero)}, max={max(non_zero)}, mean={sum(non_zero)/len(non_zero):.1f}")

cur.close()
conn.close()
