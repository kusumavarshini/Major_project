import mysql.connector
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

conn = mysql.connector.connect(
    host=os.getenv('MYSQL_HOST', 'localhost'),
    port=int(os.getenv('MYSQL_PORT', 3306)),
    user=os.getenv('MYSQL_USER', 'root'),
    password=os.getenv('MYSQL_PASSWORD', '1234'),
    database='fingerprint_authentication'
)

test_families = [
    'FAMILY-13', 'FAMILY-14', 'FAMILY-15', 'FAMILY-20', 'FAMILY-23', 
    'FAMILY-36', 'FAMILY-37', 'FAMILY-42', 'FAMILY-48', 'FAMILY-54', 
    'FAMILY-65', 'FAMILY-66', 'FAMILY-67', 'FAMILY-78', 'FAMILY-9'
]

cur = conn.cursor(dictionary=True)
format_strings = ','.join(['%s'] * len(test_families))
cur.execute(f"SELECT person_id, identity_code, family_id, member_type FROM persons WHERE family_id IN ({format_strings})", test_families)
person_rows = cur.fetchall()

families_found = set(r['family_id'] for r in person_rows)
print(f"Test families found in persons table: {len(families_found)} / {len(test_families)}")
print(f"Test persons found in persons table: {len(person_rows)} / 45")

p_ids = [r['person_id'] for r in person_rows]
cur.execute(f"SELECT fingerprint_id, person_id, impression_id, image_path, minutiae_count, (minutiae_template IS NOT NULL) as has_tpl FROM fingerprints WHERE person_id IN ({','.join(['%s']*len(p_ids))})", p_ids)
fp_rows = cur.fetchall()
print(f"Total fingerprint records in db for test persons: {len(fp_rows)} / 225")

df_fp = pd.DataFrame(fp_rows)
print("\nImpression breakdown in db:")
print(df_fp['impression_id'].value_counts())
print("Has template for all:", (df_fp['has_tpl'] == 1).all())
print("Zero minutiae records in test set:", (df_fp['minutiae_count'] == 0).sum())
print("\nSample 5 image paths from db for test set:")
for p in df_fp['image_path'].head(5):
    print(" ", p)

# Check total clean gallery templates in db
cur.execute("SELECT COUNT(*) as total_fingerprints, COUNT(DISTINCT person_id) as total_persons FROM fingerprints")
total_stats = cur.fetchone()
print(f"\nTotal in entire database: {total_stats['total_fingerprints']} fingerprints across {total_stats['total_persons']} persons")

# If gallery is Impression 1 across all 300 persons:
cur.execute("SELECT COUNT(*) as imp1_cnt FROM fingerprints WHERE impression_id = 1")
print(f"Impression 1 enrolled templates in database (full gallery): {cur.fetchone()['imp1_cnt']}")

# If gallery is Impression 1 across only the 45 test persons:
cur.execute(f"SELECT COUNT(*) as test_imp1_cnt FROM fingerprints WHERE impression_id = 1 AND person_id IN ({','.join(['%s']*len(p_ids))})", p_ids)
print(f"Impression 1 enrolled templates for test set: {cur.fetchone()['test_imp1_cnt']}")

cur.close()
conn.close()
