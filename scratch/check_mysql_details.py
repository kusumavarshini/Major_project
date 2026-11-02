import os
import sys
import mysql.connector

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from database import get_db_connection

conn = get_db_connection()
cursor = conn.cursor(dictionary=True)

query = "SELECT id, participant_id, hand, finger, `condition`, filename, capture_timestamp FROM physical_dataset_collection WHERE participant_id = 'Kusuma'"
cursor.execute(query)
rows = cursor.fetchall()
print(f"Total Kusuma rows in MySQL: {len(rows)}")
for r in rows:
    print(r)

print("\n--- Check all rows in MySQL grouped by (participant_id, hand, finger, condition) ---")
cursor.execute("""
    SELECT participant_id, hand, finger, `condition`, COUNT(*) as cnt
    FROM physical_dataset_collection
    GROUP BY participant_id, hand, finger, `condition`
    HAVING cnt > 1
""")
dups = cursor.fetchall()
print(f"Duplicates in MySQL: {len(dups)}")

print("\n--- Check if ashitta is in MySQL ---")
cursor.execute("SELECT id, participant_id, hand, finger, `condition`, filename, capture_timestamp FROM physical_dataset_collection WHERE participant_id = 'ashitta'")
ashitta_rows = cursor.fetchall()
print(f"Total ashitta rows in MySQL: {len(ashitta_rows)}")
for r in ashitta_rows:
    print(r)

cursor.close()
conn.close()

