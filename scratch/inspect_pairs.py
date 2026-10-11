import os
from pathlib import Path
import mysql.connector

# 1. Check data directory subfolders
data_dir = Path("data")
print("Folders in data/:")
for p in data_dir.iterdir():
    if p.is_dir():
        count = len(list(p.glob("*.*")))
        print(f"  {p.name}: {count} files/entries")

# 2. Check fingerprint_external.external_pairs
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="fingerprint_external"
)
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM external_pairs LIMIT 10;")
pairs = cursor.fetchall()
print("\nSample 10 rows from fingerprint_external.external_pairs:")
for r in pairs:
    print(r)

cursor.execute("SELECT COUNT(*) FROM external_pairs;")
print("Total rows in external_pairs:", cursor.fetchone())

cursor.close()
conn.close()
