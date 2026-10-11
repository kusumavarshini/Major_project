import inspect
import pickle
import mysql.connector

# 1. Check existing records in fingerprint_authentication
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="fingerprint_authentication"
)
cursor = conn.cursor(dictionary=True)
cursor.execute("SHOW TABLES;")
print("Tables in fingerprint_authentication:", cursor.fetchall())

try:
    cursor.execute("SELECT * FROM fingerprints LIMIT 3;")
    rows = cursor.fetchall()
    print(f"\nSample from fingerprint_authentication.fingerprints ({len(rows)} rows):")
    for r in rows:
        print("Keys:", [k for k in r.keys() if k != "minutiae_template"])
        if "minutiae_template" in r and r["minutiae_template"]:
            data = pickle.loads(r["minutiae_template"])
            print("Type of minutiae_template:", type(data))
            if isinstance(data, dict):
                print("Dict keys:", data.keys())
            else:
                print("Object class:", data.__class__.__name__)
                print("Object attrs:", [a for a in dir(data) if not a.startswith("_")])
        if "minutiae_count" in r:
            print("minutiae_count in DB:", r["minutiae_count"])
except Exception as e:
    print("Error querying fingerprints:", e)

cursor.close()
conn.close()

# 2. Check afis.pipeline
try:
    import afis.pipeline as pipeline
    print("\nafis.pipeline members:", [m for m in dir(pipeline) if not m.startswith("_")])
    if hasattr(pipeline, "FingerprintTemplate"):
        print("FingerprintTemplate:", pipeline.FingerprintTemplate)
        print("FingerprintTemplate init sig:", inspect.signature(pipeline.FingerprintTemplate.__init__))
except Exception as e:
    print("Error checking afis.pipeline:", e)

# 3. Check matcher.py SourceAFISMatcher
try:
    from matcher import SourceAFISMatcher
    print("\nSourceAFISMatcher found successfully!")
except Exception as e:
    print("Error importing SourceAFISMatcher:", e)
