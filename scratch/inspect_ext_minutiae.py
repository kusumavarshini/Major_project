import pickle
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="fingerprint_external"
)
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM external_minutiae LIMIT 3;")
rows = cursor.fetchall()
for r in rows:
    print(r["minutiae_id"], r["pair_id"], r["image_type"], r["image_path"], r["minutiae_count"], r["image_width"], r["image_height"])
    data = pickle.loads(r["minutiae_template"])
    print("Template type:", type(data))
    if isinstance(data, dict):
        print("Dict keys:", data.keys())
    else:
        print("Object class:", data.__class__.__name__)
        print("Attrs:", [a for a in dir(data) if not a.startswith("_")])
        print("Minutiae shape:", data.minutiae.shape)
        if hasattr(data, "n_minutiae"):
            print("n_minutiae:", data.n_minutiae)

cursor.execute("SELECT * FROM external_pairs LIMIT 5;")
print("\nExternal pairs sample:")
for r in cursor.fetchall():
    print(r)

cursor.close()
conn.close()
