import hashlib
from pathlib import Path
import mysql.connector

# Compare 1.png and 1_2.png
p_clean = Path("data/clean_fvc/1.png")
p_ext = Path(r"C:\Users\kusum\OneDrive\Desktop\fingerprint\external_dataset\FVC-Latent_Test_Dataset\data\VC2000DbsDb2a1\clean\1_2.png")
h1 = hashlib.sha256(p_clean.read_bytes()).hexdigest()
h2 = hashlib.sha256(p_ext.read_bytes()).hexdigest()
print("1.png hash:      ", h1)
print("1_2.png hash:    ", h2)
print("Are they identical?:", h1 == h2)

# Check latent_fvc directory as well
p_lat = Path(r"C:\Users\kusum\OneDrive\Desktop\latent_fvc\1_5.png")
p_ext_lat = Path(r"C:\Users\kusum\OneDrive\Desktop\fingerprint\external_dataset\FVC-Latent_Test_Dataset\data\VC2000DbsDb2a1\latent\1_5.png")
print("latent_fvc 1_5.png exists:", p_lat.exists())
if p_lat.exists():
    h_lat1 = hashlib.sha256(p_lat.read_bytes()).hexdigest()
    h_lat2 = hashlib.sha256(p_ext_lat.read_bytes()).hexdigest()
    print("latent hashes match?:", h_lat1 == h_lat2)

# Check more samples: 2, 100, 200, 316
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="fingerprint_external"
)
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM external_pairs WHERE entry_name IN ('VC2000DbsDb2a1', 'VC2000DbsDb2a2', 'VC2000DbsDb2a100', 'VC2000DbsDb2a200', 'VC2000DbsDb2a316');")
for row in cursor.fetchall():
    print("\nPair:", row["entry_name"])
    print("  clean:", row["clean_image_path"])
    print("  latent:", row["latent_image_path"])
cursor.close()
conn.close()
