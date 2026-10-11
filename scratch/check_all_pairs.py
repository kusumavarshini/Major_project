import os
from pathlib import Path
import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="fingerprint_external"
)
cursor = conn.cursor(dictionary=True)
cursor.execute("SELECT * FROM external_pairs;")
all_pairs = cursor.fetchall()
cursor.close()
conn.close()

print(f"Total pairs in external_pairs: {len(all_pairs)}")

# Check where latent files are located
desktop_latent = Path(r"C:\Users\kusum\OneDrive\Desktop\latent_fvc")
fvc_latent_dataset = Path(r"C:\Users\kusum\OneDrive\Desktop\fingerprint\external_dataset\FVC-Latent_Test_Dataset\data")

found_in_desktop_latent = 0
found_in_fvc_dataset = 0

for p in all_pairs:
    latent_name = Path(p["latent_image_path"]).name  # e.g. 1_5.png
    if (desktop_latent / latent_name).exists():
        found_in_desktop_latent += 1
    
    # In FVC-Latent_Test_Dataset: entry_name/latent/filename
    entry = p["entry_name"]
    if (fvc_latent_dataset / entry / "latent" / latent_name).exists():
        found_in_fvc_dataset += 1

print(f"Found in C:\\Users\\kusum\\OneDrive\\Desktop\\latent_fvc: {found_in_desktop_latent}")
print(f"Found in FVC-Latent_Test_Dataset: {found_in_fvc_dataset}")

# Check first 15 pairs
print("\nFirst 15 pairs:")
for p in all_pairs[:15]:
    clean_stem = Path(p["clean_image_path"]).stem # e.g. 1_2
    clean_id = clean_stem.split("_")[0] # e.g. 1
    latent_name = Path(p["latent_image_path"]).name # e.g. 1_5.png
    print(f"  entry: {p['entry_name']:<18} | clean: {clean_id:>3} ({p['clean_image_path']}) | latent: {latent_name}")
