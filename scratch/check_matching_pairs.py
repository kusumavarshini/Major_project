from pathlib import Path
import mysql.connector

desktop_latent = Path(r"C:\Users\kusum\OneDrive\Desktop\latent_fvc")

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

available_in_desktop = []
for p in all_pairs:
    latent_name = Path(p["latent_image_path"]).name
    clean_stem = Path(p["clean_image_path"]).stem
    clean_id = int(clean_stem.split("_")[0])
    
    # Check if clean_id is in clean_fvc
    clean_file = Path(f"data/clean_fvc/{clean_id}.png")
    
    if (desktop_latent / latent_name).exists() and clean_file.exists():
        available_in_desktop.append({
            "entry_name": p["entry_name"],
            "latent_filename": latent_name,
            "expected_clean_id": clean_id,
            "clean_file": clean_file,
            "latent_path": desktop_latent / latent_name,
        })

print(f"Total available matching pairs in desktop_latent & clean_fvc: {len(available_in_desktop)}")
print("Sample pairs:")
for item in available_in_desktop[:10]:
    print(f"  latent: {item['latent_filename']:<12} -> expected clean ID: {item['expected_clean_id']}")
