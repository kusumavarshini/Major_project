from pathlib import Path
import mysql.connector

desktop_latent = Path(r"C:\Users\kusum\OneDrive\Desktop\latent_fvc")
fvc_dataset = Path(r"C:\Users\kusum\OneDrive\Desktop\fingerprint\external_dataset\FVC-Latent_Test_Dataset\data")
clean_fvc = Path("data/clean_fvc")

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

# Find pairs where both the latent is in desktop_latent AND clean ID is in clean_fvc
available = []
for p in all_pairs:
    latent_name = Path(p["latent_image_path"]).name
    clean_stem = Path(p["clean_image_path"]).stem
    clean_id = int(clean_stem.split("_")[0])
    clean_file = clean_fvc / f"{clean_id}.png"
    lat_desktop = desktop_latent / latent_name
    lat_fvc = fvc_dataset / p["entry_name"] / "latent" / latent_name
    
    if lat_desktop.exists() and clean_file.exists():
        available.append({
            "entry_name": p["entry_name"],
            "latent_filename": latent_name,
            "expected_clean_id": clean_id,
            "source": "desktop_latent",
        })
    elif lat_fvc.exists() and clean_file.exists():
        available.append({
            "entry_name": p["entry_name"],
            "latent_filename": latent_name,
            "expected_clean_id": clean_id,
            "source": "fvc_dataset",
        })

print(f"Total available pairs: {len(available)}")

# Print first 20 to pick 5 diverse ones
for item in available[:20]:
    print(f"  {item['latent_filename']:<14} -> ID {item['expected_clean_id']:<4} ({item['source']})")

# Check what IDs are in desktop_latent with diverse spread
desktop_pairs = [a for a in available if a["source"] == "desktop_latent"]
print(f"\nDesktop latent pairs count: {len(desktop_pairs)}")
ids = [a["expected_clean_id"] for a in desktop_pairs]
print(f"Desktop latent IDs (sorted): {sorted(ids)}")
