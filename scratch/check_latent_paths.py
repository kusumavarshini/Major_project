from pathlib import Path

# Check latent_fvc on Desktop
d1 = Path(r"C:\Users\kusum\OneDrive\Desktop\latent_fvc")
print(f"Entries in {d1}: {len(list(d1.glob('*.png')))}")

# Check FVC-Latent_Test_Dataset
d2 = Path(r"C:\Users\kusum\OneDrive\Desktop\fingerprint\external_dataset\FVC-Latent_Test_Dataset")
print(f"Exists d2: {d2.exists()}")
if d2.exists():
    print("Subdirs in d2:", [p.name for p in d2.iterdir()])

# Check trial directory
d3 = Path(r"C:\Users\kusum\OneDrive\Desktop\trial")
print("Entries in trial:", [p.name for p in d3.iterdir()])

# Check inside Fingerprint-distortion-detection-for-accurate-recognition
d4 = Path(r"C:\Users\kusum\OneDrive\Desktop\trial\Fingerprint-distortion-detection-for-accurate-recognition\data")
print("Folders in project data:", [p.name for p in d4.iterdir()])
