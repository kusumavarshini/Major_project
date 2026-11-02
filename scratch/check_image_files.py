import os
import hashlib

def get_file_info(path):
    if not os.path.exists(path):
        return None
    stat = os.stat(path)
    with open(path, "rb") as f:
        h = hashlib.sha256(f.read()).hexdigest()
    return {
        "path": path,
        "size_bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": h
    }

clean_dir = "FingerSure_Dataset/clean"
distorted_dir = "FingerSure_Dataset/distorted"

print("--- Kusuma files on disk ---")
for folder in [clean_dir, distorted_dir]:
    for f in os.listdir(folder):
        if "kusuma" in f.lower():
            info = get_file_info(os.path.join(folder, f))
            print(f"{folder}/{f}: size={info['size_bytes']}, sha256={info['sha256'][:16]}..., mtime={info['mtime']}")

print("\n--- Check all SHA-256 hashes across clean folder ---")
clean_hashes = {}
clean_dups = []
for f in os.listdir(clean_dir):
    p = os.path.join(clean_dir, f)
    info = get_file_info(p)
    if info['sha256'] in clean_hashes:
        clean_dups.append((f, clean_hashes[info['sha256']]))
    else:
        clean_hashes[info['sha256']] = f

print(f"Total clean files: {len(clean_hashes) + len(clean_dups)}, duplicate binary files: {len(clean_dups)}")
for f1, f2 in clean_dups:
    print(f"  Binary identical clean images: {f1} == {f2}")

print("\n--- Check all SHA-256 hashes across distorted folder ---")
distorted_hashes = {}
distorted_dups = []
for f in os.listdir(distorted_dir):
    p = os.path.join(distorted_dir, f)
    info = get_file_info(p)
    if info['sha256'] in distorted_hashes:
        distorted_dups.append((f, distorted_hashes[info['sha256']]))
    else:
        distorted_hashes[info['sha256']] = f

print(f"Total distorted files: {len(distorted_hashes) + len(distorted_dups)}, duplicate binary files: {len(distorted_dups)}")
for f1, f2 in distorted_dups:
    print(f"  Binary identical distorted images: {f1} == {f2}")

