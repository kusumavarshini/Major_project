import os
import re
import shutil
from pathlib import Path

# Paths
dataset_root = Path("data/FVC-Latent_Test_Dataset/data")
dest_dir = Path("data/clean_fvc_new")

dest_dir.mkdir(parents=True, exist_ok=True)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# 1. Scan dataset folders
all_subdirs = [d for d in os.listdir(dataset_root) if os.path.isdir(dataset_root / d)]
subdirs = sorted(all_subdirs, key=natural_sort_key)

scanned_folders_count = len(subdirs)
clean_folders_count = 0
clean_folders_without_png = []
clean_png_sources = []

for d in subdirs:
    clean_path = dataset_root / d / "clean"
    if clean_path.exists() and clean_path.is_dir():
        clean_folders_count += 1
        png_files = sorted([f for f in os.listdir(clean_path) if f.lower().endswith('.png')])
        if not png_files:
            clean_folders_without_png.append(d)
        else:
            for f in png_files:
                clean_png_sources.append((d, clean_path / f))

total_clean_pngs = len(clean_png_sources)

# 2. Copy and rename sequentially
copied_count = 0
failures = []
mappings = []

for idx, (folder_name, src_file) in enumerate(clean_png_sources, start=1):
    dest_file = dest_dir / f"{idx}.png"
    try:
        shutil.copy2(src_file, dest_file)
        copied_count += 1
        # Store relative path for cleaner display
        rel_src = f"{folder_name}/clean/{src_file.name}"
        mappings.append((rel_src, f"{idx}.png"))
    except Exception as e:
        failures.append((str(src_file), str(e)))

# 3. Destination count verification
dest_png_files = [f for f in os.listdir(dest_dir) if f.lower().endswith('.png')]
dest_count = len(dest_png_files)

# Print verification results
print("=" * 60)
print("VERIFICATION REPORT")
print("=" * 60)
print(f"1. Total FVC dataset folders scanned: {scanned_folders_count}")
print(f"2. Total clean/ folders found: {clean_folders_count}")
print(f"3. Total clean PNG files found: {total_clean_pngs}")
print(f"4. Total PNG files copied: {copied_count}")
print(f"5. Destination file count: {dest_count}")
print(f"6. Clean folders with no PNG file: {clean_folders_without_png if clean_folders_without_png else 'None'}")
print(f"7. Copy failures: {len(failures)} ({failures})")
print("\n8. First 10 mappings:")
for orig, new_f in mappings[:10]:
    print(f"   {orig} -> {new_f}")

print("\n9. Last 10 mappings:")
for orig, new_f in mappings[-10:]:
    print(f"   {orig} -> {new_f}")

print("\n10. Confirmation:")
print(f"   Source PNG count ({total_clean_pngs}) == Destination PNG count ({dest_count}): {total_clean_pngs == dest_count}")
print("=" * 60)
