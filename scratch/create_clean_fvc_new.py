import os
import shutil
from pathlib import Path

source_dir = Path("data/external_fvc/VC2000DbsDb2a1/clean")
dest_dir = Path("data/clean_fvc_new")

# Create destination directory
dest_dir.mkdir(parents=True, exist_ok=True)

# List source files
all_source_files = os.listdir(source_dir) if source_dir.exists() else []
png_files = sorted([f for f in all_source_files if f.lower().endswith('.png')])

print(f"Total files in source: {len(all_source_files)}")
print(f"Source files: {all_source_files}")
print(f"Source PNG count: {len(png_files)}")

# Copy and rename PNG files
mappings = []
failures = 0
for idx, f in enumerate(png_files, start=1):
    src_path = source_dir / f
    dest_path = dest_dir / f"{idx}.png"
    try:
        shutil.copy2(src_path, dest_path)
        mappings.append((f, f"{idx}.png"))
    except Exception as e:
        print(f"Failed to copy {f}: {e}")
        failures += 1

dest_files = os.listdir(dest_dir) if dest_dir.exists() else []
dest_pngs = [f for f in dest_files if f.lower().endswith('.png')]

print(f"Destination PNG count: {len(dest_pngs)}")
print(f"Copy failures: {failures}")
print(f"Mappings count: {len(mappings)}")
