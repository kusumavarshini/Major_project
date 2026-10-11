import os
from pathlib import Path

img_dir = Path("data/clean_fvc")
all_files = list(img_dir.iterdir())
print(f"Total entries in {img_dir}: {len(all_files)}")
pngs = [f for f in all_files if f.suffix.lower() == ".png"]
print(f"PNG files: {len(pngs)}")

stems = {}
for p in pngs:
    try:
        val = int(p.stem)
        stems[val] = p.name
    except ValueError:
        print("Non-integer stem:", p.name)

for i in range(1, 317):
    if i not in stems:
        print(f"ID {i} is MISSING from {img_dir}")
