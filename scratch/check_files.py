from pathlib import Path

img_dir = Path("data/clean_fvc")
files = sorted(list(img_dir.glob("*.png")), key=lambda p: int(p.stem) if p.stem.isdigit() else 999999)
print(f"Total files in {img_dir}: {len(files)}")

stems = [int(p.stem) for p in files if p.stem.isdigit()]
print(f"Count of numeric stems: {len(stems)}")
expected = set(range(1, 317))
missing = expected - set(stems)
print(f"Missing from 1..316: {sorted(list(missing))}")
extra = set(stems) - expected
print(f"Extra outside 1..316: {sorted(list(extra))}")
non_numeric = [p.name for p in files if not p.stem.isdigit()]
print(f"Non-numeric pngs: {non_numeric}")
