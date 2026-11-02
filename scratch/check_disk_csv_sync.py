import os
import sys
import csv
from collections import defaultdict

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

# Let's check disk files not in CSV, or CSV not in disk
clean_dir = "FingerSure_Dataset/clean"
distorted_dir = "FingerSure_Dataset/distorted"

disk_clean = set(os.listdir(clean_dir))
disk_distorted = set(os.listdir(distorted_dir))

csv_clean = set()
csv_distorted = set()

for r in csv_rows:
    if r['condition'] == '0':
        csv_clean.add(r['filename'])
    else:
        csv_distorted.add(r['filename'])

print(f"Disk clean: {len(disk_clean)}, CSV clean: {len(csv_clean)}")
print(f"Disk distorted: {len(distorted_distorted := disk_distorted)}, CSV distorted: {len(csv_distorted)}")

# Disk clean files not in CSV
disk_c_not_csv = [f for f in disk_clean if f not in csv_clean and f.lower() not in [x.lower() for x in csv_clean]]
print(f"Disk clean not in CSV: {disk_c_not_csv}")

# Disk distorted not in CSV
disk_d_not_csv = [f for f in disk_distorted if f not in csv_distorted and f.lower() not in [x.lower() for x in csv_distorted]]
print(f"Disk distorted not in CSV: {disk_d_not_csv}")

# CSV clean files not on disk
csv_c_not_disk = [f for f in csv_clean if f not in disk_clean and f.lower() not in [x.lower() for x in disk_clean]]
print(f"CSV clean not on disk: {csv_c_not_disk}")

# CSV distorted files not on disk
csv_d_not_disk = [f for f in csv_distorted if f not in disk_distorted and f.lower() not in [x.lower() for x in disk_distorted]]
print(f"CSV distorted not on disk: {csv_d_not_disk}")

