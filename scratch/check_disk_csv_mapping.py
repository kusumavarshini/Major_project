import os
import csv

clean_dir = os.path.abspath("FingerSure_Dataset/clean")
distorted_dir = os.path.abspath("FingerSure_Dataset/distorted")

clean_files_on_disk = sorted(os.listdir(clean_dir))
distorted_files_on_disk = sorted(os.listdir(distorted_dir))

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

# Collect unique filenames from CSV
csv_filenames = set()
for r in csv_rows:
    csv_filenames.add(r['filename'])

print(f"Total clean files on disk: {len(clean_files_on_disk)}")
print(f"Total distorted files on disk: {len(distorted_files_on_disk)}")
print(f"Total physical files: {len(clean_files_on_disk) + len(distorted_files_on_disk)}")

# Case-sensitive check
exact_matches = []
case_mismatch = []
missing_from_csv = []

for folder, flist in [("clean", clean_files_on_disk), ("distorted", distorted_files_on_disk)]:
    for fn in flist:
        if fn in csv_filenames:
            exact_matches.append((folder, fn))
        else:
            # check case insensitive
            match_ci = [cf for cf in csv_filenames if cf.lower() == fn.lower()]
            if match_ci:
                case_mismatch.append((folder, fn, match_ci[0]))
            else:
                missing_from_csv.append((folder, fn))

print(f"\nExact case-sensitive match with CSV: {len(exact_matches)} files")
print(f"Case mismatch with CSV: {len(case_mismatch)} files")
for folder, disk_fn, csv_fn in case_mismatch:
    print(f"  Disk: '{folder}/{disk_fn}' <---> CSV: '{csv_fn}'")

print(f"Files completely missing from CSV: {len(missing_from_csv)} files")
for folder, disk_fn in missing_from_csv:
    print(f"  Missing: {folder}/{disk_fn}")

print("\n--- Check CSV entries that have NO file on disk ---")
disk_all = set(clean_files_on_disk + distorted_files_on_disk)
disk_all_lower = {fn.lower(): fn for fn in disk_all}

csv_entries_no_disk = []
for r in csv_rows:
    fn = r['filename']
    if fn.lower() not in disk_all_lower:
        csv_entries_no_disk.append(r)

print(f"CSV entries with no file on disk: {len(csv_entries_no_disk)}")
for r in csv_entries_no_disk:
    print(f"  Participant: {r['participant_id']}, Finger: {r['hand']}_{r['finger']}, Cond: {r['condition']}, File: {r['filename']}, Timestamp: {r['capture_timestamp']}")

