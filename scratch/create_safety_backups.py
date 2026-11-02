import os
import sys
import shutil
import csv
import datetime

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from database import get_db_connection

timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
backup_dir = os.path.abspath(f"FingerSure_Dataset/backups/backup_{timestamp_str}")
os.makedirs(backup_dir, exist_ok=True)

print(f"Creating backup in: {backup_dir}")

# 1. Backup metadata.csv
meta_src = "FingerSure_Dataset/metadata.csv"
meta_dst = os.path.join(backup_dir, "metadata_backup.csv")
shutil.copy2(meta_src, meta_dst)
print(f"Backed up metadata.csv -> {meta_dst}")

# 2. Backup failed_captures.csv if exists
failed_src = "FingerSure_Dataset/failed_captures.csv"
if os.path.exists(failed_src):
    failed_dst = os.path.join(backup_dir, "failed_captures_backup.csv")
    shutil.copy2(failed_src, failed_dst)
    print(f"Backed up failed_captures.csv -> {failed_dst}")

# 3. Backup MySQL physical_dataset_collection table
conn = get_db_connection()
cursor = conn.cursor(dictionary=True)

backup_table_name = f"physical_dataset_collection_backup_{timestamp_str}"
cursor.execute(f"CREATE TABLE `{backup_table_name}` AS SELECT * FROM physical_dataset_collection")
conn.commit()
print(f"Backed up MySQL physical_dataset_collection -> `{backup_table_name}`")

# Also dump to CSV
cursor.execute("SELECT * FROM physical_dataset_collection")
db_rows = cursor.fetchall()

db_csv_dst = os.path.join(backup_dir, "mysql_physical_dataset_collection_dump.csv")
if db_rows:
    with open(db_csv_dst, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(db_rows[0].keys()))
        writer.writeheader()
        writer.writerows(db_rows)
print(f"Dumped MySQL records ({len(db_rows)} rows) -> {db_csv_dst}")

cursor.close()
conn.close()

print("Safety backups completed successfully!")

