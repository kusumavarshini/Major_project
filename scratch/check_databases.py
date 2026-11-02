import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv(".env")

host = os.getenv("MYSQL_HOST", "localhost")
user = os.getenv("MYSQL_USER", "root")
pwd = os.getenv("MYSQL_PASSWORD", "1234")

try:
    conn = mysql.connector.connect(host=host, user=user, password=pwd)
    cursor = conn.cursor()
    cursor.execute("SHOW DATABASES;")
    dbs = [r[0] for r in cursor.fetchall()]
    print("Databases in MySQL:", dbs)
    for db in dbs:
        if db in ("fvc_db", "newdb", "fingerprint_db", "external_fingerprint_db", "fingerprint_authentication"):
            cursor.execute(f"USE `{db}`;")
            cursor.execute("SHOW TABLES;")
            tables = [r[0] for r in cursor.fetchall()]
            print(f"Tables in {db}:", tables)
            for t in tables:
                cursor.execute(f"SELECT COUNT(*) FROM `{db}`.`{t}`;")
                cnt = cursor.fetchone()[0]
                print(f"  {t}: {cnt} rows")
    cursor.close()
    conn.close()
except Exception as e:
    print("MySQL Error:", e)
