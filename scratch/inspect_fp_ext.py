import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="fingerprint_external"
)
cursor = conn.cursor()
cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()
print("Tables in fingerprint_external:", tables)
for (tbl,) in tables:
    print(f"\nTable: {tbl}")
    cursor.execute(f"DESCRIBE `{tbl}`;")
    for col in cursor.fetchall():
        print(col)
    cursor.execute(f"SELECT COUNT(*) FROM `{tbl}`;")
    print("Count:", cursor.fetchone()[0])
cursor.close()
conn.close()
