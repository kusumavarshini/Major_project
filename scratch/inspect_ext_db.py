import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="external_fingerprint_db"
)
cursor = conn.cursor()
cursor.execute("SHOW TABLES;")
tables = cursor.fetchall()
print("Tables in external_fingerprint_db:", tables)

for (table_name,) in tables:
    print(f"\n--- Schema for table: {table_name} ---")
    cursor.execute(f"DESCRIBE `{table_name}`;")
    for col in cursor.fetchall():
        print(col)
    
    cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
    cnt = cursor.fetchone()[0]
    print(f"Row count in {table_name}: {cnt}")
    
    cursor.execute(f"SHOW CREATE TABLE `{table_name}`;")
    create_stmt = cursor.fetchone()[1]
    print(f"Create statement:\n{create_stmt}")

cursor.close()
conn.close()
