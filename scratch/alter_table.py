import mysql.connector

conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1234",
    database="external_fingerprint_db"
)
cursor = conn.cursor()

# Check current schema
cursor.execute("DESCRIBE external_clean_gallery;")
print("Before alteration:")
for row in cursor.fetchall():
    print(row)

# Alter to allow NULL in template_data
cursor.execute("ALTER TABLE external_clean_gallery MODIFY COLUMN template_data LONGBLOB NULL;")
conn.commit()

cursor.execute("DESCRIBE external_clean_gallery;")
print("\nAfter alteration:")
for row in cursor.fetchall():
    print(row)

cursor.close()
conn.close()
