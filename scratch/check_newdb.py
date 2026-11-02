import mysql.connector

conn = mysql.connector.connect(host="localhost", user="root", password="1234", database="newdb")
cursor = conn.cursor(dictionary=True)

cursor.execute("SELECT enrollment_id, subject_id, filename, image_path, minutiae_count, template_version FROM enrolled_fingerprints;")
print("Enrolled fingerprints:", cursor.fetchall())

cursor.execute("SELECT fingerprint_id, minutiae_count, template_version, generation_status FROM external_fvc_clean_gallery WHERE fingerprint_id >= 315;")
print("Gallery recent rows:", cursor.fetchall())

cursor.close()
conn.close()
