import sys
import os
# pyrefly: ignore [missing-import]
import mysql.connector

print("Python version:", sys.version)

passwords_to_try = ["", "root", "password", "admin", "1234", "123456", "12345678", "kusum", "kusum123"]

for pwd in passwords_to_try:
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password=pwd,
            connection_timeout=5
        )
        print(f"SUCCESS connecting with password: '{pwd}'")
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES;")
        print("Databases:", [r[0] for r in cursor.fetchall()])
        cursor.close()
        conn.close()
        break
    except Exception as e:
        print(f"Failed with password '{pwd}': {e}")
