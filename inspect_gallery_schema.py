#!/usr/bin/env python3
"""
Inspect the external_fvc_clean_gallery table schema and existing records
to understand the exact format for insertion.
"""

import mysql.connector
from database import get_connection_config

def inspect_gallery():
    try:
        config = get_connection_config()
        config['database'] = 'newdb'
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        print("=== TABLE SCHEMA ===")
        cursor.execute("DESCRIBE external_fvc_clean_gallery")
        schema = cursor.fetchall()
        for row in schema:
            field, data_type, null, key, default, extra = row
            print(f"{field}: {data_type} {'NOT NULL' if null == 'NO' else 'NULL'} {key} {default or ''} {extra or ''}")
        
        print("\n=== SAMPLE RECORDS ===")
        cursor.execute("""
            SELECT fingerprint_id, minutiae_count, template_version, generation_status, 
                   LENGTH(template_data) as template_size 
            FROM external_fvc_clean_gallery 
            WHERE fingerprint_id <= 3
        """)
        records = cursor.fetchall()
        for row in records:
            fingerprint_id, minutiae_count, template_version, generation_status, template_size = row
            print(f"ID: {fingerprint_id}, Minutiae: {minutiae_count}, Version: {template_version}, Status: {generation_status}, Template Size: {template_size} bytes")
        
        print("\n=== ID RANGE ===")
        cursor.execute("SELECT MIN(fingerprint_id) as min_id, MAX(fingerprint_id) as max_id, COUNT(*) as total FROM external_fvc_clean_gallery")
        min_id, max_id, total = cursor.fetchone()
        print(f"ID Range: {min_id} to {max_id}, Total Records: {total}")
        
        print(f"Next available fingerprint_id: {max_id + 1}")
        
        print("\n=== TEMPLATE DATA SAMPLE ===")
        cursor.execute("SELECT fingerprint_id, template_data FROM external_fvc_clean_gallery WHERE fingerprint_id = 1")
        sample_record = cursor.fetchone()
        if sample_record:
            fingerprint_id, template_data = sample_record
            print(f"Record ID {fingerprint_id}: Template type = {type(template_data)}, Length = {len(template_data)} bytes")
            print(f"First 50 bytes (hex): {template_data[:50].hex() if template_data else 'None'}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    inspect_gallery()