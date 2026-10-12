#!/usr/bin/env python3
"""
Basic test to check components.
"""

import os
import mysql.connector
from database import get_connection_config

# Test 1: Database connection
print("=== Testing Database ===")
try:
    config = get_connection_config()
    config['database'] = 'newdb'
    conn = mysql.connector.connect(**config)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM external_fvc_clean_gallery")
    count = cursor.fetchone()[0]
    print(f"Gallery records: {count}")
    
    cursor.execute("SELECT image_path FROM enrolled_fingerprints WHERE subject_id = %s ORDER BY enrollment_id DESC LIMIT 1", ("MY-FINGERPRINT-001",))
    record = cursor.fetchone()
    
    if record:
        image_path = record[0]
        print(f"Enrolled image: {image_path}")
        print(f"File exists: {os.path.exists(image_path)}")
    else:
        print("No enrollment found")
    
    conn.close()
    
except Exception as e:
    print(f"Database error: {e}")

# Test 2: Gallery loading
print("\n=== Testing Gallery Loading ===")
try:
    from mysql_auth import get_all_gallery_templates
    gallery = get_all_gallery_templates(force_refresh=True)
    print(f"Gallery size: {len(gallery)}")
    
    # Find ID 318
    id_318 = None
    for record in gallery:
        if record.get('fingerprint_id') == 318:
            id_318 = record
            break
    
    if id_318:
        print(f"ID 318 found: {id_318['minutiae_count']} minutiae, subject: {id_318['subject_id']}")
    else:
        print("ID 318 not found")
        
except Exception as e:
    print(f"Gallery loading error: {e}")

# Test 3: Matcher
print("\n=== Testing Matcher ===")
try:
    from matcher import MATCHER, SourceAFISMatcher
    
    if isinstance(MATCHER, SourceAFISMatcher):
        matcher = MATCHER
    else:
        matcher = SourceAFISMatcher()
    
    print(f"Matcher initialized: {type(matcher)}")
    
    # Test ID 318 self-match if available
    if id_318:
        template_318 = id_318.get('template')
        if template_318:
            self_match = matcher.match_1to1(template_318, template_318)
            print(f"ID 318 self-match: Score {self_match.score}, Match {self_match.match}")
            
except Exception as e:
    print(f"Matcher error: {e}")

print("\nBasic tests completed.")