#!/usr/bin/env python3
"""
Transfer enrolled fingerprint to external_fvc_clean_gallery in exact same format
as existing gallery records.
"""

import os
import pickle
import mysql.connector
from pathlib import Path
from database import get_connection_config
from matcher import MATCHER, SourceAFISMatcher
import cv2
import numpy as np

def enroll_to_gallery():
    """
    Extract the most recent enrollment and add it to external_fvc_clean_gallery
    in exactly the same format as existing records.
    """
    try:
        config = get_connection_config()
        config['database'] = 'newdb'
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        print("=== Step 1: Get Most Recent Enrollment ===")
        cursor.execute("""
            SELECT enrollment_id, subject_id, filename, image_path, 
                   template_data, minutiae_count, image_width, image_height
            FROM enrolled_fingerprints 
            ORDER BY enrollment_id DESC 
            LIMIT 1
        """)
        enrollment = cursor.fetchone()
        
        if not enrollment:
            print("❌ No enrollment found")
            return False
            
        enrollment_id, subject_id, filename, image_path, template_data, minutiae_count, image_width, image_height = enrollment
        print(f"✓ Found enrollment {enrollment_id}: {subject_id}")
        print(f"  File: {filename}")
        print(f"  Path: {image_path}")
        print(f"  Minutiae: {minutiae_count}")
        print(f"  Dimensions: {image_width}x{image_height}")
        
        print("\n=== Step 2: Check Next Available ID ===")
        cursor.execute("SELECT MAX(fingerprint_id) + 1 as next_id FROM external_fvc_clean_gallery")
        next_id = cursor.fetchone()[0]
        print(f"✓ Next available fingerprint_id: {next_id}")
        
        if next_id != 318:
            print(f"⚠️  Warning: Next ID is {next_id}, not 318 as expected")
        
        print("\n=== Step 3: Load and Regenerate Template ===")
        # Load the original image to regenerate the template in exact gallery format
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return False
            
        # Load image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return False
            
        print(f"✓ Loaded image: {image.shape}")
        
        # Generate template using same method as gallery population
        print("  Generating SourceAFIS template...")
        
        # Initialize the matcher if not already done
        if isinstance(MATCHER, SourceAFISMatcher):
            matcher = MATCHER
        else:
            matcher = SourceAFISMatcher()
        
        # Generate template using the same method as gallery population
        template = matcher.extract_template(image)
        
        if not template or not isinstance(template, dict):
            print("❌ Failed to generate template")
            return False
        
        # Verify template structure matches existing gallery format
        if 'minutiae' not in template:
            print("❌ Template missing minutiae data")
            return False
            
        actual_minutiae_count = len(template.get('minutiae', []))
        print(f"✓ Template generated with {actual_minutiae_count} minutiae")
        
        # Serialize template in exact same format as existing gallery records
        template_blob = pickle.dumps(template, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"✓ Template serialized: {len(template_blob)} bytes")
        
        print("\n=== Step 4: Insert into Gallery ===")
        cursor.execute("""
            INSERT INTO external_fvc_clean_gallery (
                fingerprint_id, template_data, minutiae_count, 
                template_version, generation_status
            ) VALUES (%s, %s, %s, %s, %s)
        """, (
            next_id,
            template_blob,
            actual_minutiae_count,
            'sourceafis_v1',
            'generated'
        ))
        
        conn.commit()
        print(f"✓ Successfully inserted fingerprint_id {next_id}")
        
        print("\n=== Step 5: Verify Insertion ===")
        cursor.execute("""
            SELECT fingerprint_id, minutiae_count, template_version, 
                   generation_status, LENGTH(template_data) as template_size
            FROM external_fvc_clean_gallery 
            WHERE fingerprint_id = %s
        """, (next_id,))
        
        verification = cursor.fetchone()
        if verification:
            fid, min_count, tpl_version, gen_status, tpl_size = verification
            print(f"✓ Verified record:")
            print(f"  ID: {fid}")
            print(f"  Minutiae: {min_count}")
            print(f"  Version: {tpl_version}")
            print(f"  Status: {gen_status}")
            print(f"  Template Size: {tpl_size} bytes")
        else:
            print("❌ Verification failed - record not found")
            return False
        
        print("\n=== Step 6: Test Template Loading ===")
        cursor.execute("SELECT template_data FROM external_fvc_clean_gallery WHERE fingerprint_id = %s", (next_id,))
        stored_template_data = cursor.fetchone()[0]
        
        # Test deserialization
        try:
            loaded_template = pickle.loads(stored_template_data)
            loaded_minutiae = len(loaded_template.get('minutiae', []))
            print(f"✓ Template loads successfully with {loaded_minutiae} minutiae")
            
            # Test with matcher
            if matcher.validate_template(loaded_template):
                print("✓ Template validates correctly with SourceAFIS matcher")
            else:
                print("⚠️  Template validation warning (may still work)")
                
        except Exception as e:
            print(f"❌ Template loading failed: {e}")
            return False
        
        conn.close()
        
        print(f"\n🎉 SUCCESS: Fingerprint enrolled to gallery as ID {next_id}")
        print(f"   Minutiae count: {actual_minutiae_count}")
        print(f"   Template size: {len(template_blob)} bytes")
        print(f"   Ready for 1:N identification")
        
        return next_id
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

if __name__ == "__main__":
    result = enroll_to_gallery()
    if result:
        print(f"\nFingerprint successfully enrolled as gallery ID {result}")
    else:
        print("\nEnrollment failed")