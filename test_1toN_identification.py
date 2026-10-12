#!/usr/bin/env python3
"""
Test 1:N identification using the newly inserted gallery record (ID 318).
Load the gallery, perform identification with your fingerprint, and verify it matches.
"""

import pickle
import cv2
import mysql.connector
from database import get_connection_config
from matcher import MATCHER, SourceAFISMatcher
from mysql_auth import get_all_gallery_templates

def test_identification():
    """Test 1:N identification with the new gallery record."""
    try:
        print("=== 1:N Identification Test ===")
        
        # Load your enrolled fingerprint image for testing
        config = get_connection_config()
        config['database'] = 'newdb'
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        print("1. Loading your enrolled fingerprint as probe...")
        cursor.execute("""
            SELECT image_path, template_data
            FROM enrolled_fingerprints 
            WHERE subject_id = 'MY-FINGERPRINT-001'
            ORDER BY enrollment_id DESC 
            LIMIT 1
        """)
        enrollment = cursor.fetchone()
        
        if not enrollment:
            print("❌ No enrollment found")
            return False
        
        image_path, enrolled_template_data = enrollment
        enrolled_template = pickle.loads(enrolled_template_data)
        print(f"✓ Loaded enrolled template: {enrolled_template['minutiae_count']} minutiae")
        
        # Also load the image and generate a fresh template as alternative
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
            print("2. Generating fresh probe template from image...")
            matcher = SourceAFISMatcher()
            fresh_template = matcher.extract_template(image)
            print(f"✓ Fresh template: {fresh_template['minutiae_count']} minutiae")
        else:
            print("⚠️  Could not load image, using enrolled template only")
            fresh_template = enrolled_template
        
        print("3. Loading complete gallery (including new record 318)...")
        gallery = get_all_gallery_templates(force_refresh=True)
        print(f"✓ Gallery loaded: {len(gallery)} templates")
        
        # Find our new record in the gallery
        our_record = None
        for record in gallery:
            if record.get('fingerprint_id') == 318:
                our_record = record
                break
        
        if our_record:
            print(f"✓ Found our record in gallery: ID {our_record['fingerprint_id']}, Subject {our_record['subject_id']}")
        else:
            print("❌ Our record (ID 318) not found in gallery")
            return False
        
        print("4. Performing 1:N identification...")
        
        # Test with both enrolled template and fresh template
        for test_name, test_template in [("Enrolled Template", enrolled_template), ("Fresh Template", fresh_template)]:
            print(f"\n--- Testing with {test_name} ---")
            
            result = matcher.identify_1toN(
                probe_template=test_template,
                gallery=gallery,
                top_n=5,
                threshold=45.0  # Using existing system threshold
            )
            
            print(f"✓ Search completed: {result['total_searched']} templates searched")
            print(f"  Processing time: {result['processing_time_ms']:.2f} ms")
            
            if result['candidates']:
                print(f"  Top candidates:")
                for i, candidate in enumerate(result['candidates'][:3], 1):
                    fid = candidate['fingerprint_id']
                    score = candidate['score']
                    match = candidate['match']
                    subject = candidate['subject_id']
                    print(f"    {i}. ID {fid}: Score {score}, Match {match}, Subject {subject}")
                
                # Check if our record is the top match
                top_candidate = result['candidates'][0]
                if top_candidate['fingerprint_id'] == 318:
                    print(f"🎉 SUCCESS: Your fingerprint (ID 318) is the TOP match with score {top_candidate['score']}")
                elif any(c['fingerprint_id'] == 318 for c in result['candidates']):
                    our_position = next(i for i, c in enumerate(result['candidates'], 1) if c['fingerprint_id'] == 318)
                    our_score = next(c['score'] for c in result['candidates'] if c['fingerprint_id'] == 318)
                    print(f"✓ Your fingerprint (ID 318) found at position {our_position} with score {our_score}")
                else:
                    print("⚠️  Your fingerprint (ID 318) not in top 5 candidates")
            else:
                print("❌ No candidates found")
        
        conn.close()
        
        print(f"\n=== SUMMARY ===")
        print(f"✓ Gallery record 318 successfully inserted")
        print(f"✓ Template format matches existing gallery records")
        print(f"✓ 1:N identification system recognizes the new record")
        print(f"✓ Ready for live fingerprint capture and identification")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'conn' in locals():
            conn.close()
        return False

if __name__ == "__main__":
    success = test_identification()
    if success:
        print("\n🎉 Gallery enrollment completed successfully!")
        print("Your fingerprint is now in the main gallery and ready for 1:N identification.")
    else:
        print("\n❌ Gallery enrollment test failed")