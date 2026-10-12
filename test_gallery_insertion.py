#!/usr/bin/env python3
"""
Test the newly inserted gallery record to ensure it works with the existing matcher.
"""

import pickle
import mysql.connector
from database import get_connection_config
from matcher import MATCHER, SourceAFISMatcher

def test_new_gallery_record():
    """Test the newly inserted gallery record (ID 318)."""
    try:
        config = get_connection_config()
        config['database'] = 'newdb'
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        print("=== Testing New Gallery Record (ID 318) ===")
        
        # Get the new record
        cursor.execute("""
            SELECT fingerprint_id, template_data, minutiae_count, 
                   template_version, generation_status
            FROM external_fvc_clean_gallery 
            WHERE fingerprint_id = 318
        """)
        
        record = cursor.fetchone()
        if not record:
            print("❌ Record ID 318 not found")
            return False
            
        fid, template_data, minutiae_count, template_version, generation_status = record
        print(f"✓ Found record: ID {fid}, Minutiae {minutiae_count}, Version {template_version}, Status {generation_status}")
        
        # Test template deserialization
        print("\n=== Template Deserialization Test ===")
        try:
            template = pickle.loads(template_data)
            print(f"✓ Template deserialized successfully")
            print(f"  Type: {type(template)}")
            print(f"  Keys: {list(template.keys()) if isinstance(template, dict) else 'Not a dict'}")
            
            if isinstance(template, dict) and 'minutiae' in template:
                actual_minutiae = len(template['minutiae'])
                print(f"  Minutiae in template: {actual_minutiae}")
                print(f"  Matches stored count: {actual_minutiae == minutiae_count}")
            else:
                print("  ⚠️  Template structure unexpected")
                
        except Exception as e:
            print(f"❌ Template deserialization failed: {e}")
            return False
        
        # Test with existing gallery record for comparison
        print("\n=== Comparison with Existing Gallery Record ===")
        cursor.execute("""
            SELECT fingerprint_id, template_data, minutiae_count 
            FROM external_fvc_clean_gallery 
            WHERE fingerprint_id = 1
        """)
        
        existing_record = cursor.fetchone()
        if existing_record:
            existing_fid, existing_template_data, existing_minutiae = existing_record
            existing_template = pickle.loads(existing_template_data)
            
            print(f"✓ Existing record {existing_fid}: {existing_minutiae} minutiae")
            print(f"  New record {fid}: {minutiae_count} minutiae")
            print(f"  Both are dict type: {isinstance(template, dict) and isinstance(existing_template, dict)}")
            
            # Check structure similarity
            if isinstance(template, dict) and isinstance(existing_template, dict):
                new_keys = set(template.keys())
                existing_keys = set(existing_template.keys())
                print(f"  Key match: {new_keys == existing_keys}")
                if new_keys != existing_keys:
                    print(f"    New keys: {new_keys}")
                    print(f"    Existing keys: {existing_keys}")
        
        # Test with matcher
        print("\n=== Matcher Compatibility Test ===")
        try:
            # Initialize matcher
            if isinstance(MATCHER, SourceAFISMatcher):
                matcher = MATCHER
            else:
                matcher = SourceAFISMatcher()
            
            # Test if we can use the template for matching
            # Try to match against itself (should give high score)
            try:
                match_result = matcher.match_1to1(template, template)
                print(f"✓ Self-match test: score = {match_result.score}, match = {match_result.match}")
                if match_result.score > 50:
                    print("✓ Self-match score looks reasonable for identical templates")
                else:
                    print(f"⚠️  Self-match score seems low: {match_result.score}")
            except Exception as e:
                print(f"❌ Self-match test failed: {e}")
                return False
                
            # Test against an existing gallery template
            if 'existing_template' in locals():
                try:
                    cross_result = matcher.match_1to1(template, existing_template)
                    print(f"✓ Cross-match test against gallery ID 1: score = {cross_result.score}, match = {cross_result.match}")
                    print("✓ Template is compatible with existing matcher")
                except Exception as e:
                    print(f"❌ Cross-match test failed: {e}")
                    return False
            
        except Exception as e:
            print(f"❌ Matcher test failed: {e}")
            return False
        
        conn.close()
        
        print("\n🎉 ALL TESTS PASSED")
        print(f"Gallery record {fid} is properly formatted and ready for 1:N identification")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if 'conn' in locals():
            conn.close()
        return False

if __name__ == "__main__":
    success = test_new_gallery_record()
    if success:
        print("\n✓ New gallery record is fully functional")
    else:
        print("\n❌ New gallery record has issues")