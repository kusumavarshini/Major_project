#!/usr/bin/env python3
"""
Focused debugging tests to identify inconsistent live scan behavior.
"""

import os
import cv2
import time
from debug_live_scans import LiveScanDebugger
from database import get_connection_config
import mysql.connector

def run_focused_debug():
    """Run focused debugging tests."""
    
    print("=== INITIALIZING DEBUGGER ===")
    debugger = LiveScanDebugger()
    
    print(f"Gallery loaded: {len(debugger.gallery)} templates")
    print(f"ID 318 available: {debugger.id_318_record is not None}")
    
    if debugger.id_318_record:
        print(f"ID 318 details: {debugger.id_318_record['minutiae_count']} minutiae, Subject: {debugger.id_318_record['subject_id']}")
    
    # Test 1: Template consistency
    print("\n=== TEST 1: TEMPLATE CONSISTENCY ===")
    template_tests = debugger.test_template_consistency()
    
    for test_id, result in template_tests.items():
        status = "✓ PASS" if result['consistent'] else "❌ FAIL"
        print(f"Template ID {test_id}: {status} - {result['minutiae_count']} minutiae")
    
    # Test 2: ID 318 Direct Testing
    print("\n=== TEST 2: ID 318 DIRECT TESTING ===")
    id_318_result = debugger.test_id_318_directly()
    
    if 'error' in id_318_result:
        print(f"❌ Error: {id_318_result['error']}")
    else:
        print(f"Self-match score: {id_318_result.get('self_match_score', 'N/A')}")
        print(f"Enrolled image match score: {id_318_result.get('enrolled_match_score', 'N/A')}")
        
        enrolled_top = id_318_result.get('enrolled_1toN_top', [])
        print("Enrolled image 1:N results:")
        for i, candidate in enumerate(enrolled_top[:5], 1):
            fid = candidate.get('fingerprint_id')
            score = candidate.get('score', 0)
            match = candidate.get('match', False)
            print(f"  {i}. ID {fid}: Score {score:.1f}, Match {match}")
    
    # Test 3: Load test image
    print("\n=== TEST 3: LOADING TEST IMAGE ===")
    
    try:
        config = get_connection_config()
        config['database'] = 'newdb'
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_path FROM enrolled_fingerprints 
            WHERE subject_id = 'MY-FINGERPRINT-001'
            ORDER BY enrollment_id DESC LIMIT 1
        """)
        
        enrollment_record = cursor.fetchone()
        conn.close()
        
        if enrollment_record and os.path.exists(enrollment_record[0]):
            test_image = cv2.imread(enrollment_record[0], cv2.IMREAD_GRAYSCALE)
            print(f"✓ Loaded test image: {test_image.shape} from {enrollment_record[0]}")
            
            # Test 4: Same image multiple times
            print("\n=== TEST 4: SAME IMAGE MULTIPLE TIMES ===")
            
            results = debugger.test_same_image_multiple_times(test_image, iterations=3)
            
            # Test 5: Consistency analysis
            print("\n=== TEST 5: CONSISTENCY ANALYSIS ===")
            
            consistency = debugger.analyze_consistency(results)
            
            # Test 6: Generate report
            print("\n=== TEST 6: DIAGNOSTIC REPORT ===")
            
            report = debugger.generate_diagnostic_report(results, consistency)
            
            return True
            
        else:
            print("❌ Test image not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Error in testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_focused_debug()
    if success:
        print("\n🎉 Debugging completed successfully")
    else:
        print("\n❌ Debugging failed")