#!/usr/bin/env python3
"""
Simple debugging test to identify live scan inconsistencies.
"""

import os
import cv2
import time
import hashlib
import mysql.connector
from database import get_connection_config
from matcher import MATCHER, SourceAFISMatcher
from authentication import classify_distortion, rectify_with_ddrnet
from mysql_auth import get_all_gallery_templates

def hash_image(image):
    """Generate hash of image data."""
    return hashlib.sha256(image.tobytes()).hexdigest()[:16]

def main():
    print("=== SIMPLE DEBUGGING TEST ===")
    
    # Step 1: Load gallery
    print("1. Loading gallery...")
    gallery = get_all_gallery_templates(force_refresh=True)
    print(f"   Gallery size: {len(gallery)}")
    
    # Check ID 318
    id_318_record = None
    for record in gallery:
        if record.get('fingerprint_id') == 318:
            id_318_record = record
            break
    
    if id_318_record:
        print(f"   ✓ ID 318 found: {id_318_record['minutiae_count']} minutiae")
    else:
        print("   ❌ ID 318 not found")
        return
    
    # Step 2: Load test image
    print("2. Loading test image...")
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
    
    if not enrollment_record:
        print("   ❌ No enrollment found")
        return
    
    image_path = enrollment_record[0]
    print(f"   Image path: {image_path}")
    
    if not os.path.exists(image_path):
        print("   ❌ Image file not found")
        return
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("   ❌ Failed to load image")
        return
    
    print(f"   ✓ Image loaded: {image.shape}")
    
    # Step 3: Test same image multiple times
    print("3. Testing same image 3 times...")
    
    matcher = SourceAFISMatcher() if not isinstance(MATCHER, SourceAFISMatcher) else MATCHER
    
    results = []
    
    for i in range(3):
        print(f"\n   --- Run {i+1} ---")
        
        # Hash original image
        original_hash = hash_image(image)
        print(f"   Original hash: {original_hash}")
        
        # CNN classification
        try:
            cnn_result = classify_distortion(image)
            predicted_class = cnn_result.get('predicted_class', 'unknown')
            clean_prob = cnn_result.get('clean_probability', 0)
            print(f"   CNN: {predicted_class} (clean: {clean_prob:.3f})")
        except Exception as e:
            print(f"   CNN error: {e}")
            predicted_class = 'error'
        
        # Process image (rectification if needed)
        processed_image = image.copy()
        rectified = False
        
        if predicted_class == 'distorted':
            try:
                rect_result = rectify_with_ddrnet(image)
                if rect_result and 'rectified_image' in rect_result:
                    processed_image = rect_result['rectified_image']
                    rectified = True
                    print(f"   DDRNet: Applied")
                else:
                    print(f"   DDRNet: Failed")
            except Exception as e:
                print(f"   DDRNet error: {e}")
        
        processed_hash = hash_image(processed_image)
        print(f"   Processed hash: {processed_hash}")
        
        # Extract template
        try:
            template = matcher.extract_template(processed_image)
            minutiae_count = template.get('minutiae_count', 0)
            print(f"   Template: {minutiae_count} minutiae")
        except Exception as e:
            print(f"   Template error: {e}")
            continue
        
        # 1:N identification
        try:
            search_result = matcher.identify_1toN(
                probe_template=template,
                gallery=gallery,
                top_n=5,
                threshold=45.0
            )
            
            candidates = search_result.get('candidates', [])
            print(f"   Matching: {len(candidates)} candidates")
            
            # Log top 3
            top_ids = []
            for j, candidate in enumerate(candidates[:3], 1):
                fid = candidate.get('fingerprint_id')
                score = candidate.get('score', 0)
                match = candidate.get('match', False)
                print(f"     {j}. ID {fid}: Score {score:.1f}, Match {match}")
                top_ids.append(fid)
            
            # Check ID 318 position
            id_318_pos = None
            id_318_score = None
            for pos, candidate in enumerate(candidates, 1):
                if candidate.get('fingerprint_id') == 318:
                    id_318_pos = pos
                    id_318_score = candidate.get('score', 0)
                    break
            
            print(f"     ID 318: Position {id_318_pos}, Score {id_318_score}")
            
            results.append({
                'run': i + 1,
                'original_hash': original_hash,
                'cnn_class': predicted_class,
                'rectified': rectified,
                'processed_hash': processed_hash,
                'minutiae_count': minutiae_count,
                'top_ids': top_ids,
                'id_318_pos': id_318_pos,
                'id_318_score': id_318_score
            })
            
        except Exception as e:
            print(f"   Matching error: {e}")
    
    # Step 4: Analyze consistency
    print("\n4. Consistency Analysis:")
    
    if results:
        original_hashes = [r['original_hash'] for r in results]
        cnn_classes = [r['cnn_class'] for r in results]
        processed_hashes = [r['processed_hash'] for r in results]
        minutiae_counts = [r['minutiae_count'] for r in results]
        top_ids = [r['top_ids'][0] if r['top_ids'] else None for r in results]
        id_318_positions = [r['id_318_pos'] for r in results]
        id_318_scores = [r['id_318_score'] for r in results]
        
        print(f"   Input consistency: {len(set(original_hashes)) == 1}")
        print(f"   CNN consistency: {len(set(cnn_classes)) == 1} ({set(cnn_classes)})")
        print(f"   Processed consistency: {len(set(processed_hashes)) == 1}")
        print(f"   Template consistency: {len(set(minutiae_counts)) == 1} ({set(minutiae_counts)})")
        print(f"   Top match consistency: {len(set(top_ids)) == 1} ({set(top_ids)})")
        print(f"   ID 318 positions: {id_318_positions}")
        print(f"   ID 318 scores: {id_318_scores}")
        
        # Create diagnostic table
        print("\n5. Diagnostic Table:")
        print("Run | Input_Hash | CNN_Class | Rectified | Processed_Hash | Minutiae | Top_ID | ID_318_Pos | ID_318_Score")
        print("-" * 100)
        
        for result in results:
            print(f"{result['run']:3d} | {result['original_hash'][:8]:8s} | {result['cnn_class'][:9]:9s} | {str(result['rectified'])[:9]:9s} | {result['processed_hash'][:8]:8s} | {result['minutiae_count']:8d} | {str(result['top_ids'][0] if result['top_ids'] else 'None')[:6]:6s} | {str(result['id_318_pos'])[:11]:11s} | {result['id_318_score']}")
    else:
        print("   No results to analyze")

if __name__ == "__main__":
    main()