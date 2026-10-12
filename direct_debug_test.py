#!/usr/bin/env python3
"""
Direct debugging test without using the problematic global MATCHER.
"""

import os
import cv2
import time
import hashlib
import pickle
import mysql.connector
from database import get_connection_config, MATCH_THRESHOLD
from mysql_auth import get_all_gallery_templates
import afis.algorithms.safis as safis

# Import directly from the class
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

class SimpleSourceAFISMatcher:
    """Simplified SourceAFIS matcher for debugging."""
    
    def __init__(self):
        self.default_threshold = MATCH_THRESHOLD
        self.params = safis.SafisParams(
            max_neighbors=8,
            max_edge_length=300.0,
            length_tol=8.0,
            angle_tol=0.35,
            max_roots=15,
        )
    
    def identify_1toN_simple(self, probe_template, gallery, top_n=5):
        """Simplified 1:N identification for debugging."""
        t0 = time.time()
        
        probe_minu = probe_template.get("minutiae")
        if probe_minu is None or len(probe_minu) < 2:
            return {
                "candidates": [],
                "processing_time_ms": 0.0,
                "total_searched": len(gallery)
            }
        
        # Build probe edges
        try:
            ea = safis.build_edges(probe_minu, self.params)
        except Exception as e:
            print(f"Error building probe edges: {e}")
            return {"candidates": [], "processing_time_ms": 0.0, "total_searched": 0}
        
        candidate_results = []
        
        # Simple scoring for each candidate
        for cand_record in gallery:
            cand_tpl = cand_record.get("template", cand_record)
            c_minu = cand_tpl.get("minutiae")
            
            if c_minu is None or len(c_minu) < 2:
                raw_score = 0.0
            else:
                try:
                    # Use direct matching without caching
                    score, raw, pairs = safis.match_minutiae(probe_minu, c_minu, params=self.params)
                    raw_score = float(raw)
                except Exception as e:
                    print(f"Error matching candidate {cand_record.get('fingerprint_id')}: {e}")
                    raw_score = 0.0
            
            candidate_results.append({
                "fingerprint_id": cand_record.get("fingerprint_id"),
                "subject_id": cand_record.get("subject_id"),
                "score": round(raw_score, 2),
                "raw_score": round(raw_score, 2),
                "match": raw_score >= self.default_threshold
            })
        
        # Sort by score descending
        candidate_results.sort(key=lambda x: x["raw_score"], reverse=True)
        
        total_runtime_ms = (time.time() - t0) * 1000.0
        
        return {
            "candidates": candidate_results[:top_n],
            "processing_time_ms": round(total_runtime_ms, 2),
            "total_searched": len(gallery)
        }

def main():
    print("=== DIRECT DEBUG TEST ===")
    
    # Step 1: Load gallery
    print("1. Loading gallery...")
    try:
        gallery = get_all_gallery_templates(force_refresh=True)
        print(f"   ✓ Gallery size: {len(gallery)}")
        
        # Find ID 318
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
            
    except Exception as e:
        print(f"   ❌ Gallery loading error: {e}")
        return
    
    # Step 2: Create matcher
    print("2. Creating matcher...")
    try:
        matcher = SimpleSourceAFISMatcher()
        print(f"   ✓ Matcher created with threshold {matcher.default_threshold}")
    except Exception as e:
        print(f"   ❌ Matcher creation error: {e}")
        return
    
    # Step 3: Load enrolled image
    print("3. Loading enrolled image...")
    try:
        config = get_connection_config()
        config['database'] = 'newdb'
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT image_path FROM enrolled_fingerprints 
            WHERE subject_id = %s
            ORDER BY enrollment_id DESC LIMIT 1
        """, ("MY-FINGERPRINT-001",))
        
        enrollment_record = cursor.fetchone()
        conn.close()
        
        if not enrollment_record or not os.path.exists(enrollment_record[0]):
            print("   ❌ Enrolled image not accessible")
            return
        
        image = cv2.imread(enrollment_record[0], cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("   ❌ Failed to load image")
            return
        
        print(f"   ✓ Image loaded: {image.shape}")
        
    except Exception as e:
        print(f"   ❌ Image loading error: {e}")
        return
    
    # Step 4: Extract template from enrolled image
    print("4. Extracting template from enrolled image...")
    try:
        # Use the same template as stored for ID 318 for direct comparison
        template_318 = id_318_record.get('template')
        if not template_318:
            print("   ❌ ID 318 template not available")
            return
        
        print(f"   ✓ Using ID 318 template: {template_318.get('minutiae_count', 0)} minutiae")
        
    except Exception as e:
        print(f"   ❌ Template error: {e}")
        return
    
    # Step 5: Test matching consistency
    print("5. Testing matching consistency (3 runs)...")
    
    results = []
    
    for i in range(3):
        print(f"\n   --- Run {i+1} ---")
        
        try:
            # Use the same template each time to test consistency
            search_result = matcher.identify_1toN_simple(
                probe_template=template_318,
                gallery=gallery,
                top_n=10
            )
            
            candidates = search_result.get('candidates', [])
            processing_time = search_result.get('processing_time_ms', 0)
            
            print(f"   Processed {search_result.get('total_searched', 0)} candidates in {processing_time:.1f}ms")
            
            # Log top 5
            top_ids = []
            for j, candidate in enumerate(candidates[:5], 1):
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
                'top_ids': top_ids,
                'id_318_pos': id_318_pos,
                'id_318_score': id_318_score,
                'processing_time': processing_time
            })
            
        except Exception as e:
            print(f"   ❌ Run {i+1} error: {e}")
            results.append({
                'run': i + 1,
                'error': str(e)
            })
    
    # Step 6: Analyze results
    print("\n6. Analyzing consistency...")
    
    if results and not any('error' in r for r in results):
        top_ids_all = [r['top_ids'] for r in results]
        id_318_positions = [r['id_318_pos'] for r in results]
        id_318_scores = [r['id_318_score'] for r in results]
        
        # Check consistency
        top_1_consistent = len(set(r['top_ids'][0] if r['top_ids'] else None for r in results)) == 1
        id_318_pos_consistent = len(set(id_318_positions)) == 1
        id_318_score_consistent = len(set(id_318_scores)) == 1
        
        print(f"   Top match consistency: {'✓ PASS' if top_1_consistent else '❌ FAIL'}")
        print(f"   ID 318 position consistency: {'✓ PASS' if id_318_pos_consistent else '❌ FAIL'}")
        print(f"   ID 318 score consistency: {'✓ PASS' if id_318_score_consistent else '❌ FAIL'}")
        
        print(f"\n   Top matches: {[r['top_ids'][0] if r['top_ids'] else None for r in results]}")
        print(f"   ID 318 positions: {id_318_positions}")
        print(f"   ID 318 scores: {id_318_scores}")
        
        # Create diagnostic table
        print("\n7. Diagnostic Summary:")
        print("Run | Top_ID | ID_318_Pos | ID_318_Score | Processing_Time")
        print("-" * 60)
        
        for result in results:
            if 'error' not in result:
                top_id = result['top_ids'][0] if result['top_ids'] else 'None'
                print(f"{result['run']:3d} | {str(top_id)[:6]:6s} | {str(result['id_318_pos'])[:10]:10s} | {str(result['id_318_score'])[:12]:12s} | {result['processing_time']:.1f}ms")
            else:
                print(f"{result['run']:3d} | ERROR  | {result['error'][:30]}")
        
        # Identify root cause
        print(f"\n8. Root Cause Analysis:")
        if top_1_consistent and id_318_pos_consistent and id_318_score_consistent:
            print("   ✅ CONSISTENT: Same template produces identical results")
            print("   ✅ No inconsistency detected in matching algorithm")
            print("   → Problem likely in: Image capture, preprocessing, or template extraction")
        else:
            print("   ❌ INCONSISTENT: Same template produces different results")
            print("   ❌ Problem detected in: Matching algorithm or gallery loading")
            
            if not top_1_consistent:
                print("   → Top match changes between runs")
            if not id_318_pos_consistent:
                print("   → ID 318 position changes between runs")
            if not id_318_score_consistent:
                print("   → ID 318 score changes between runs")
    else:
        print("   ❌ Cannot analyze consistency due to errors")
    
    print("\nDirect debugging completed.")

if __name__ == "__main__":
    main()