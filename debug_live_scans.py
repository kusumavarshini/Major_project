#!/usr/bin/env python3
"""
Systematic debugging of inconsistent live scan results.
Captures multiple scans, logs all processing steps, and identifies root cause.
"""

import os
import time
import pickle
import hashlib
import json
import cv2
import numpy as np
import mysql.connector
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Import system components
from database import get_connection_config
from matcher import MATCHER, SourceAFISMatcher
from authentication import classify_distortion, rectify_with_ddrnet
from mysql_auth import get_all_gallery_templates

@dataclass
class ScanResult:
    """Complete scan processing result with all diagnostic data."""
    timestamp: str
    image_hash: str
    image_path: str
    original_dims: tuple
    cnn_clean_prob: float
    cnn_distorted_prob: float
    cnn_predicted_class: str
    rectification_applied: bool
    processed_image_hash: str
    processed_dims: tuple
    template_minutiae_count: int
    top_candidates: List[Dict]
    processing_time_ms: float

class LiveScanDebugger:
    """Debug framework for analyzing live scan inconsistencies."""
    
    def __init__(self):
        self.output_dir = Path("experiments/live_scan_debug")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize matcher
        self.matcher = SourceAFISMatcher() if not isinstance(MATCHER, SourceAFISMatcher) else MATCHER
        
        # Load gallery once and cache
        print("Loading gallery templates...")
        self.gallery = get_all_gallery_templates(force_refresh=True)
        print(f"Gallery loaded: {len(self.gallery)} templates")
        
        # Verify ID 318 is present
        self.id_318_record = None
        for record in self.gallery:
            if record.get('fingerprint_id') == 318:
                self.id_318_record = record
                break
        
        if self.id_318_record:
            print(f"✓ ID 318 found in gallery: {self.id_318_record['minutiae_count']} minutiae")
        else:
            print("❌ ID 318 not found in gallery")

    def hash_image(self, image: np.ndarray) -> str:
        """Generate consistent hash of image data."""
        return hashlib.sha256(image.tobytes()).hexdigest()[:16]
    
    def save_debug_image(self, image: np.ndarray, prefix: str, scan_id: str) -> str:
        """Save image for debugging with consistent naming."""
        filename = f"{prefix}_{scan_id}.png"
        filepath = self.output_dir / filename
        cv2.imwrite(str(filepath), image)
        return str(filepath)
    
    def test_template_consistency(self) -> Dict[str, Any]:
        """Test that gallery templates load consistently."""
        print("\n=== Testing Template Consistency ===")
        
        config = get_connection_config()
        config['database'] = 'newdb'
        conn = mysql.connector.connect(**config)
        cursor = conn.cursor()
        
        # Test loading template multiple times
        template_tests = {}
        
        for test_id in [1, 100, 200, 317, 318]:  # Sample across ID range including our record
            cursor.execute("""
                SELECT fingerprint_id, template_data, minutiae_count 
                FROM external_fvc_clean_gallery 
                WHERE fingerprint_id = %s
            """, (test_id,))
            
            db_record = cursor.fetchone()
            if not db_record:
                continue
                
            fid, template_data, minutiae_count = db_record
            
            # Test multiple deserializations
            hashes = []
            for i in range(3):
                try:
                    template = pickle.loads(template_data)
                    template_hash = hashlib.sha256(str(sorted(template.items())).encode()).hexdigest()[:16]
                    hashes.append(template_hash)
                except Exception as e:
                    hashes.append(f"ERROR: {e}")
            
            template_tests[test_id] = {
                'minutiae_count': minutiae_count,
                'hashes_consistent': len(set(hashes)) == 1,
                'hashes': hashes
            }
        
        conn.close()
        return template_tests
    
    def process_single_scan(self, image: np.ndarray, scan_id: str) -> ScanResult:
        """Process a single scan through the complete pipeline."""
        start_time = time.time()
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Step 1: Original image analysis
        original_hash = self.hash_image(image)
        original_dims = image.shape
        original_path = self.save_debug_image(image, "original", scan_id)
        
        print(f"\nProcessing scan {scan_id}:")
        print(f"  Original: {original_dims}, hash: {original_hash}")
        
        # Step 2: CNN Classification
        try:
            cnn_result = classify_distortion(image)
            clean_prob = float(cnn_result.get('clean_probability', 0))
            distorted_prob = float(cnn_result.get('distorted_probability', 0))
            predicted_class = cnn_result.get('predicted_class', 'unknown')
            
            print(f"  CNN: Clean={clean_prob:.3f}, Distorted={distorted_prob:.3f}, Class={predicted_class}")
        except Exception as e:
            print(f"  CNN Error: {e}")
            clean_prob = distorted_prob = 0.0
            predicted_class = "error"
        
        # Step 3: Rectification Decision
        rectification_applied = False
        processed_image = image.copy()
        
        if predicted_class == "distorted":
            try:
                rectified_result = rectify_with_ddrnet(image)
                if rectified_result and 'rectified_image' in rectified_result:
                    processed_image = rectified_result['rectified_image']
                    rectification_applied = True
                    print(f"  DDRNet: Applied rectification")
                else:
                    print(f"  DDRNet: Failed, using original")
            except Exception as e:
                print(f"  DDRNet Error: {e}, using original")
        
        # Step 4: Processed image analysis
        processed_hash = self.hash_image(processed_image)
        processed_dims = processed_image.shape
        processed_path = self.save_debug_image(processed_image, "processed", scan_id)
        
        print(f"  Processed: {processed_dims}, hash: {processed_hash}")
        
        # Step 5: Template extraction
        try:
            template = self.matcher.extract_template(processed_image)
            minutiae_count = template.get('minutiae_count', 0)
            print(f"  Template: {minutiae_count} minutiae")
        except Exception as e:
            print(f"  Template Error: {e}")
            template = {'minutiae_count': 0}
            minutiae_count = 0
        
        # Step 6: 1:N Identification
        top_candidates = []
        if minutiae_count >= 2:
            try:
                result = self.matcher.identify_1toN(
                    probe_template=template,
                    gallery=self.gallery,
                    top_n=10,  # Get more candidates for analysis
                    threshold=45.0
                )
                
                top_candidates = result.get('candidates', [])
                print(f"  Matching: {len(top_candidates)} candidates")
                
                # Log top 3 for immediate visibility
                for i, candidate in enumerate(top_candidates[:3], 1):
                    fid = candidate.get('fingerprint_id')
                    score = candidate.get('score', 0)
                    match = candidate.get('match', False)
                    print(f"    {i}. ID {fid}: Score {score}, Match {match}")
                    
            except Exception as e:
                print(f"  Matching Error: {e}")
        
        processing_time = (time.time() - start_time) * 1000
        
        return ScanResult(
            timestamp=timestamp,
            image_hash=original_hash,
            image_path=original_path,
            original_dims=original_dims,
            cnn_clean_prob=clean_prob,
            cnn_distorted_prob=distorted_prob,
            cnn_predicted_class=predicted_class,
            rectification_applied=rectification_applied,
            processed_image_hash=processed_hash,
            processed_dims=processed_dims,
            template_minutiae_count=minutiae_count,
            top_candidates=top_candidates,
            processing_time_ms=processing_time
        )
    
    def test_same_image_multiple_times(self, image: np.ndarray, iterations: int = 5) -> List[ScanResult]:
        """Process the same image multiple times to test consistency."""
        print(f"\n=== Testing Same Image {iterations} Times ===")
        
        results = []
        base_hash = self.hash_image(image)
        
        for i in range(iterations):
            scan_id = f"same_{base_hash}_{i+1}"
            result = self.process_single_scan(image, scan_id)
            results.append(result)
        
        return results
    
    def analyze_consistency(self, results: List[ScanResult]) -> Dict[str, Any]:
        """Analyze consistency across multiple runs."""
        print(f"\n=== Consistency Analysis ===")
        
        if not results:
            return {"error": "No results to analyze"}
        
        # Check input consistency
        input_hashes = [r.image_hash for r in results]
        input_consistent = len(set(input_hashes)) == 1
        print(f"Input consistency: {input_consistent}")
        
        # Check CNN consistency
        cnn_classes = [r.cnn_predicted_class for r in results]
        cnn_consistent = len(set(cnn_classes)) == 1
        print(f"CNN consistency: {cnn_consistent} (classes: {set(cnn_classes)})")
        
        # Check rectification consistency
        rectifications = [r.rectification_applied for r in results]
        rect_consistent = len(set(rectifications)) == 1
        print(f"Rectification consistency: {rect_consistent}")
        
        # Check processed image consistency
        processed_hashes = [r.processed_image_hash for r in results]
        processed_consistent = len(set(processed_hashes)) == 1
        print(f"Processed image consistency: {processed_consistent}")
        
        # Check template consistency
        minutiae_counts = [r.template_minutiae_count for r in results]
        template_consistent = len(set(minutiae_counts)) == 1
        print(f"Template consistency: {template_consistent} (counts: {set(minutiae_counts)})")
        
        # Check top match consistency
        top_matches = [r.top_candidates[0]['fingerprint_id'] if r.top_candidates else None for r in results]
        match_consistent = len(set(top_matches)) == 1
        print(f"Top match consistency: {match_consistent} (IDs: {set(top_matches)})")
        
        # Check if ID 318 appears consistently
        id_318_positions = []
        id_318_scores = []
        
        for result in results:
            position = None
            score = None
            for i, candidate in enumerate(result.top_candidates, 1):
                if candidate.get('fingerprint_id') == 318:
                    position = i
                    score = candidate.get('score', 0)
                    break
            id_318_positions.append(position)
            id_318_scores.append(score)
        
        print(f"ID 318 positions: {id_318_positions}")
        print(f"ID 318 scores: {id_318_scores}")
        
        return {
            "input_consistent": input_consistent,
            "cnn_consistent": cnn_consistent,
            "rectification_consistent": rect_consistent,
            "processed_image_consistent": processed_consistent,
            "template_consistent": template_consistent,
            "top_match_consistent": match_consistent,
            "id_318_positions": id_318_positions,
            "id_318_scores": id_318_scores,
            "top_matches": top_matches
        }
    
    def test_id_318_directly(self) -> Dict[str, Any]:
        """Test ID 318's stored template against itself and the enrolled image."""
        print(f"\n=== Testing ID 318 Directly ===")
        
        if not self.id_318_record:
            return {"error": "ID 318 not found in gallery"}
        
        # Test self-match
        id_318_template = self.id_318_record['template']
        self_match = self.matcher.match_1to1(id_318_template, id_318_template)
        print(f"ID 318 self-match: Score {self_match.score}, Match {self_match.match}")
        
        # Load and test against enrolled image
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
                enrolled_image = cv2.imread(enrollment_record[0], cv2.IMREAD_GRAYSCALE)
                enrolled_template = self.matcher.extract_template(enrolled_image)
                
                enrolled_match = self.matcher.match_1to1(id_318_template, enrolled_template)
                print(f"ID 318 vs enrolled image: Score {enrolled_match.score}, Match {enrolled_match.match}")
                
                # Test enrolled image in 1:N search
                enrolled_1toN = self.matcher.identify_1toN(enrolled_template, self.gallery, top_n=5)
                enrolled_top = enrolled_1toN.get('candidates', [])
                
                print(f"Enrolled image 1:N search:")
                for i, candidate in enumerate(enrolled_top[:3], 1):
                    fid = candidate.get('fingerprint_id')
                    score = candidate.get('score', 0)
                    print(f"  {i}. ID {fid}: Score {score}")
                
                return {
                    "self_match_score": self_match.score,
                    "enrolled_match_score": enrolled_match.score,
                    "enrolled_1toN_top": enrolled_top
                }
            else:
                print("❌ Enrolled image not found")
                return {"error": "Enrolled image not accessible"}
                
        except Exception as e:
            print(f"❌ Error testing enrolled image: {e}")
            return {"error": str(e)}
    
    def generate_diagnostic_report(self, all_results: List[ScanResult], consistency_analysis: Dict) -> str:
        """Generate comprehensive diagnostic report."""
        report_lines = []
        
        report_lines.append("=== LIVE SCAN DEBUGGING DIAGNOSTIC REPORT ===")
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total scans analyzed: {len(all_results)}")
        report_lines.append("")
        
        # Consistency Summary
        report_lines.append("=== CONSISTENCY ANALYSIS ===")
        for key, value in consistency_analysis.items():
            if isinstance(value, bool):
                status = "✓ PASS" if value else "❌ FAIL"
                report_lines.append(f"{key}: {status}")
            else:
                report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # Detailed scan results table
        report_lines.append("=== DETAILED SCAN RESULTS ===")
        report_lines.append("Scan | Input_Hash | CNN_Class | Rectified | Processed_Hash | Minutiae | Top_ID | Top_Score | ID_318_Pos | ID_318_Score")
        report_lines.append("-" * 130)
        
        for i, result in enumerate(all_results, 1):
            top_id = result.top_candidates[0]['fingerprint_id'] if result.top_candidates else 'None'
            top_score = result.top_candidates[0]['score'] if result.top_candidates else 0
            
            # Find ID 318 position and score
            id_318_pos = 'None'
            id_318_score = 'None'
            for pos, candidate in enumerate(result.top_candidates, 1):
                if candidate.get('fingerprint_id') == 318:
                    id_318_pos = pos
                    id_318_score = candidate.get('score', 0)
                    break
            
            report_lines.append(f"{i:4d} | {result.image_hash:10s} | {result.cnn_predicted_class:9s} | {str(result.rectification_applied):9s} | {result.processed_image_hash:10s} | {result.template_minutiae_count:8d} | {top_id:6s} | {top_score:9.1f} | {id_318_pos:10s} | {id_318_score}")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_path = self.output_dir / f"diagnostic_report_{int(time.time())}.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"\n📋 Diagnostic report saved: {report_path}")
        return report_content

def main():
    """Main debugging routine."""
    debugger = LiveScanDebugger()
    
    # Step 1: Test template consistency
    template_tests = debugger.test_template_consistency()
    print(f"Template consistency tests: {template_tests}")
    
    # Step 2: Test ID 318 directly
    id_318_tests = debugger.test_id_318_directly()
    print(f"ID 318 direct tests: {id_318_tests}")
    
    # Step 3: Load a test image (use the enrolled image)
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
            print(f"✓ Loaded test image: {test_image.shape}")
            
            # Step 4: Test same image multiple times
            same_image_results = debugger.test_same_image_multiple_times(test_image, iterations=5)
            
            # Step 5: Analyze consistency
            consistency_analysis = debugger.analyze_consistency(same_image_results)
            
            # Step 6: Generate diagnostic report
            report = debugger.generate_diagnostic_report(same_image_results, consistency_analysis)
            
            print("\n" + "="*50)
            print("DEBUGGING COMPLETE")
            print("="*50)
            print(report)
            
        else:
            print("❌ Cannot load test image for debugging")
            
    except Exception as e:
        print(f"❌ Main debugging error: {e}")

if __name__ == "__main__":
    main()