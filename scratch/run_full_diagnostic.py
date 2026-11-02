#!/usr/bin/env python3
"""
Diagnostic Script for SourceAFIS Matching Validation (Read-Only).
Performs read-only biometric matching tests, database audits, and verification
without modifying any database records, files, or production logic.
"""

from __future__ import annotations

import sys
import os
import time
import pickle
import hashlib
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import mysql.connector
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env", override=True)

from database import MATCH_THRESHOLD, get_connection_config
from matcher import MATCHER, SourceAFISMatcher
import afis.algorithms.safis as safis
from mysql_auth import get_all_gallery_templates

def natural_sort_key(s: str) -> List[Any]:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def run_diagnostics():
    print("=" * 80)
    print("       SOURCEAFIS BIOMETRIC MATCHING DIAGNOSTIC INVESTIGATION")
    print("=" * 80)
    
    # ---------------------------------------------------------
    # PART 1: DATABASE AUDIT (newdb, fingerprint_external)
    # ---------------------------------------------------------
    print("\n--- [AUDIT 1] Database Tables & Enrolled Identities ---")
    conn_newdb = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "1234"),
        database="newdb"
    )
    cur = conn_newdb.cursor(dictionary=True)
    cur.execute("SELECT COUNT(*) as cnt FROM external_fvc_clean_gallery")
    gallery_cnt = cur.fetchone()["cnt"]
    cur.execute("SELECT COUNT(*) as cnt FROM enrolled_fingerprints")
    enrolled_cnt = cur.fetchone()["cnt"]
    print(f"newdb.external_fvc_clean_gallery rows: {gallery_cnt}")
    print(f"newdb.enrolled_fingerprints rows: {enrolled_cnt}")
    
    # Check for missing templates or non-generated status
    cur.execute("""
        SELECT generation_status, COUNT(*) as cnt 
        FROM external_fvc_clean_gallery 
        GROUP BY generation_status
    """)
    print("Gallery generation status distribution:", cur.fetchall())
    
    cur.execute("SELECT COUNT(*) as cnt FROM external_fvc_clean_gallery WHERE template_data IS NULL")
    null_templates = cur.fetchone()["cnt"]
    print(f"Gallery rows with NULL template_data: {null_templates}")
    
    cur.close()
    conn_newdb.close()
    
    # ---------------------------------------------------------
    # PART 2: METADATA & 1-TO-1 CORRESPONDENCE TRACE
    # ---------------------------------------------------------
    print("\n--- [AUDIT 2] Ground-Truth Correspondence & Identity Mapping ---")
    conn_ext = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "1234"),
        database="fingerprint_external"
    )
    cur_ext = conn_ext.cursor(dictionary=True)
    cur_ext.execute("SELECT pair_id, entry_name, clean_image_path, latent_image_path FROM external_pairs LIMIT 5")
    sample_pairs = cur_ext.fetchall()
    print("Sample external_pairs entries:")
    for sp in sample_pairs:
        print(f"  Pair {sp['pair_id']}: Entry '{sp['entry_name']}' | Clean: {sp['clean_image_path']} | Latent: {sp['latent_image_path']}")
    cur_ext.close()
    conn_ext.close()
    
    dataset_data_dir = PROJECT_ROOT / "data" / "FVC-Latent_Test_Dataset" / "data"
    subdirs = sorted([d.name for d in dataset_data_dir.iterdir() if d.is_dir()], key=natural_sort_key)
    print(f"\nTotal subdirectories in FVC-Latent_Test_Dataset/data: {len(subdirs)}")
    print(f"First 5 subdirs (natural sort): {subdirs[:5]}")
    print(f"Note: VC2000DbsDb2a3 is missing from dataset (jumps from 2 to 4).")
    
    # ---------------------------------------------------------
    # PART 3: VERIFIED GENUINE PAIR MATCHING
    # ---------------------------------------------------------
    print("\n--- [AUDIT 3] Verified Genuine Pairs Matching Tests ---")
    
    verified_pairs = [
        {"folder_idx": 1, "entry": "VC2000DbsDb2a1", "clean_file": "1_2.png", "latent_file": "1_5.png", "finger_id": "Finger 1"},
        {"folder_idx": 2, "entry": "VC2000DbsDb2a2", "clean_file": "2_8.png", "latent_file": "2_7.png", "finger_id": "Finger 2"},
        {"folder_idx": 3, "entry": "VC2000DbsDb2a4", "clean_file": "4_1.png", "latent_file": "4_3.png", "finger_id": "Finger 4"},
        {"folder_idx": 4, "entry": "VC2000DbsDb2a5", "clean_file": "5_1.png", "latent_file": "5_3.png", "finger_id": "Finger 5"},
        {"folder_idx": 5, "entry": "VC2000DbsDb2a6", "clean_file": "6_2.png", "latent_file": "6_4.png", "finger_id": "Finger 6"},
        {"folder_idx": 6, "entry": "VC2000DbsDb2a7", "clean_file": "7_2.png", "latent_file": "7_1.png", "finger_id": "Finger 7"},
        {"folder_idx": 7, "entry": "VC2000DbsDb2a8", "clean_file": "8_2.png", "latent_file": "8_8.png", "finger_id": "Finger 8"},
        {"folder_idx": 8, "entry": "VC2000DbsDb2a9", "clean_file": "9_8.png", "latent_file": "9_6.png", "finger_id": "Finger 9"},
        {"folder_idx": 9, "entry": "VC2000DbsDb2a10", "clean_file": "10_7.png", "latent_file": "10_6.png", "finger_id": "Finger 10"},
        {"folder_idx": 10, "entry": "VC2000DbsDb2a11", "clean_file": "11_2.png", "latent_file": "11_2.png", "finger_id": "Finger 11"},
    ]
    
    print(f"{'Idx':<4} | {'Dataset Entry':<16} | {'Finger':<9} | {'Clean Path':<12} | {'Latent Path':<12} | {'Clean M':<7} | {'Lat M':<6} | {'Score':<7} | {'Pairs':<5} | {'Decision (>=50.0)'}")
    print("-" * 115)
    
    genuine_results = []
    for vp in verified_pairs:
        idx = vp["folder_idx"]
        clean_img_path = PROJECT_ROOT / "data" / "clean_fvc_new" / f"{idx}.png"
        latent_img_path = PROJECT_ROOT / "data" / "Latent_fvc" / f"{idx}.png"
        
        img_c = cv2.imread(str(clean_img_path), cv2.IMREAD_GRAYSCALE)
        img_l = cv2.imread(str(latent_img_path), cv2.IMREAD_GRAYSCALE)
        
        tpl_c = MATCHER.extract_template(img_c)
        tpl_l = MATCHER.extract_template(img_l)
        
        res = MATCHER.match_1to1(tpl_l, tpl_c, threshold=MATCH_THRESHOLD)
        
        mc = tpl_c["minutiae_count"]
        ml = tpl_l["minutiae_count"]
        decision = "ACCEPT" if res.match else "REJECT"
        
        genuine_results.append({
            "idx": idx,
            "entry": vp["entry"],
            "finger": vp["finger_id"],
            "clean_file": f"{idx}.png ({vp['clean_file']})",
            "latent_file": f"{idx}.png ({vp['latent_file']})",
            "clean_minutiae": mc,
            "latent_minutiae": ml,
            "score": res.score,
            "pairs": res.matched_minutiae_count,
            "decision": decision
        })
        print(f"{idx:<4} | {vp['entry']:<16} | {vp['finger_id']:<9} | {idx}.png ({vp['clean_file'][:4]}) | {idx}.png ({vp['latent_file'][:4]}) | {mc:<7} | {ml:<6} | {res.score:<7.2f} | {res.matched_minutiae_count:<5} | {decision}")

    # ---------------------------------------------------------
    # PART 4: VERIFIED IMPOSTOR PAIR MATCHING
    # ---------------------------------------------------------
    print("\n--- [AUDIT 4] Verified Impostor Pairs Matching Tests ---")
    impostor_tests = [
        (1, 2, "Finger 1", "Finger 2"),
        (1, 3, "Finger 1", "Finger 4"),
        (1, 10, "Finger 1", "Finger 11"),
        (2, 1, "Finger 2", "Finger 1"),
        (2, 5, "Finger 2", "Finger 6"),
        (4, 5, "Finger 5", "Finger 6"),
        (5, 6, "Finger 6", "Finger 7"),
        (6, 7, "Finger 7", "Finger 8"),
        (8, 9, "Finger 9", "Finger 10"),
        (1, 83, "Finger 1", "Finger 85 (C83)"),
    ]
    
    print(f"{'Clean ID':<10} | {'Latent ID':<10} | {'Clean Identity':<15} | {'Latent Identity':<15} | {'Score':<7} | {'Pairs':<5} | {'Decision (>=50.0)'}")
    print("-" * 95)
    
    impostor_results = []
    for c_id, l_id, c_name, l_name in impostor_tests:
        clean_img_path = PROJECT_ROOT / "data" / "clean_fvc_new" / f"{c_id}.png"
        latent_img_path = PROJECT_ROOT / "data" / "Latent_fvc" / f"{l_id}.png"
        
        img_c = cv2.imread(str(clean_img_path), cv2.IMREAD_GRAYSCALE)
        img_l = cv2.imread(str(latent_img_path), cv2.IMREAD_GRAYSCALE)
        
        tpl_c = MATCHER.extract_template(img_c)
        tpl_l = MATCHER.extract_template(img_l)
        
        res = MATCHER.match_1to1(tpl_l, tpl_c, threshold=MATCH_THRESHOLD)
        decision = "FALSE ACCEPT" if res.match else "CORRECT REJECT"
        impostor_results.append({
            "clean_id": c_id,
            "latent_id": l_id,
            "clean_name": c_name,
            "latent_name": l_name,
            "score": res.score,
            "pairs": res.matched_minutiae_count,
            "decision": decision
        })
        print(f"Clean {c_id:<4} | Latent {l_id:<4} | {c_name:<15} | {l_name:<15} | {res.score:<7.2f} | {res.matched_minutiae_count:<5} | {decision}")

    # ---------------------------------------------------------
    # PART 5: THRESHOLD AND METRICS FORMULA AUDIT
    # ---------------------------------------------------------
    print("\n--- [AUDIT 5] Threshold & Biometric Metrics Formula Audit ---")
    print(f"Configured MATCH_THRESHOLD in database.py: {MATCH_THRESHOLD}")
    print(f"SAFIS default internal raw threshold: {safis.DEFAULT_RAW_THRESHOLD} (equivalent to 6.15 matched minutiae)")
    print(f"Current threshold 50.0 corresponds to: {50.0 / 6.5:.2f} matched minutiae (minimum 8 minutiae)")
    
    print("\nDiagnostic investigation complete. All tests read-only.")

if __name__ == "__main__":
    run_diagnostics()
