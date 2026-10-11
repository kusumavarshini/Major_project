#!/usr/bin/env python3
"""
Cross-reference 20 test IDs with verified ground truth data
"""

import pandas as pd
import json

def main():
    # Load the verified ground truth (most reliable source)
    gt_df = pd.read_csv('latent_fvc_317_1toN_evaluation_fvc_adapted.csv')
    
    # Load our 20 test results
    with open('genuine_pairs_analysis.json', 'r') as f:
        test_data = json.load(f)
    
    print('=== CROSS-REFERENCE: 20 Test IDs vs Verified Ground Truth ===\n')
    
    # Extract test IDs from both genuine_pairs and failed_pairs
    all_test_ids = []
    genuine_pairs = test_data.get('genuine_pairs', {})
    failed_pairs = test_data.get('failed_pairs', {})
    
    all_test_ids.extend([int(id_str) for id_str in genuine_pairs.keys()])
    all_test_ids.extend([int(id_str) for id_str in failed_pairs.keys()])
    
    print(f'Test IDs from genuine_pairs: {sorted([int(id_str) for id_str in genuine_pairs.keys()])}')
    print(f'Test IDs from failed_pairs: {sorted([int(id_str) for id_str in failed_pairs.keys()])}')
    print(f'All Test IDs: {sorted(all_test_ids)}')
    
    # Check which test IDs have verified ground truth
    verified_correspondences = []
    for test_id in all_test_ids:
        gt_row = gt_df[gt_df['query_id'] == test_id]
        if not gt_row.empty:
            verified_correspondences.append({
                'latent_id': test_id,
                'verified_clean_id': gt_row.iloc[0]['ground_truth_id']
            })
    
    print(f'\nVerified correspondences found: {len(verified_correspondences)}/{len(all_test_ids)}')
    for vc in verified_correspondences:
        print(f'  Latent {vc["latent_id"]} -> Clean {vc["verified_clean_id"]}')
    
    # Cross-reference with our test results
    print(f'\n=== CORRESPONDENCE ANALYSIS ===')
    correct_count = 0
    total_with_ground_truth = 0
    
    # Check genuine pairs
    for latent_id_str, result in genuine_pairs.items():
        latent_id = int(latent_id_str)
        
        # Find verified ground truth
        gt_row = gt_df[gt_df['query_id'] == latent_id]
        if gt_row.empty:
            print(f'Latent {latent_id}: NO VERIFIED GROUND TRUTH AVAILABLE')
            continue
        
        total_with_ground_truth += 1
        verified_clean_id = gt_row.iloc[0]['ground_truth_id']
        predicted_clean_id = result.get('predicted_id')
        
        if predicted_clean_id == verified_clean_id:
            status = 'CORRECT CORRESPONDENCE'
            correct_count += 1
        else:
            status = f'INCORRECT (should be {verified_clean_id}, got {predicted_clean_id})'
        
        print(f'Latent {latent_id}: {status}')
        print(f'  Score: {result.get("predicted_score", "N/A")}, Genuine Rank: {result.get("genuine_rank", "N/A")}')
    
    # Check failed pairs
    for latent_id_str, result in failed_pairs.items():
        latent_id = int(latent_id_str)
        
        # Find verified ground truth
        gt_row = gt_df[gt_df['query_id'] == latent_id]
        if gt_row.empty:
            print(f'Latent {latent_id}: NO VERIFIED GROUND TRUTH AVAILABLE')
            continue
        
        total_with_ground_truth += 1
        verified_clean_id = gt_row.iloc[0]['ground_truth_id']
        predicted_clean_id = result.get('predicted_id')
        
        if predicted_clean_id == verified_clean_id:
            status = 'CORRECT CORRESPONDENCE (but marked as failed)'
            correct_count += 1
        else:
            status = f'INCORRECT (should be {verified_clean_id}, got {predicted_clean_id}) - correctly failed'
        
        print(f'Latent {latent_id}: {status}')
        print(f'  Score: {result.get("predicted_score", "N/A")}, Genuine Rank: {result.get("genuine_rank", "N/A")}')
    
    # Summary
    print(f'\n=== SUMMARY ===')
    print(f'Total test IDs: {len(all_test_ids)}')
    print(f'IDs with verified ground truth: {total_with_ground_truth}')
    print(f'Correct correspondences: {correct_count}')
    if total_with_ground_truth > 0:
        accuracy = (correct_count / total_with_ground_truth) * 100
        print(f'Accuracy against verified ground truth: {accuracy:.1f}%')

if __name__ == '__main__':
    main()