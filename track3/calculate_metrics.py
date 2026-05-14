#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
from collections import defaultdict

import numpy as np
import scipy.stats


def compute_metrics(true_scores, pred_scores):
    """Calculates MSE, LCC (Pearson), and SRCC (Spearman)."""
    if len(true_scores) == 0:
        return 0.0, 0.0, 0.0
    
    mse = np.mean((true_scores - pred_scores) ** 2)
    lcc, _ = scipy.stats.pearsonr(true_scores, pred_scores)
    srcc, _ = scipy.stats.spearmanr(true_scores, pred_scores)
    
    return mse, lcc, srcc


def main():
    parser = argparse.ArgumentParser(description="Calculate evaluation metrics for VoiceMOS 2026 Track 3.")
    parser.add_argument("--prediction-csv", required=True, type=str, help="Path to the system's prediction CSV.")
    parser.add_argument("--ground-truth-csv", required=True, type=str, help="Path to the ground truth CSV (e.g. dev_with_labels.csv).")
    parser.add_argument("--verbose", type=int, default=1, help="logging level.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s")

    # 1. Load Ground Truth
    logging.info(f"Loading ground truth from {args.ground_truth_csv}...")
    gt_dict = {}
    with open(args.ground_truth_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Use the audio paths as the unique identifier for the pair
            key = (row['wav_a_path'], row['wav_b_path'])
            gt_dict[key] = {
                'spk_sim': float(row['spk_sim']) if 'spk_sim' in row and row['spk_sim'].strip() else None,
                'acc_sim': float(row['acc_sim']) if 'acc_sim' in row and row['acc_sim'].strip() else None,
                'system_id': row['system_id']
            }
            
    logging.info(f"Loaded {len(gt_dict)} ground truth pairs.")

    # 2. Check Prediction Headers to Determine Target Metrics
    eval_metrics = []
    preds = []
    
    logging.info(f"Loading predictions from {args.prediction_csv}...")
    with open(args.prediction_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames

        if 'pred_spk_sim' in headers:
            eval_metrics.append('spk_sim')
        if 'pred_acc_sim' in headers:
            eval_metrics.append('acc_sim')
            
        if not eval_metrics:
            logging.error("Could not find 'pred_spk_sim' or 'pred_acc_sim' in prediction CSV headers.")
            return

        for row in reader:
            preds.append(row)
            
    logging.info(f"Loaded {len(preds)} prediction rows. Evaluating metrics: {', '.join(eval_metrics)}")

    # 3. Calculate Metrics for each detected target
    for metric in eval_metrics:
        utt_true = []
        utt_pred = []
        
        # Dictionaries to aggregate scores by system for system-level metrics
        sys_true_dict = defaultdict(list)
        sys_pred_dict = defaultdict(list)

        matched_count = 0
        for row in preds:
            key = (row['wav_a_path'], row['wav_b_path'])
            
            if key not in gt_dict:
                continue
                
            t_val = gt_dict[key][metric]
            if t_val is None:
                continue
                
            p_val = float(row[f'pred_{metric}'])
            sys_id = gt_dict[key]['system_id']

            utt_true.append(t_val)
            utt_pred.append(p_val)
            sys_true_dict[sys_id].append(t_val)
            sys_pred_dict[sys_id].append(p_val)
            matched_count += 1

        if matched_count == 0:
            logging.warning(f"No matching pairs found for metric {metric}. Skipping.")
            continue

        utt_true = np.array(utt_true)
        utt_pred = np.array(utt_pred)

        # Average utterance scores to get system-level scores
        sys_true = np.array([np.mean(sys_true_dict[s]) for s in sys_true_dict])
        sys_pred = np.array([np.mean(sys_pred_dict[s]) for s in sys_pred_dict])

        # Compute correlations and MSE
        u_mse, u_lcc, u_srcc = compute_metrics(utt_true, utt_pred)
        s_mse, s_lcc, s_srcc = compute_metrics(sys_true, sys_pred)

        print(f"\n{'='*40}")
        print(f"Results for {metric.upper()}")
        print(f"{'='*40}")
        print(f"Evaluated Pairs : {matched_count}")
        print(f"Evaluated Systems : {len(sys_true)}")
        print("-" * 40)
        print(f"[UTTERANCE LEVEL]")
        print(f"MSE  : {u_mse:.4f}")
        print(f"LCC  : {u_lcc:.4f}")
        print(f"SRCC : {u_srcc:.4f}")
        print("-" * 40)
        print(f"[SYSTEM LEVEL]")
        print(f"MSE  : {s_mse:.4f}")
        print(f"LCC  : {s_lcc:.4f}")
        print(f"SRCC : {s_srcc:.4f}")
        print(f"{'='*40}\n")

if __name__ == "__main__":
    main()