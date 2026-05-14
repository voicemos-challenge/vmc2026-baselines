#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import time

import torch
import torchaudio
from tqdm import tqdm

from model import Model


def main():
    parser = argparse.ArgumentParser(description="Inference for VoiceMOS 2026 Track 3 (Zero-shot or Fine-tuned).")
    parser.add_argument("--data-root", required=True, type=str, help="Root directory of the dataset distribution.")
    parser.add_argument("--csv-path", required=True, type=str, help="CSV file path to do inference (e.g. sets/dev.csv).")
    parser.add_argument("--out", type=str, required=True, help="Path to save the output predictions CSV.")
    parser.add_argument("--model-name", type=str, default="speechbrain/spkrec-ecapa-voxceleb", help="SpeechBrain model ID.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to fine-tuned .pt checkpoint. If not provided, runs zero-shot.")
    parser.add_argument("--target-metric", type=str, default="spk_sim", choices=["spk_sim", "acc_sim"], help="Metric to predict.")
    parser.add_argument("--verbose", type=int, default=1, help="logging level.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    is_zero_shot = args.checkpoint is None

    # 1. Load Model dynamically based on the mode
    if is_zero_shot:
        logging.info(f"Initializing Model in ZERO-SHOT mode using {args.model_name}...")
        model = Model(
            model_name=args.model_name,
            use_projection=False,
            freeze_ssl=True,
            mlp_heads=[]
        )
    else:
        logging.info(f"Initializing Model in FINE-TUNED mode for {args.target_metric}...")
        model = Model(
            model_name=args.model_name,
            use_projection=True,
            freeze_ssl=False,
            mlp_heads=[args.target_metric]
        )
        logging.info(f"Loading weights from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        
    model.to(device)
    model.eval()

    # 2. Load Dataset Metadata
    logging.info(f"Loading Dataset from {args.csv_path}")
    dataset = []
    with open(args.csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset.append(row)
    logging.info(f"Number of inference samples = {len(dataset)}.")

    out_results = []
    start_time = time.time()
    
    # 3. Inference Loop
    for batch in tqdm(dataset, desc="[inference]"):
        wav_a_rel = batch.get("wav_a_path")
        wav_b_rel = batch.get("wav_b_path")
        
        if not wav_a_rel or not wav_b_rel:
            logging.warning(f"Skipping row - missing audio paths: {batch}")
            continue
            
        # Resolve absolute paths based on the provided data root
        wav_a_path = os.path.join(args.data_root, wav_a_rel)
        wav_b_path = os.path.join(args.data_root, wav_b_rel)
        
        try:
            # Load audio using torchaudio
            wav_a, sr_a = torchaudio.load(wav_a_path)
            wav_b, sr_b = torchaudio.load(wav_b_path)
            
            # Ensure 16kHz sample rate
            if sr_a != 16000: wav_a = torchaudio.functional.resample(wav_a, sr_a, 16000)
            if sr_b != 16000: wav_b = torchaudio.functional.resample(wav_b, sr_b, 16000)

            with torch.no_grad():
                # Pass batch=1 tensors to the model
                outputs = model(wav_a.to(device), wav_b.to(device))
                
                if is_zero_shot:
                    # In zero-shot, fallback to the raw cosine similarity computation
                    pred_score = outputs["cos_sim"].item()
                else:
                    # In fine-tuned mode, grab the specific metric projection head output
                    pred_score = outputs[args.target_metric].item()
            
        except Exception as e:
            logging.error(f"Failed to process pair {wav_a_rel} and {wav_b_rel}: {e}")
            continue

        # Build the output row: copy the original data and append our predictions
        out_row = batch.copy()
        out_row[f"pred_{args.target_metric}"] = pred_score
        
        out_results.append(out_row)

    total_time = time.time() - start_time
    logging.info(f"Total inference time = {total_time:.2f} secs.")
    if len(out_results) > 0:
        logging.info(f"Average speed = {total_time / len(out_results):.3f} sec / pair.")

    # 4. Save Results
    if len(out_results) > 0:
        fieldnames = list(out_results[0].keys())
        with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_results)
        logging.info(f"Predictions saved to {args.out}")
    else:
        logging.warning("No results to save.")

if __name__ == "__main__":
    main()