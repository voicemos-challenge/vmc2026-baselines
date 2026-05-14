#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from model import Model


class SimilarityDataset(Dataset):
    def __init__(self, data_root, data_rows, target_metric):
        self.data_root = data_root
        self.target_metric = target_metric
        
        # Aggregate scores by unique audio pairs
        aggregated_data = {}
        for row in data_rows:
            pair_key = (row["wav_a_path"], row["wav_b_path"])
            if pair_key not in aggregated_data:
                aggregated_data[pair_key] = row.copy()
                aggregated_data[pair_key][target_metric] = []
            
            # Accumulate the scores for this specific pair
            aggregated_data[pair_key][target_metric].append(float(row[target_metric]))
            
        self.data_rows = []
        for pair_key, row_data in aggregated_data.items():
            # Calculate the mean score
            scores = row_data[target_metric]
            row_data[target_metric] = sum(scores) / len(scores)
            
            # Remove listener_id since this is an averaged sample
            if "listener_id" in row_data:
                del row_data["listener_id"]
                
            self.data_rows.append(row_data)
            
        logging.info(f"Aggregated {len(data_rows)} raw rows into {len(self.data_rows)} unique averaged pairs.")

    def __len__(self):
        return len(self.data_rows)

    def __getitem__(self, idx):
        row = self.data_rows[idx]
        wav_a_path = os.path.join(self.data_root, row["wav_a_path"])
        wav_b_path = os.path.join(self.data_root, row["wav_b_path"])
        
        # Load audio
        wav_a, sr_a = torchaudio.load(wav_a_path)
        wav_b, sr_b = torchaudio.load(wav_b_path)
        
        # Ensure 16kHz
        if sr_a != 16000: wav_a = torchaudio.functional.resample(wav_a, sr_a, 16000)
        if sr_b != 16000: wav_b = torchaudio.functional.resample(wav_b, sr_b, 16000)

        target = float(row[self.target_metric])

        # Return a dictionary. The collater will dynamically pack the target_metric.
        return {
            "wav_a": wav_a.squeeze(0),
            "wav_b": wav_b.squeeze(0),
            self.target_metric: target
        }


class SimilarityCollater:
    """Collates a batch of similarity dicts into padded tensors. Safely handles missing 'wav_b'."""
    
    def __init__(self, padding_mode="repetitive"):
        self.padding_mode = padding_mode 

    def __call__(self, batch):
        batch_dict = {}
        
        # 1. Sort batch by wav_a length (longest first)
        sorted_batch = sorted(batch, key=lambda x: -x["wav_a"].shape[0])
        bs = len(sorted_batch)
        all_keys = list(sorted_batch[0].keys())

        # Helper function for padding (Zero / Repetitive)
        def pad_features(feats):
            feat_lengths = torch.tensor([feat.shape[0] for feat in feats], dtype=torch.long)
            if self.padding_mode == "zero_padding":
                feats_padded = pad_sequence(feats, batch_first=True, padding_value=0.0)
            elif self.padding_mode == "repetitive":
                max_len = feat_lengths.max().item() # Use max across the specific feature list
                feats_padded = []
                for feat in feats:
                    this_len = feat.shape[0]
                    if this_len == 0:
                        feats_padded.append(torch.zeros(max_len, dtype=feat.dtype))
                        continue
                    dup_times = max_len // this_len
                    remain = max_len - this_len * dup_times
                    to_dup = [feat for _ in range(dup_times)]
                    if remain > 0:
                        to_dup.append(feat[:remain])
                    duplicated_feat = torch.cat(to_dup, dim=0)
                    feats_padded.append(duplicated_feat)
                feats_padded = torch.stack(feats_padded, dim=0)
            else:
                raise NotImplementedError(f"Padding mode {self.padding_mode} not implemented.")
            return feats_padded, feat_lengths
        
        # 2. Process Audio
        wav_a_list = [sorted_batch[i]["wav_a"] for i in range(bs)]
        batch_dict["wav_a"], batch_dict["wav_a_lengths"] = pad_features(wav_a_list)        
        wav_b_list = [item["wav_b"] for item in sorted_batch if item.get("wav_b") is not None]
        batch_dict["wav_b"], batch_dict["wav_b_lengths"] = pad_features(wav_b_list)

        # 3. Explicit ID packing
        if "system_id" in all_keys:
            batch_dict["system_ids"] = [sorted_batch[i]["system_id"] for i in range(bs)]
        if "sample_id" in all_keys:
            batch_dict["sample_ids"] = [sorted_batch[i]["sample_id"] for i in range(bs)]
        if "listener_id" in all_keys:
            batch_dict["listener_ids"] = [sorted_batch[i]["listener_id"] for i in range(bs)]

        # 4. Dynamically pack all metric targets (e.g., 'score', 'mos', 'spk', 'acc')
        ignore_keys = ["wav_path", "wav_a_path", "wav_b_path", "wav_a", "wav_b", 
                       "system_id", "sample_id", "listener_id"]
        
        for key in all_keys:
            if key not in ignore_keys:
                if isinstance(sorted_batch[0][key], (int, float)):
                    batch_dict[key] = torch.tensor([item[key] for item in sorted_batch], dtype=torch.float32)
                elif isinstance(sorted_batch[0][key], str):
                    batch_dict[key] = [item[key] for item in sorted_batch]
                    
        return batch_dict


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Baseline for VoiceMOS 2026 Track 3.")
    parser.add_argument("--data-root", required=True, type=str, help="Root directory of the dataset distribution.")
    parser.add_argument("--target-metric", required=True, type=str, choices=["spk_sim", "acc_sim"], help="Metric to train the projection head on.")
    parser.add_argument("--outdir", required=True, type=str, help="Directory to save the trained checkpoints.")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size.")
    parser.add_argument("--accumulate-steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--train-steps", type=int, default=20000, help="Total number of training steps.")
    parser.add_argument("--save-steps", type=int, default=5000, help="Save a checkpoint every N steps.")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--verbose", type=int, default=1, help="logging level.")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose > 0 else logging.WARN
    logging.basicConfig(level=log_level, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Data Preparation
    train_csv_path = os.path.join(args.data_root, "sets", "train.csv")
    logging.info(f"Loading {train_csv_path}...")
    
    train_data = []
    with open(train_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_data.append(row)

    logging.info(f"Loaded {len(train_data)} training samples.")

    train_dataset = SimilarityDataset(args.data_root, train_data, args.target_metric)
    collater = SimilarityCollater(padding_mode="repetitive")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collater, 
        num_workers=4,
        drop_last=True
    )

    # 2. Model Initialization
    logging.info(f"Initializing Model for {args.target_metric} prediction...")
    model = Model(
        model_name="speechbrain/spkrec-ecapa-voxceleb",
        use_projection=True,
        mlp_heads=[args.target_metric],
        freeze_ssl=False # Fine-tuning everything
    )
    model.to(device)

    # 3. Optimizer & Criterion
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss() 

    # 4. Training Loop
    model.train()
    global_step = 0
    forward_steps = 0
    train_loss = 0.0
    
    optimizer.zero_grad()
    pbar = tqdm(total=args.train_steps, desc="Training")
    
    while global_step < args.train_steps:
        for batch in train_loader:
            if global_step >= args.train_steps:
                break
                
            wav_a = batch["wav_a"].to(device)
            len_a = batch["wav_a_lengths"].to(device)
            wav_b = batch["wav_b"].to(device)
            len_b = batch["wav_b_lengths"].to(device)
            targets = batch[args.target_metric].to(device)

            outputs = model(wav_a, wav_b, len_a, len_b)
            preds = outputs[args.target_metric]
            
            # Loss and Gradient Accumulation
            loss = criterion(preds, targets)
            scaled_loss = loss / args.accumulate_steps
            scaled_loss.backward()
            
            train_loss += loss.item()
            forward_steps += 1
            
            if forward_steps % args.accumulate_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                global_step += 1
                pbar.update(1)
                
                # Update progress bar description every step
                pbar.set_postfix({"MSE": f"{train_loss / args.accumulate_steps:.4f}"})
                train_loss = 0.0

                # Save checkpoint every N steps
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.outdir, f"model_{args.target_metric}_step{global_step}.pt")
                    torch.save(model.state_dict(), save_path)
                    logging.info(f"Checkpoint saved to {save_path}")

    pbar.close()
    
    # 5. Save Final Model
    save_path = os.path.join(args.outdir, f"finetuned_model_{args.target_metric}_final.pt")
    torch.save(model.state_dict(), save_path)
    logging.info(f"Training complete. Final model saved to {save_path}")

if __name__ == "__main__":
    main()