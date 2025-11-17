#!/usr/bin/env python3
"""
Final Training Script - With Multiple Random Negatives

Key improvements:
- Supports multiple negatives per code (N triplets per example)
- Uses all negatives for richer training signal
- Compatible with EmbeddingGemma for better separation
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
import json
import numpy as np
from tqdm import tqdm
from typing import Dict, List

from utils.triplet_monitor import TripletTrainingMonitor, compute_triplet_metrics


class MultiNegativeTripletDataset(TorchDataset):
    """Dataset that returns one anchor/positive and ALL negatives"""

    def __init__(self, data: List[Dict], tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        anchor = item.get('code_snippet', '')
        positive = item.get('conceptual_feedback', item.get('refined_feedback', ''))

        # Get all negatives (support both formats)
        if 'negative_feedbacks' in item:
            negatives = item['negative_feedbacks']
        elif 'hard_negative_feedback' in item:
            # Fallback to single negative
            negatives = [item['hard_negative_feedback']]
        else:
            negatives = [item.get('negative_feedback', '')]

        # Tokenize anchor and positive
        anchor_enc = self.tokenizer(anchor, max_length=self.max_length,
                                   truncation=True, padding='max_length',
                                   return_tensors='pt')
        positive_enc = self.tokenizer(positive, max_length=self.max_length,
                                     truncation=True, padding='max_length',
                                     return_tensors='pt')

        # Tokenize all negatives
        negatives_enc = []
        for neg in negatives:
            neg_enc = self.tokenizer(neg, max_length=self.max_length,
                                   truncation=True, padding='max_length',
                                   return_tensors='pt')
            negatives_enc.append({
                'input_ids': neg_enc['input_ids'].squeeze(0),
                'attention_mask': neg_enc['attention_mask'].squeeze(0)
            })

        return {
            'anchor_input_ids': anchor_enc['input_ids'].squeeze(0),
            'anchor_attention_mask': anchor_enc['attention_mask'].squeeze(0),
            'positive_input_ids': positive_enc['input_ids'].squeeze(0),
            'positive_attention_mask': positive_enc['attention_mask'].squeeze(0),
            'negatives': negatives_enc,  # List of dicts
            'num_negatives': len(negatives)
        }


def collate_multi_negative(batch):
    """Custom collate function for multi-negative batches"""
    anchor_ids = torch.stack([item['anchor_input_ids'] for item in batch])
    anchor_mask = torch.stack([item['anchor_attention_mask'] for item in batch])
    positive_ids = torch.stack([item['positive_input_ids'] for item in batch])
    positive_mask = torch.stack([item['positive_attention_mask'] for item in batch])

    # Flatten all negatives from the batch
    all_neg_ids = []
    all_neg_mask = []
    neg_counts = []

    for item in batch:
        for neg in item['negatives']:
            all_neg_ids.append(neg['input_ids'])
            all_neg_mask.append(neg['attention_mask'])
        neg_counts.append(item['num_negatives'])

    neg_ids = torch.stack(all_neg_ids)
    neg_mask = torch.stack(all_neg_mask)

    return {
        'anchor_input_ids': anchor_ids,
        'anchor_attention_mask': anchor_mask,
        'positive_input_ids': positive_ids,
        'positive_attention_mask': positive_mask,
        'negative_input_ids': neg_ids,
        'negative_attention_mask': neg_mask,
        'neg_counts': neg_counts  # How many negatives per example
    }


def mean_pooling(model_output, attention_mask):
    """Mean pooling with attention mask"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_multi_negative_triplet_loss(anchor_emb, positive_emb, negative_embs, neg_counts, margin=0.5):
    """
    Compute triplet loss with multiple negatives per anchor.

    For each anchor/positive pair, compute loss against ALL negatives,
    then average within each example and across batch.
    """
    batch_size = anchor_emb.size(0)
    total_loss = 0.0

    neg_start_idx = 0
    for i in range(batch_size):
        # Get embeddings for this example
        anch = anchor_emb[i:i+1]  # [1, D]
        pos = positive_emb[i:i+1]  # [1, D]

        # Get all negatives for this example
        num_negs = neg_counts[i]
        negs = negative_embs[neg_start_idx:neg_start_idx + num_negs]  # [N, D]
        neg_start_idx += num_negs

        # Compute distances
        pos_dist = torch.nn.functional.pairwise_distance(anch, pos)  # [1]

        # Distance to each negative
        neg_dists = torch.cdist(anch, negs, p=2).squeeze(0)  # [N]

        # Triplet loss for each negative
        losses = torch.relu(pos_dist + margin - neg_dists)  # [N]

        # Average loss for this example
        example_loss = losses.mean()
        total_loss += example_loss

    # Average across batch
    return total_loss / batch_size


def evaluate_multi_negative(model, val_loader, device, margin=0.5):
    """Evaluate with multiple negatives"""
    model.eval()

    all_metrics = {
        'loss': [],
        'pos_dist': [],
        'neg_dist': [],
        'violations_pct': [],
        'separation': []
    }

    with torch.no_grad():
        for batch in val_loader:
            anchor_ids = batch['anchor_input_ids'].to(device)
            anchor_mask = batch['anchor_attention_mask'].to(device)
            pos_ids = batch['positive_input_ids'].to(device)
            pos_mask = batch['positive_attention_mask'].to(device)
            neg_ids = batch['negative_input_ids'].to(device)
            neg_mask = batch['negative_attention_mask'].to(device)
            neg_counts = batch['neg_counts']

            # Get embeddings
            anchor_out = model(anchor_ids, attention_mask=anchor_mask)
            pos_out = model(pos_ids, attention_mask=pos_mask)
            neg_out = model(neg_ids, attention_mask=neg_mask)

            anchor_emb = mean_pooling(anchor_out, anchor_mask)
            pos_emb = mean_pooling(pos_out, pos_mask)
            neg_emb = mean_pooling(neg_out, neg_mask)

            # Normalize
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            pos_emb = F.normalize(pos_emb, p=2, dim=1)
            neg_emb = F.normalize(neg_emb, p=2, dim=1)

            # Compute loss
            loss = compute_multi_negative_triplet_loss(anchor_emb, pos_emb, neg_emb, neg_counts, margin)

            # Compute metrics (on first negative for simplicity)
            batch_size = anchor_emb.size(0)
            neg_start_idx = 0
            for i in range(batch_size):
                anch = anchor_emb[i:i+1]
                pos = pos_emb[i:i+1]
                # Use first negative for metrics
                neg = neg_emb[neg_start_idx:neg_start_idx+1]
                neg_start_idx += neg_counts[i]

                pos_dist = torch.nn.functional.pairwise_distance(anch, pos).item()
                neg_dist = torch.nn.functional.pairwise_distance(anch, neg).item()

                violation = 1 if (pos_dist + margin > neg_dist) else 0
                separation = neg_dist - pos_dist

                all_metrics['pos_dist'].append(pos_dist)
                all_metrics['neg_dist'].append(neg_dist)
                all_metrics['violations_pct'].append(violation)
                all_metrics['separation'].append(separation)

            all_metrics['loss'].append(loss.item())

    return {
        'loss': np.mean(all_metrics['loss']),
        'pos_dist': np.mean(all_metrics['pos_dist']),
        'neg_dist': np.mean(all_metrics['neg_dist']),
        'violations_pct': np.mean(all_metrics['violations_pct']) * 100,
        'separation': np.mean(all_metrics['separation'])
    }


def train(args, custom_save_dir=None):
    # Device selection: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"  Using device: {device}")

    # Load data
    print(f"\n Loading dataset: {args.data}")
    with open(args.data, 'r') as f:
        examples = [json.loads(line) for line in f]

    if args.max_samples:
        examples = examples[:args.max_samples]

    print(f"   Loaded {len(examples)} examples")

    # Check if dataset has multiple negatives
    sample = examples[0]
    if 'negative_feedbacks' in sample:
        print(f"    Multiple negatives detected: {len(sample['negative_feedbacks'])} per example")
    else:
        print(f"     Single negative format - consider using utils/add_multiple_random_negatives.py")

    # Split
    train_data, val_data = train_test_split(examples, test_size=0.1, random_state=42)
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    # Load model & tokenizer
    print(f"\n Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model, trust_remote_code=True).to(device)

    # Datasets
    train_dataset = MultiNegativeTripletDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = MultiNegativeTripletDataset(val_data, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multi_negative)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multi_negative)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Monitoring
    monitor = TripletTrainingMonitor(window_size=100, margin=args.margin)

    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    best_val_loss = float('inf')
    patience_counter = 0

    # Use custom save directory if provided, otherwise use default
    if custom_save_dir:
        save_dir = Path(custom_save_dir)
    else:
        save_dir = Path('checkpoints') / f"{args.model.split('/')[-1]}_best"

    for epoch in range(args.epochs):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch in pbar:
            anchor_ids = batch['anchor_input_ids'].to(device)
            anchor_mask = batch['anchor_attention_mask'].to(device)
            pos_ids = batch['positive_input_ids'].to(device)
            pos_mask = batch['positive_attention_mask'].to(device)
            neg_ids = batch['negative_input_ids'].to(device)
            neg_mask = batch['negative_attention_mask'].to(device)
            neg_counts = batch['neg_counts']

            # Forward pass
            anchor_out = model(anchor_ids, attention_mask=anchor_mask)
            pos_out = model(pos_ids, attention_mask=pos_mask)
            neg_out = model(neg_ids, attention_mask=neg_mask)

            # Mean pooling
            anchor_emb = mean_pooling(anchor_out, anchor_mask)
            pos_emb = mean_pooling(pos_out, pos_mask)
            neg_emb = mean_pooling(neg_out, neg_mask)

            # Normalize
            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            pos_emb = F.normalize(pos_emb, p=2, dim=1)
            neg_emb = F.normalize(neg_emb, p=2, dim=1)

            # Compute loss
            loss = compute_multi_negative_triplet_loss(anchor_emb, pos_emb, neg_emb, neg_counts, args.margin)

            # Backward
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Validation
        val_metrics = evaluate_multi_negative(model, val_loader, device, args.margin)

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {np.mean(epoch_losses):.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Violations: {val_metrics['violations_pct']:.1f}%")
        print(f"  Val Separation: {val_metrics['separation']:.4f}")

        # Monitor (update with validation metrics)
        monitor.update(val_metrics, step=epoch)

        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # Save best model
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"   Best model saved to {save_dir}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  Early stopping triggered (patience={args.patience})")
                break

    # Save training report
    report_dir = Path('training_logs')
    report_dir.mkdir(parents=True, exist_ok=True)
    monitor.save_report('training_logs/final_report.png')
    print("\n Training complete!")

    # Return final metrics
    return {
        'final_test_loss': best_val_loss,
        'final_metric_value': best_val_loss,
        'model_output_path': str(save_dir),
        'notes': f'Training completed with {epoch+1} epochs'
    }


def run_model_training(config: Dict, dataset_path: str) -> Dict:
    """
    Stage 4: Model Training

    Trains an embedding model using triplet loss with the generated dataset.

    Args:
        config: Configuration dictionary containing training settings
        dataset_path: Path to the JSONL dataset file

    Returns:
        Dictionary containing training metrics and model path
    """
    from argparse import Namespace

    training_config = config.get('training', {})
    hyperparams = training_config.get('hyperparameters', {})

    # Create args namespace from config
    args = Namespace(
        data=dataset_path,
        model=training_config.get('base_embedding_model', 'google/embeddinggemma-300m'),
        batch_size=hyperparams.get('batch_size', 8),
        epochs=hyperparams.get('num_epochs', 10),
        lr=hyperparams.get('learning_rate', 2e-5),
        margin=hyperparams.get('triplet_margin', 0.5),
        max_length=training_config.get('max_sequence_length', 128),
        patience=training_config.get('early_stopping_patience', 5),
        max_samples=training_config.get('max_samples', None)
    )

    # Update save directory to use run_id
    run_id = config.get('run_id', 'UNKNOWN')
    model_output_dir = Path(training_config.get('model_output_dir', './models'))
    save_dir = model_output_dir / run_id

    print(f"\n[Stage 4] Starting model training...")
    print(f"  Model: {args.model}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Output: {save_dir}")

    # Run training with custom save directory
    metrics = train(args, custom_save_dir=str(save_dir))

    # Update model output path with run_id
    if metrics:
        metrics['model_output_path'] = str(save_dir)

    return metrics or {}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to JSONL dataset')
    parser.add_argument('--model', default='google/embeddinggemma-300m', help='Base model')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5, help='Triplet loss margin')
    parser.add_argument('--max-length', type=int, default=128, help='Max sequence length')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit dataset size (for testing)')

    args = parser.parse_args()
    train(args)
