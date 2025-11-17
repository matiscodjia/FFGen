#!/usr/bin/env python3
"""
LoRA Fine-tuning Script for Embedding Models with Multiple Negatives

Generic script that works with any embedding model and uses LoRA for efficient fine-tuning.
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

# LoRA imports
from peft import LoraConfig, get_peft_model, TaskType

# Import existing utilities
try:
    from utils.triplet_monitor import TripletTrainingMonitor
except ImportError:
    TripletTrainingMonitor = None
    print("Warning: TripletTrainingMonitor not available")


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
            'negatives': negatives_enc,
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
        'neg_counts': neg_counts
    }


def mean_pooling(model_output, attention_mask):
    """Mean pooling with attention mask"""
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_multi_negative_triplet_loss(anchor_emb, positive_emb, negative_embs, neg_counts, margin=0.5):
    """
    Compute triplet loss with multiple negatives per anchor.
    """
    batch_size = anchor_emb.size(0)
    total_loss = 0.0

    neg_start_idx = 0
    for i in range(batch_size):
        anch = anchor_emb[i:i+1]
        pos = positive_emb[i:i+1]

        num_negs = neg_counts[i]
        negs = negative_embs[neg_start_idx:neg_start_idx + num_negs]
        neg_start_idx += num_negs

        pos_dist = torch.nn.functional.pairwise_distance(anch, pos)
        neg_dists = torch.cdist(anch, negs, p=2).squeeze(0)
        losses = torch.relu(pos_dist + margin - neg_dists)
        example_loss = losses.mean()
        total_loss += example_loss

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

            anchor_out = model(anchor_ids, attention_mask=anchor_mask)
            pos_out = model(pos_ids, attention_mask=pos_mask)
            neg_out = model(neg_ids, attention_mask=neg_mask)

            anchor_emb = mean_pooling(anchor_out, anchor_mask)
            pos_emb = mean_pooling(pos_out, pos_mask)
            neg_emb = mean_pooling(neg_out, neg_mask)

            anchor_emb = F.normalize(anchor_emb, p=2, dim=1)
            pos_emb = F.normalize(pos_emb, p=2, dim=1)
            neg_emb = F.normalize(neg_emb, p=2, dim=1)

            loss = compute_multi_negative_triplet_loss(anchor_emb, pos_emb, neg_emb, neg_counts, margin)

            batch_size = anchor_emb.size(0)
            neg_start_idx = 0
            for i in range(batch_size):
                anch = anchor_emb[i:i+1]
                pos = pos_emb[i:i+1]
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


def setup_lora_model(model, lora_r=8, lora_alpha=16, lora_dropout=0.05, target_modules=None):
    """
    Apply LoRA to any embedding model.
    """
    if target_modules is None:
        # Auto-detect target modules from model
        target_modules = []
        for name, module in model.named_modules():
            # Look for common linear layers in transformers
            if any(key in name for key in ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                            'gate_proj', 'up_proj', 'down_proj',
                                            'dense', 'query', 'key', 'value',
                                            'wi', 'wo', 'wi_0', 'wi_1']):
                # Extract module type without the parent path
                module_type = name.split('.')[-1]
                if module_type not in target_modules:
                    target_modules.append(module_type)

        if not target_modules:
            print("⚠️  Warning: No target modules auto-detected, using default list")
            target_modules = ["q_proj", "v_proj"]

    print(f"🎯 Target modules for LoRA: {target_modules}")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🔧 LoRA Configuration:")
    print(f"  Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Total params: {total_params:,}")
    print(f"  LoRA rank: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")

    return model


def train_lora(args, custom_save_dir=None):
    """Main training function with LoRA"""

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"🖥️  Using device: {device}")

    # Load data
    print(f"\n📂 Loading dataset: {args.data}")
    with open(args.data, 'r') as f:
        examples = [json.loads(line) for line in f]

    if args.max_samples:
        examples = examples[:args.max_samples]

    print(f"   Loaded {len(examples)} examples")

    # Check dataset format
    sample = examples[0]
    if 'negative_feedbacks' in sample:
        num_negs = len(sample['negative_feedbacks'])
        print(f"   ✓ Multiple negatives detected: {num_negs} per example")
    else:
        print(f"   ⚠️  Single negative format detected")

    # Split
    train_data, val_data = train_test_split(examples, test_size=0.1, random_state=42)
    print(f"   Train: {len(train_data)}, Val: {len(val_data)}")

    # Load model & tokenizer
    print(f"\n🤖 Loading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Load with 8-bit quantization if requested
    if args.use_8bit:
        print("   Loading with 8-bit quantization...")
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0
        )
        model = AutoModel.from_pretrained(
            args.model,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto"
        )
        # Prepare model for kbit training
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        model = model.to(device)

    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        print("   Enabling gradient checkpointing...")
        model.gradient_checkpointing_enable()

    # Apply LoRA
    print("\n🔧 Applying LoRA adaptation...")
    model = setup_lora_model(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules
    )

    # Move to device if not already done by device_map
    if not args.use_8bit:
        model = model.to(device)

    # Datasets
    train_dataset = MultiNegativeTripletDataset(train_data, tokenizer, max_length=args.max_length)
    val_dataset = MultiNegativeTripletDataset(val_data, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_multi_negative
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_multi_negative
    )

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Monitoring
    monitor = None
    if TripletTrainingMonitor is not None:
        monitor = TripletTrainingMonitor(window_size=100, margin=args.margin)

    print("\n" + "="*80)
    print("🚀 STARTING LORA TRAINING")
    print("="*80)

    best_val_loss = float('inf')
    patience_counter = 0

    # Output directory
    if custom_save_dir:
        output_dir = Path(custom_save_dir)
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
            loss = compute_multi_negative_triplet_loss(
                anchor_emb, pos_emb, neg_emb, neg_counts, args.margin
            )

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

        print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {np.mean(epoch_losses):.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Val Violations: {val_metrics['violations_pct']:.1f}%")
        print(f"  Val Separation: {val_metrics['separation']:.4f}")

        # Monitor
        if monitor is not None:
            monitor.update(val_metrics, step=epoch)

        # Early stopping & checkpointing
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0

            # Save LoRA adapter
            checkpoint_dir = output_dir / "best_model"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

            print(f" Best LoRA adapter saved to {checkpoint_dir}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⏹  Early stopping triggered (patience={args.patience})")
                break

    # Save training report
    if monitor is not None:
        report_dir = output_dir / 'training_logs'
        report_dir.mkdir(parents=True, exist_ok=True)
        monitor.save_report(str(report_dir / 'final_report.png'))

    print("\nTraining complete!")
    print(f"Best model saved at: {output_dir / 'best_model'}")

    return {
        'final_test_loss': best_val_loss,
        'final_metric_value': best_val_loss,
        'model_output_path': str(output_dir / 'best_model'),
        'notes': f'LoRA training completed with {epoch+1} epochs'
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train embedding model with LoRA')

    # Data
    parser.add_argument('--data', type=str, required=True, help='Path to JSONL dataset')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit dataset size')

    # Model
    parser.add_argument('--model', type=str, default='google/embeddinggemma-300m',
                        help='Base embedding model')
    parser.add_argument('--max-length', type=int, default=512, help='Max sequence length')

    # LoRA
    parser.add_argument('--lora-r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--target-modules', type=str, nargs='+', default=None,
                        help='Target modules for LoRA')

    # Training
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size (reduced for memory)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--margin', type=float, default=0.5, help='Triplet loss margin')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')

    # Memory optimization
    parser.add_argument('--use-8bit', action='store_true',
                        help='Use 8-bit quantization (saves ~50% memory)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Enable gradient checkpointing (saves memory, slower)')

    # Output
    parser.add_argument('--output-dir', type=str, default='./lora_output',
                        help='Output directory for LoRA adapters')

    args = parser.parse_args()
    train_lora(args)
