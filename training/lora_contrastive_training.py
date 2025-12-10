"""
LoRA Contrastive Training for Code-Feedback Alignment

This script implements sophisticated contrastive learning to align code snippets
with their feedback using LoRA fine-tuning and InfoNCE loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

import pandas as pd
import numpy as np
from pathlib import Path
import json
from tqdm.auto import tqdm
from typing import Dict, List, Optional, Tuple

# Transformers & PEFT
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel
)

# Datasets
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Utilities
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# Configuration
# =============================================================================

class Config:
    # Model
    model_name = "Salesforce/SFR-Embedding-Code-400M_R"

    # LoRA config
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.1
    lora_target_modules = ["qkv_proj", "o_proj", "down_proj", "up_gate_proj"]

    # Training
    batch_size = 16
    learning_rate = 2e-4
    weight_decay = 0.01
    num_epochs = 5
    warmup_steps = 500
    gradient_accumulation_steps = 1

    # InfoNCE
    temperature = 0.07

    # Negative sampling
    negative_strategy = "cluster"  # "random" or "cluster"

    # Data
    max_code_length = 512
    max_feedback_length = 256

    # Paths
    dataset_name = "matis35/RAFT"
    clustered_dataset_path = "./data/dataset_clustered_no_tests.csv"
    output_dir = "./checkpoints/lora_contrastive"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")


# =============================================================================
# Set Random Seeds
# =============================================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# =============================================================================
# Dataset
# =============================================================================

class ContrastiveCodeDataset(Dataset):
    """Dataset for contrastive learning with cluster-aware negative sampling."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_code_length: int = 512,
        max_feedback_length: int = 256,
        negative_strategy: str = "random"
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_feedback_length = max_feedback_length
        self.negative_strategy = negative_strategy

        # Build cluster index for cluster-based sampling
        self.cluster_to_indices = {}
        for idx, cluster in enumerate(df['cluster_kmeans']):
            if cluster not in self.cluster_to_indices:
                self.cluster_to_indices[cluster] = []
            self.cluster_to_indices[cluster].append(idx)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        return {
            'code': row['code'],
            'feedback': row['feedback'],
            'cluster': row['cluster_kmeans'],
            'idx': idx
        }

    def get_cluster_batch_indices(self, batch_size: int) -> List[int]:
        """Sample a batch from similar clusters (for hard negatives)."""
        cluster = np.random.choice(list(self.cluster_to_indices.keys()))
        cluster_indices = self.cluster_to_indices[cluster]

        if len(cluster_indices) < batch_size:
            sampled = cluster_indices.copy()
            remaining = batch_size - len(sampled)
            all_indices = list(range(len(self.df)))
            random_indices = np.random.choice(
                [i for i in all_indices if i not in sampled],
                size=remaining,
                replace=False
            )
            sampled.extend(random_indices.tolist())
        else:
            sampled = np.random.choice(cluster_indices, size=batch_size, replace=False).tolist()

        return sampled


# =============================================================================
# Collator
# =============================================================================

class ContrastiveCollator:
    """Collator that tokenizes code and feedback separately for contrastive learning."""

    def __init__(
        self,
        tokenizer,
        max_code_length: int = 512,
        max_feedback_length: int = 256
    ):
        self.tokenizer = tokenizer
        self.max_code_length = max_code_length
        self.max_feedback_length = max_feedback_length

    def __call__(self, batch):
        codes = [item['code'] for item in batch]
        feedbacks = [item['feedback'] for item in batch]
        clusters = torch.tensor([item['cluster'] for item in batch])
        indices = torch.tensor([item['idx'] for item in batch])

        code_encodings = self.tokenizer(
            codes,
            padding=True,
            truncation=True,
            max_length=self.max_code_length,
            return_tensors='pt'
        )

        feedback_encodings = self.tokenizer(
            feedbacks,
            padding=True,
            truncation=True,
            max_length=self.max_feedback_length,
            return_tensors='pt'
        )

        return {
            'code_input_ids': code_encodings['input_ids'],
            'code_attention_mask': code_encodings['attention_mask'],
            'feedback_input_ids': feedback_encodings['input_ids'],
            'feedback_attention_mask': feedback_encodings['attention_mask'],
            'clusters': clusters,
            'indices': indices
        }


# =============================================================================
# Cluster Batch Sampler
# =============================================================================

class ClusterBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        num_batches = len(self.dataset) // self.batch_size
        for _ in range(num_batches):
            yield self.dataset.get_cluster_batch_indices(self.batch_size)

    def __len__(self):
        return len(self.dataset) // self.batch_size


# =============================================================================
# Model
# =============================================================================

class ContrastiveCodeModel(nn.Module):
    """Dual encoder model for code-feedback alignment."""

    def __init__(
        self,
        base_model_name: str,
        lora_config: LoraConfig,
        temperature: float = 0.07
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)
        self.encoder = get_peft_model(self.encoder, lora_config)
        self.temperature = temperature

        print(f"\nModel architecture:")
        self.encoder.print_trainable_parameters()

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask."""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        """Encode input to dense vector."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(
        self,
        code_input_ids,
        code_attention_mask,
        feedback_input_ids,
        feedback_attention_mask
    ):
        code_embeddings = self.encode(code_input_ids, code_attention_mask)
        feedback_embeddings = self.encode(feedback_input_ids, feedback_attention_mask)

        return code_embeddings, feedback_embeddings


# =============================================================================
# Loss
# =============================================================================

class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive learning with in-batch negatives."""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        code_embeddings: torch.Tensor,
        feedback_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size = code_embeddings.shape[0]

        similarity = torch.matmul(code_embeddings, feedback_embeddings.T) / self.temperature
        labels = torch.arange(batch_size, device=similarity.device)

        loss_code_to_feedback = F.cross_entropy(similarity, labels)
        loss_feedback_to_code = F.cross_entropy(similarity.T, labels)
        loss = (loss_code_to_feedback + loss_feedback_to_code) / 2

        with torch.no_grad():
            positive_sim = torch.diagonal(similarity).mean()
            mask = torch.eye(batch_size, device=similarity.device).bool()
            negative_sim = similarity.masked_select(~mask).mean()
            margin = positive_sim - negative_sim
            preds = similarity.argmax(dim=1)
            accuracy = (preds == labels).float().mean()

        metrics = {
            'positive_sim': positive_sim.item(),
            'negative_sim': negative_sim.item(),
            'margin': margin.item(),
            'accuracy': accuracy.item()
        }

        return loss, metrics


# =============================================================================
# Trainer
# =============================================================================

class ContrastiveTrainer:
    """Custom trainer for contrastive learning."""

    def __init__(
        self,
        model: ContrastiveCodeModel,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        loss_fn: InfoNCELoss,
        device: torch.device,
        output_dir: str,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps

        self.global_step = 0
        self.best_val_loss = float('inf')
        self.history = []

    def train_epoch(self, epoch: int):
        """Train for one epoch."""
        self.model.train()

        epoch_loss = 0
        epoch_metrics = {
            'positive_sim': 0,
            'negative_sim': 0,
            'margin': 0,
            'accuracy': 0
        }

        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")

        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            code_embeddings, feedback_embeddings = self.model(
                code_input_ids=batch['code_input_ids'],
                code_attention_mask=batch['code_attention_mask'],
                feedback_input_ids=batch['feedback_input_ids'],
                feedback_attention_mask=batch['feedback_attention_mask']
            )

            loss, metrics = self.loss_fn(code_embeddings, feedback_embeddings)
            loss = loss / self.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            epoch_loss += loss.item() * self.gradient_accumulation_steps
            for k, v in metrics.items():
                epoch_metrics[k] += v

            progress_bar.set_postfix({
                'loss': loss.item() * self.gradient_accumulation_steps,
                'margin': metrics['margin'],
                'acc': metrics['accuracy']
            })

        num_batches = len(self.train_dataloader)
        epoch_loss /= num_batches
        for k in epoch_metrics:
            epoch_metrics[k] /= num_batches

        return epoch_loss, epoch_metrics

    @torch.no_grad()
    def validate(self):
        """Validate on validation set."""
        self.model.eval()

        val_loss = 0
        val_metrics = {
            'positive_sim': 0,
            'negative_sim': 0,
            'margin': 0,
            'accuracy': 0
        }

        for batch in tqdm(self.val_dataloader, desc="Validation"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            code_embeddings, feedback_embeddings = self.model(
                code_input_ids=batch['code_input_ids'],
                code_attention_mask=batch['code_attention_mask'],
                feedback_input_ids=batch['feedback_input_ids'],
                feedback_attention_mask=batch['feedback_attention_mask']
            )

            loss, metrics = self.loss_fn(code_embeddings, feedback_embeddings)

            val_loss += loss.item()
            for k, v in metrics.items():
                val_metrics[k] += v

        num_batches = len(self.val_dataloader)
        val_loss /= num_batches
        for k in val_metrics:
            val_metrics[k] /= num_batches

        return val_loss, val_metrics

    def train(self, num_epochs: int):
        """Full training loop."""
        print("="*100)
        print("STARTING TRAINING")
        print("="*100)
        print()

        for epoch in range(1, num_epochs + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            val_loss, val_metrics = self.validate()

            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train Margin: {train_metrics['margin']:.4f} | Val Margin: {val_metrics['margin']:.4f}")
            print(f"  Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Train Pos/Neg: {train_metrics['positive_sim']:.4f}/{train_metrics['negative_sim']:.4f}")
            print(f"  Val Pos/Neg: {val_metrics['positive_sim']:.4f}/{val_metrics['negative_sim']:.4f}")

            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            })

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(f"best_model")
                print(f"  ✓ New best model saved (val_loss: {val_loss:.4f})")

            self.save_checkpoint(f"checkpoint_epoch_{epoch}")
            print()

        print("="*100)
        print("TRAINING COMPLETE")
        print("="*100)

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = self.output_dir / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.encoder.save_pretrained(checkpoint_dir)

        torch.save({
            'global_step': self.global_step,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }, checkpoint_dir / 'trainer_state.pt')


# =============================================================================
# Main Training Script
# =============================================================================

def main():
    config = Config()

    print("="*100)
    print("LORA CONTRASTIVE TRAINING")
    print("="*100)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {config.device}")
    print()

    # -------------------------------------------------------------------------
    # Print Configuration
    # -------------------------------------------------------------------------
    print("="*100)
    print("CONFIGURATION")
    print("="*100)
    print(f"Model: {config.model_name}")
    print(f"LoRA r={config.lora_r}, alpha={config.lora_alpha}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Temperature: {config.temperature}")
    print(f"Negative strategy: {config.negative_strategy}")
    print(f"Dataset: {config.dataset_name}")
    print(f"Device: {config.device}")
    print()

    # -------------------------------------------------------------------------
    # Load Data
    # -------------------------------------------------------------------------
    print("="*100)
    print("LOADING DATA")
    print("="*100)
    print()

    print("Loading dataset from Hugging Face...")
    hf_dataset = load_dataset(config.dataset_name)
    print(f"  Available splits: {list(hf_dataset.keys())}")

    all_dfs = []
    for split_name, split_data in hf_dataset.items():
        split_df = split_data.to_pandas()
        print(f"  - {split_name}: {len(split_df):,} samples")
        all_dfs.append(split_df)

    df_feedback = pd.concat(all_dfs, ignore_index=True)
    print(f"✓ Combined all splits: {len(df_feedback):,} total samples")

    df_clustered = pd.read_csv(config.clustered_dataset_path)
    print(f"✓ Loaded {len(df_clustered):,} samples with clustering info")

    df_clustered = df_clustered.rename(columns={'code_snippet': 'code'})

    print(f"\nMerging datasets...")
    df = df_feedback.merge(
        df_clustered[['code', 'cluster_kmeans']],
        on='code',
        how='inner'
    )

    print(f"✓ Merged dataset: {len(df):,} samples")

    missing_clusters = df['cluster_kmeans'].isna().sum()
    if missing_clusters > 0:
        print(f"Warning: {missing_clusters} samples without cluster info (filling with -1)")
        df['cluster_kmeans'] = df['cluster_kmeans'].fillna(-1).astype(int)
    else:
        df['cluster_kmeans'] = df['cluster_kmeans'].astype(int)

    print(f"\nCluster distribution:")
    print(df['cluster_kmeans'].value_counts().sort_index())

    cluster_counts = df['cluster_kmeans'].value_counts()
    min_cluster_size = cluster_counts.min()
    print(f"\nSmallest cluster has {min_cluster_size} samples")

    if min_cluster_size >= 3:
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['cluster_kmeans'])
        train_df, val_df = train_test_split(train_df, test_size=0.111, random_state=42, stratify=train_df['cluster_kmeans'])
        print("✓ Using stratified splits")
    else:
        train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.111, random_state=42)
        print("✓ Using random splits")

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")
    print()

    # -------------------------------------------------------------------------
    # Initialize Model and Tokenizer
    # -------------------------------------------------------------------------
    print("="*100)
    print("INITIALIZING MODEL")
    print("="*100)
    print()

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✓ Tokenizer loaded: {config.model_name}")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=config.lora_target_modules,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION
    )
    print(f"✓ LoRA config created")

    model = ContrastiveCodeModel(
        base_model_name=config.model_name,
        lora_config=lora_config,
        temperature=config.temperature,
    ).to(config.device)
    print(f"✓ Model created and moved to {config.device}")
    print()

    # -------------------------------------------------------------------------
    # Create Dataloaders
    # -------------------------------------------------------------------------
    print("="*100)
    print("CREATING DATALOADERS")
    print("="*100)
    print()

    train_dataset = ContrastiveCodeDataset(
        df=train_df,
        tokenizer=tokenizer,
        max_code_length=config.max_code_length,
        max_feedback_length=config.max_feedback_length,
        negative_strategy=config.negative_strategy
    )

    val_dataset = ContrastiveCodeDataset(
        df=val_df,
        tokenizer=tokenizer,
        max_code_length=config.max_code_length,
        max_feedback_length=config.max_feedback_length,
        negative_strategy="random"
    )

    print(f"✓ Train dataset: {len(train_dataset):,} samples")
    print(f"✓ Val dataset: {len(val_dataset):,} samples")

    collator = ContrastiveCollator(
        tokenizer=tokenizer,
        max_code_length=config.max_code_length,
        max_feedback_length=config.max_feedback_length
    )

    if config.negative_strategy == "cluster":
        train_sampler = ClusterBatchSampler(train_dataset, config.batch_size)
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=collator,
            num_workers=0
        )
    else:
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0
        )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    print(f"✓ Train dataloader: {len(train_dataloader)} batches")
    print(f"✓ Val dataloader: {len(val_dataloader)} batches")
    print(f"\nNegative sampling strategy: {config.negative_strategy}")
    print()

    # -------------------------------------------------------------------------
    # Setup Training
    # -------------------------------------------------------------------------
    print("="*100)
    print("SETUP TRAINING")
    print("="*100)
    print()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    print(f"✓ Optimizer: AdamW (lr={config.learning_rate})")

    num_training_steps = len(train_dataloader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=num_training_steps
    )
    print(f"✓ Scheduler: Linear warmup ({config.warmup_steps} steps) + decay")

    loss_fn = InfoNCELoss(temperature=config.temperature)
    print(f"✓ Loss: InfoNCE (temperature={config.temperature})")

    trainer = ContrastiveTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=config.device,
        output_dir=config.output_dir,
        gradient_accumulation_steps=config.gradient_accumulation_steps
    )
    print(f"✓ Trainer initialized")
    print(f"\nTotal training steps: {num_training_steps:,}")
    print()

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------
    trainer.train(num_epochs=config.num_epochs)

    # -------------------------------------------------------------------------
    # Evaluate on Test Set
    # -------------------------------------------------------------------------
    print("="*100)
    print("EVALUATING ON TEST SET")
    print("="*100)
    print()

    best_model_path = Path(config.output_dir) / "best_model"
    model.encoder = PeftModel.from_pretrained(model.encoder.base_model, best_model_path)
    print(f"✓ Loaded best model from: {best_model_path}")

    test_dataset = ContrastiveCodeDataset(
        df=test_df,
        tokenizer=tokenizer,
        max_code_length=config.max_code_length,
        max_feedback_length=config.max_feedback_length,
        negative_strategy="random"
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )

    print(f"✓ Test dataset: {len(test_dataset):,} samples")

    model.eval()
    test_loss = 0
    test_metrics = {
        'positive_sim': 0,
        'negative_sim': 0,
        'margin': 0,
        'accuracy': 0
    }

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Testing"):
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            code_embeddings, feedback_embeddings = model(
                code_input_ids=batch['code_input_ids'],
                code_attention_mask=batch['code_attention_mask'],
                feedback_input_ids=batch['feedback_input_ids'],
                feedback_attention_mask=batch['feedback_attention_mask']
            )

            loss, metrics = loss_fn(code_embeddings, feedback_embeddings)

            test_loss += loss.item()
            for k, v in metrics.items():
                test_metrics[k] += v

    num_batches = len(test_dataloader)
    test_loss /= num_batches
    for k in test_metrics:
        test_metrics[k] /= num_batches

    print("\n" + "="*100)
    print("TEST RESULTS")
    print("="*100)
    print(f"Loss: {test_loss:.4f}")
    print(f"Margin (Pos - Neg): {test_metrics['margin']:.4f}")
    print(f"Accuracy (Top-1): {test_metrics['accuracy']:.4f}")
    print(f"Positive Similarity: {test_metrics['positive_sim']:.4f}")
    print(f"Negative Similarity: {test_metrics['negative_sim']:.4f}")
    print("="*100)
    print()

    # -------------------------------------------------------------------------
    # Save Training Report
    # -------------------------------------------------------------------------
    history = trainer.history
    report = f"""# LoRA Contrastive Training Report

## Configuration

- **Model**: {config.model_name}
- **LoRA**: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}
- **Loss**: InfoNCE (temperature={config.temperature})
- **Batch size**: {config.batch_size}
- **Learning rate**: {config.learning_rate}
- **Epochs**: {config.num_epochs}
- **Negative sampling**: {config.negative_strategy}
- **Device**: {config.device}

## Dataset

- Train: {len(train_df):,} samples
- Validation: {len(val_df):,} samples
- Test: {len(test_df):,} samples

## Training Results

### Best Validation Performance

- **Best Epoch**: {history[np.argmin([h['val_loss'] for h in history])]['epoch']}
- **Best Val Loss**: {trainer.best_val_loss:.4f}
- **Best Val Margin**: {history[np.argmin([h['val_loss'] for h in history])]['val_metrics']['margin']:.4f}
- **Best Val Accuracy**: {history[np.argmin([h['val_loss'] for h in history])]['val_metrics']['accuracy']:.4f}

### Test Performance

- **Test Loss**: {test_loss:.4f}
- **Test Margin**: {test_metrics['margin']:.4f}
- **Test Accuracy**: {test_metrics['accuracy']:.4f}
- **Positive Similarity**: {test_metrics['positive_sim']:.4f}
- **Negative Similarity**: {test_metrics['negative_sim']:.4f}

## Model Checkpoint

Best model saved to: `{Path(config.output_dir) / 'best_model'}`

## Usage

```python
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load base model and tokenizer
base_model = AutoModel.from_pretrained("{config.model_name}")
tokenizer = AutoTokenizer.from_pretrained("{config.model_name}")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "{Path(config.output_dir) / 'best_model'}")

# Encode code
code = "int factorial(int n) {{ return n == 0 ? 1 : n * factorial(n-1); }}"
inputs = tokenizer(code, return_tensors="pt")
outputs = model(**inputs)
embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
```
"""

    report_path = Path(config.output_dir) / 'training_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"✓ Training report saved to: {report_path}")
    print()
    print(report)


if __name__ == "__main__":
    main()
