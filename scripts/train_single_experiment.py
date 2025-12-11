"""
Single Experiment Training Script
==================================

This script is called by the pipeline to train a single LoRA adapter.
It handles:
- Loading the configuration
- Initializing the model with LoRA
- Training with NCE loss
- Tracking comprehensive metrics
- Pushing to HuggingFace Hub
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi


# =============================================================================
# Dataset & Collator
# =============================================================================

class CodeFeedbackDataset(Dataset):
    """Dataset for code-feedback pairs"""

    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return {
            "code": item.get("code", ""),
            "feedback": item.get("feedback", "")
        }


class CodeFeedbackCollator:
    """Collator for batching code-feedback pairs"""

    def __init__(self, tokenizer, code_max_length=512, feedback_max_length=256):
        self.tokenizer = tokenizer
        self.code_max_length = code_max_length
        self.feedback_max_length = feedback_max_length

    def __call__(self, batch):
        codes = [item["code"] for item in batch]
        feedbacks = [item["feedback"] for item in batch]

        token_args = {
            "padding": True,
            "truncation": True,
            "return_tensors": "pt"
        }

        codes_outputs = self.tokenizer(
            codes,
            max_length=self.code_max_length,
            **token_args
        )
        feedbacks_outputs = self.tokenizer(
            feedbacks,
            max_length=self.feedback_max_length,
            **token_args
        )

        # Match original nce_trainer.py naming convention
        dummy_labels = torch.arange(len(codes))

        return {
            "code_input_id": codes_outputs["input_ids"],
            "code_attention_mask": codes_outputs["attention_mask"],
            "feedback_input_id": feedbacks_outputs["input_ids"],
            "feedback_attention_mask": feedbacks_outputs["attention_mask"],
            "labels": dummy_labels
        }


# =============================================================================
# Bi-Encoder Model with NCE Loss
# =============================================================================

class BiEncoderNCE(nn.Module):
    """Bi-encoder with Noise Contrastive Estimation loss - matches nce_trainer.py"""

    def __init__(self, model_name: str, lora_config: LoraConfig, temperature: float = 0.07):
        super().__init__()
        self.base_model_name = model_name
        self.temperature = temperature

        # Load base model
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Apply LoRA
        self.encoder = get_peft_model(self.encoder, lora_config)

        # Print trainable parameters
        print(f"\nModel architecture:")
        self.encoder.print_trainable_parameters()

    def enable_input_require_grads(self):
        """Enable gradients on input embeddings (crucial for LoRA + gradient checkpointing)"""
        self.encoder.enable_input_require_grads()

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for memory efficiency"""
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def mean_pooling(self, token_embeddings, attention_mask):
        """Mean pooling with attention mask"""
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        """Encode inputs to embeddings"""
        outputs = self.encoder(input_ids, attention_mask)
        token_embeddings = outputs.last_hidden_state
        embeddings = self.mean_pooling(token_embeddings, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def forward(self, code_input_id, code_attention_mask,
                feedback_input_id, feedback_attention_mask, labels=None):
        """Forward pass - returns embeddings (loss computed in trainer)"""
        code_embeddings = self.encode(code_input_id, code_attention_mask)
        feedback_embeddings = self.encode(feedback_input_id, feedback_attention_mask)

        return code_embeddings, feedback_embeddings


# =============================================================================
# Custom Trainer with Metrics Tracking
# =============================================================================

class ContrastiveTrainer(Trainer):
    """Custom trainer for bi-encoder NCE training - matches nce_trainer.py"""

    def __init__(self, *args, metrics_logger=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics_logger = metrics_logger

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss with NCE (matches original implementation)"""
        # 1. Forward Pass - get embeddings
        code_emb, feedback_emb = model(**inputs)

        # 2. Compute similarity matrix and labels
        similarity_matrix = torch.matmul(code_emb, feedback_emb.T) / model.temperature
        batch_size = code_emb.size(0)
        labels = torch.arange(batch_size).to(code_emb.device)

        # 3. Compute NCE loss (symmetric)
        loss_c2f = F.cross_entropy(similarity_matrix, labels)
        loss_f2c = F.cross_entropy(similarity_matrix.T, labels)
        loss = (loss_c2f + loss_f2c) / 2

        # 4. Compute accuracy for logging
        with torch.no_grad():
            pred_code = similarity_matrix.argmax(dim=1)
            accuracy = (pred_code == labels).float().mean()

        # 5. Log metrics
        if self.metrics_logger:
            self.metrics_logger.log_step({
                "loss": loss.item(),
                "accuracy": accuracy.item(),
                "step": self.state.global_step
            })

        if return_outputs:
            # Concatenate embeddings for compute_metrics
            outputs = torch.cat((code_emb, feedback_emb), dim=1)
            return (loss, outputs)

        return loss


# =============================================================================
# Compute Metrics (for evaluation)
# =============================================================================

def compute_metrics(eval_pred):
    """Compute retrieval metrics: MRR, Recall@1, Recall@5, Recall@10"""
    import numpy as np

    predictions = eval_pred.predictions

    # Safety: if it's a tuple, take first element
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # predictions shape: (N_samples, 2 * Embedding_Dim)
    # Split in half
    mid_point = predictions.shape[1] // 2

    code_emb = predictions[:, :mid_point]
    feedback_emb = predictions[:, mid_point:]

    # 1. Compute similarity matrix
    similarity_matrix = np.matmul(code_emb, feedback_emb.T)

    # 2. Labels (diagonal)
    labels = np.arange(len(code_emb))

    # 3. Compute ranks
    sorted_indices = np.argsort(-similarity_matrix, axis=1)
    hits = (sorted_indices == labels[:, None])
    ranks = np.argwhere(hits)[:, 1] + 1

    return {
        "mrr": float(np.mean(1 / ranks)),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10))
    }


class MetricsLogger:
    """Logger for tracking training metrics"""

    def __init__(self, output_dir: str, experiment_id: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.experiment_id = experiment_id
        self.metrics_file = self.output_dir / "metrics.jsonl"
        self.summary_file = self.output_dir / "summary.json"

        self.step_metrics = []
        self.epoch_metrics = []

        # Initialize summary
        self.summary = {
            "experiment_id": experiment_id,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_steps": 0,
            "total_epochs": 0,
            "best_loss": float('inf'),
            "best_accuracy": 0.0,
            "best_mrr": 0.0,
            "best_recall_at_10": 0.0,
            "final_loss": None,
            "final_accuracy": None,
            "final_mrr": None,
            "final_recall_at_10": None
        }

    def log_step(self, metrics: Dict[str, Any]):
        """Log metrics for a single step"""
        self.step_metrics.append(metrics)

        # Update summary
        if "loss" in metrics and metrics["loss"] < self.summary["best_loss"]:
            self.summary["best_loss"] = metrics["loss"]
        if "accuracy" in metrics and metrics["accuracy"] > self.summary["best_accuracy"]:
            self.summary["best_accuracy"] = metrics["accuracy"]
        if "mrr" in metrics and metrics["mrr"] > self.summary["best_mrr"]:
            self.summary["best_mrr"] = metrics["mrr"]
        if "recall_at_10" in metrics and metrics["recall_at_10"] > self.summary["best_recall_at_10"]:
            self.summary["best_recall_at_10"] = metrics["recall_at_10"]

        # Write to file
        with open(self.metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + "\n")

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics for an epoch"""
        epoch_data = {"epoch": epoch, **metrics}
        self.epoch_metrics.append(epoch_data)

        print(f"Epoch {epoch}: Loss={metrics.get('loss', 'N/A'):.4f}, "
              f"Accuracy={metrics.get('accuracy', 'N/A'):.4f}")

    def finalize(self):
        """Finalize and save summary"""
        self.summary["end_time"] = datetime.now().isoformat()
        self.summary["total_steps"] = len(self.step_metrics)
        self.summary["total_epochs"] = len(self.epoch_metrics)

        if self.step_metrics:
            last_metrics = self.step_metrics[-1]
            self.summary["final_loss"] = last_metrics.get("loss")
            self.summary["final_accuracy"] = last_metrics.get("accuracy")
            self.summary["final_mrr"] = last_metrics.get("mrr")
            self.summary["final_recall_at_10"] = last_metrics.get("recall_at_10")

        with open(self.summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2)

        print(f"\nMetrics saved to {self.output_dir}")
        print(f"  Best Loss: {self.summary['best_loss']:.4f}")
        print(f"  Best Accuracy: {self.summary['best_accuracy']:.4f}")
        print(f"  Best MRR: {self.summary['best_mrr']:.4f}")
        print(f"  Best Recall@10: {self.summary['best_recall_at_10']:.4f}")


# =============================================================================
# Main Training Function
# =============================================================================

def train_experiment(config: Dict[str, Any], output_dir: str, log_file: str = None):
    """
    Train a single experiment

    Args:
        config: Experiment configuration dictionary
        output_dir: Directory to save outputs
        log_file: Optional log file path
    """

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file) if log_file else logging.StreamHandler(),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    logger.info(f"Starting experiment: {config['experiment_id']}")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load dataset
    logger.info(f"Loading dataset from: {config['dataset_path']}")

    dataset_path = config['dataset_path']

    # Check if it's a Hugging Face Hub dataset (format: username/dataset-name)
    if '/' in dataset_path and not dataset_path.startswith('.') and not dataset_path.startswith('/'):
        # Load from Hugging Face Hub
        logger.info(f"Loading dataset from Hugging Face Hub: {dataset_path}")
        dataset_dict = load_dataset(dataset_path)

        # Handle DatasetDict - get train, validation, and test splits if available
        if isinstance(dataset_dict, dict):
            train_data = dataset_dict.get('train')
            val_data = dataset_dict.get('validation') or dataset_dict.get('val')
            test_data = dataset_dict.get('test')

            if train_data is None:
                raise ValueError(f"No 'train' split found in dataset {dataset_path}. Available splits: {list(dataset_dict.keys())}")
        else:
            # Single dataset, use as train
            train_data = dataset_dict
            val_data = None
            test_data = None

    elif dataset_path.endswith('.jsonl'):
        # Single file - load as train only
        dataset_dict = load_dataset('json', data_files=dataset_path)
        train_data = dataset_dict['train']
        val_data = None
        test_data = None
    else:
        # Directory with splits
        try:
            dataset_dict = load_dataset('json', data_dir=dataset_path)
            train_data = dataset_dict.get('train')
            val_data = dataset_dict.get('validation')
            test_data = dataset_dict.get('test')
        except:
            # Fallback: try loading individual files
            train_file = Path(dataset_path) / "train.jsonl"
            val_file = Path(dataset_path) / "validation.jsonl"
            test_file = Path(dataset_path) / "test.jsonl"

            if train_file.exists():
                train_data = load_dataset('json', data_files=str(train_file), split='train')
            else:
                raise FileNotFoundError(f"No training data found at {dataset_path}")

            if val_file.exists():
                val_data = load_dataset('json', data_files=str(val_file), split='train')
            else:
                logger.warning("No validation data found")
                val_data = None

            if test_file.exists():
                test_data = load_dataset('json', data_files=str(test_file), split='train')
            else:
                logger.warning("No test data found")
                test_data = None

    logger.info(f"Train dataset loaded: {len(train_data)} examples")
    if val_data:
        logger.info(f"Validation dataset loaded: {len(val_data)} examples")
    if test_data:
        logger.info(f"Test dataset loaded: {len(test_data)} examples")

    # Convert to list format
    train_list = [
        {"code": item["code"], "feedback": item["feedback"]}
        for item in train_data
    ]

    # Create datasets
    train_dataset = CodeFeedbackDataset(train_list)

    val_dataset = None
    if val_data:
        val_list = [
            {"code": item["code"], "feedback": item["feedback"]}
            for item in val_data
        ]
        val_dataset = CodeFeedbackDataset(val_list)

    test_dataset = None
    if test_data:
        test_list = [
            {"code": item["code"], "feedback": item["feedback"]}
            for item in test_data
        ]
        test_dataset = CodeFeedbackDataset(test_list)

    # Load tokenizer
    logger.info(f"Loading tokenizer for: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['base_model'],
        trust_remote_code=True
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create collator
    collator = CodeFeedbackCollator(
        tokenizer,
        code_max_length=config['code_max_length'],
        feedback_max_length=config['feedback_max_length']
    )

    # Setup LoRA config
    lora_config = LoraConfig(
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config['lora_dropout'],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]  # Common targets
    )

    # Initialize model
    logger.info("Initializing bi-encoder model with LoRA")
    model = BiEncoderNCE(
        model_name=config['base_model'],
        lora_config=lora_config,
        temperature=config['temperature']
    )
    model.enable_input_require_grads()
    model.to(device)

    # Initialize metrics logger
    metrics_logger = MetricsLogger(
        output_dir=output_dir,
        experiment_id=config['experiment_id']
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        # Logging
        logging_strategy="steps",
        logging_steps=10,
        # Evaluation
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=50 if val_dataset else None,
        # Saving
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_mrr" if val_dataset else None,
        greater_is_better=True,
        # Performance
        dataloader_num_workers=4,
        dataloader_drop_last=True,  # Important for NCE loss
        remove_unused_columns=False,
        # Reporting
        report_to=["tensorboard"],
        logging_dir=f"{output_dir}/logs",
    )

    # Initialize trainer
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics if val_dataset else None,
        metrics_logger=metrics_logger
    )

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()

    # Finalize metrics
    metrics_logger.finalize()

    # Evaluate on test set if available
    test_metrics = None
    if test_dataset:
        logger.info("="*80)
        logger.info("EVALUATING ON TEST SET")
        logger.info("="*80)

        test_output = trainer.predict(test_dataset)
        test_metrics = test_output.metrics

        logger.info("TEST SET RESULTS:")
        logger.info(f"  MRR:        {test_metrics.get('test_mrr', 0):.4f}")
        logger.info(f"  Recall@1:   {test_metrics.get('test_recall_at_1', 0):.4f}")
        logger.info(f"  Recall@5:   {test_metrics.get('test_recall_at_5', 0):.4f}")
        logger.info(f"  Recall@10:  {test_metrics.get('test_recall_at_10', 0):.4f}")
        logger.info(f"  Loss:       {test_metrics.get('test_loss', 0):.4f}")
        logger.info("="*80)

        # Save test metrics to file
        test_metrics_file = Path(output_dir) / "test_metrics.json"
        with open(test_metrics_file, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        logger.info(f"Test metrics saved to {test_metrics_file}")

    # Save final model
    logger.info(f"Saving model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save training state and metrics
    logger.info("Saving training state and metrics")
    train_state_file = Path(output_dir) / "training_state.json"
    with open(train_state_file, 'w') as f:
        json.dump({
            "config": config,
            "train_result": {
                "train_runtime": train_result.metrics.get("train_runtime"),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
                "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
                "total_flos": train_result.metrics.get("total_flos"),
                "train_loss": train_result.metrics.get("train_loss"),
            },
            "test_metrics": test_metrics if test_metrics else None
        }, f, indent=2)

    # Push to Hub
    logger.info(f"Pushing to Hub: {config['hub_model_id']}")
    try:
        # Push the LoRA adapter (model.encoder is the PEFT model)
        model.encoder.push_to_hub(
            config['hub_model_id'],
            use_auth_token=True,
            commit_message=f"Training completed: {config['experiment_id']}"
        )
        tokenizer.push_to_hub(
            config['hub_model_id'],
            use_auth_token=True
        )
        logger.info("Successfully pushed to Hub")
    except Exception as e:
        logger.error(f"Failed to push to Hub: {e}")
        logger.warning("Continuing despite push failure...")

    logger.info("Training completed successfully!")

    return train_result


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Train single experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--log-file", type=str, help="Log file path")

    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Run training
    train_experiment(config, args.output_dir, args.log_file)


if __name__ == "__main__":
    main()
