# Stage 3: Model Training and Finetuning

This directory contains modules for training and evaluating embedding models.

## Modules

### `train_embedding.py`
Embedding model training with SentenceTransformers:
- Supports Triplet Loss
- Automatic train/val/test splitting
- Configurable evaluation metrics
- Model checkpointing and versioning

**Main Function**: `run_model_training(config, dataset_path)`

**Supported Loss Functions**:
1. **Triplet Loss**
   - For learning with explicit negatives
   - Learns to separate positive from negative
   - Requires: `anchor`, `positive`, `negative`

**Usage**:
```python
from model_training import run_model_training

config = {
    'training': {
        'mode': 'triplet',
        'base_embedding_model': 'google/embeddinggemma-300m',
        'hyperparameters': {...},
        'model_output_dir': './models'
    }
}

metrics = run_model_training(config, 'data/train.jsonl')
```

### `evaluate_embedding.py`
Model evaluation utilities for comparing different embedding models on code similarity tasks.

### `ranking_eval.py`
Advanced ranking metrics for information retrieval evaluation.

## Training Configuration

### MNRL Mode
```yaml
training:
  mode: "mnrl"
  data_columns:
    sentence1: "code_snippet"
    sentence2: "refined_feedback"
  evaluator_type: "ir"
  metric_for_best_model: "validation-ir-eval_cosine_ndcg@10"
```

### Triplet Mode
```yaml
training:
  mode: "triplet"
  data_columns:
    anchor: "code_snippet"
    positive: "refined_feedback"
    negative: "negative_feedback"
  hyperparameters:
    triplet_margin: 0.5
  evaluator_type: "triplet"
  metric_for_best_model: "validation-set_cosine_accuracy"
```

## Hyperparameters

```yaml
hyperparameters:
  num_epochs: 5           # Training epochs
  batch_size: 16          # Batch size for training
  learning_rate: 0.00032  # Learning rate
  warmup_ratio: 0.1       # Warmup steps ratio
  triplet_margin: 0.5     # Margin for triplet loss
```

## Evaluation Metrics

### Triplet Evaluator
- **Cosine Accuracy**: Percentage where positive is closer than negative
- **Euclidean Accuracy**: Same using Euclidean distance
- **Manhattan Accuracy**: Same using Manhattan distance

### Information Retrieval Evaluator
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MAP@k**: Mean Average Precision
- **Recall@k**: Recall at k
- **Precision@k**: Precision at k
- **MRR@k**: Mean Reciprocal Rank

## Model Output

Trained models are saved to:
```
models/
└── [run_id]/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer_config.json
    └── modules.json
```

## Loading Trained Models

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('./models/Exp-001')

# Generate embeddings
embeddings = model.encode([
    "code snippet",
    "feedback text"
])

# Compute similarity
similarity = model.similarity(embeddings[0], embeddings[1])
```

## Supported Base Models

Pre-trained models compatible with SentenceTransformers:

### Code-Specific Models
- `microsoft/graphcodebert-base`
- `Salesforce/codet5-base`
- `BAAI/bge-code-v1`

### General Embedding Models
- `google/embeddinggemma-300m`
- `sentence-transformers/all-MiniLM-L6-v2`
- `intfloat/multilingual-e5-large`
- `BAAI/bge-large-en-v1.5`

### Multilingual Models
- `intfloat/multilingual-e5-large`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`

Choose based on:
- Model size vs. performance trade-off
- Language requirements
- Domain (code vs. natural language)
- Available compute resources

## GPU/Accelerator Support

Automatically detects and uses:
1. CUDA (NVIDIA GPUs)
2. MPS (Apple Silicon)
3. CPU (fallback)

Set device preference in code:
```python
device = "cuda"  # or "mps", "cpu"
model = SentenceTransformer(model_name, device=device)
```
