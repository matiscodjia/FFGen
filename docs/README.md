# FFGen - Focused Feedback Generation

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Installation](#installation)
4. [Data Engineering Pipeline](#data-engineering-pipeline)
5. [Model Training](#model-training)
6. [Deployment](#deployment)
7. [Experiments and Analysis](#experiments-and-analysis)
8. [Reproducibility](#reproducibility)
9. [Project Structure](#project-structure)
10. [Citation](#citation)

---

## Overview

FFGen (Focused Feedback Generation) is a research project addressing the bottleneck of pedagogical feedback in large-scale programming education. Instead of using expensive generative models (like GPT-4), FFGen implements a neural retrieval system based on fine-tuned embedding models.

**Core Innovation**: A data-centric approach that prioritizes high-quality synthetic data generation and specialized embedding models over large generative models.

**Key Results**:
- Recall@10: 74.0% (vs. 7.2% for zero-shot baselines)
- Latency: < 50ms per query
- Cost: $0 per inference (vs. $0.02 for GPT-4)
- Model Size: 300M parameters (50x smaller than GPT-4)

**Research Context**: 5-month internship project conducted at IONIS Education Group (CEPIA Lab) in collaboration with LIP6 (Sorbonne Université).

---

## Architecture

FFGen uses a Bi-Encoder architecture with contrastive learning:

```
┌─────────────┐                    ┌──────────────┐
│  Buggy Code │──── Encoder 1 ───▶│  Vector (768)│
└─────────────┘                    └──────────────┘
                                           │
                                           │ Cosine Similarity
                                           │
┌─────────────┐                    ┌──────────────┐
│  Feedback   │──── Encoder 2 ───▶│  Vector (768)│
└─────────────┘                    └──────────────┘
```

**Training**: LoRA (Low-Rank Adaptation) on Google's Gemma-300M
**Loss Function**: InfoNCE (contrastive loss with temperature=0.07)
**Vector Database**: ChromaDB with HNSW index for fast retrieval

---

## Installation

### Requirements

- Python >= 3.10
- CUDA-compatible GPU (for training) or CPU (for inference)
- 8GB RAM minimum
- uv package manager (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/matis35/ffgen.git
cd ffgen

# Install dependencies with uv
uv pip install -e .

# Alternative: pip
pip install -e .

# Set environment variables
export DEEPSEEK_API_KEY="your-api-key"  # For dataset generation only
export HF_TOKEN="your-huggingface-token"  # For model downloads
```

### Quick Test

```bash
# Verify installation
python3 -c "from src.model.training import nce_trainer; print('Installation successful!')"
```

---

## Data Engineering Pipeline

The project explores three dataset generation approaches (Chapter 2 of the paper):

### Approach 1: Local Generation (Llama-3B)
**Status**: Deprecated (insufficient quality)
**Issues**: Verbosity, variable name references, generic feedback

### Approach 2: Mistral Large API
**Status**: Legacy (RAFT dataset)
**Dataset**: 11,806 code-feedback pairs from real student submissions
**Limitations**: Lack of pedagogical context (test names, exercise instructions)

### Approach 3: Synthetic Generation (DeepSeek-695B)
**Status**: Current (SYNT dataset)
**Dataset**: 7,023 high-quality synthetic pairs
**Advantages**: Full context, controlled diversity, semantic-only bugs

### Dataset Generation

```bash
# Generate synthetic dataset with validation
cd src/data_engineering/generation
python3 generate_dataset_with_metrics.py

# Monitor generation in real-time (separate terminal)
python3 monitor_generation.py

# Analyze metrics after completion
python3 analyze_metrics.py
```

**Output Files**:
- `dataset_c_piscine_semantic_validated.jsonl` - Final dataset
- `generation_metrics.jsonl` - Per-sample metrics (tokens, retries, timing)
- `generation_summary.json` - Global statistics

**Key Parameters**:
- `TOTAL_SAMPLES_TARGET`: 15,000 samples
- `MAX_RETRIES`: 2 (budget constraint)
- `MAX_WORKERS`: 5 (parallel generation)
- Validation: GCC compilation check with `-Wall -Wextra`

### Quality Analysis

```bash
cd src/data_engineering/analysis

# Compute quality metrics (diversity, redundancy, complexity)
python3 dataset_quality_analysis.py --datasets matis35/SYNT_DATASET

# Compare multiple dataset versions
python3 compare_datasets.py
```

**Metrics Computed**:
- Semantic diversity (embedding-based entropy)
- Redundancy (Jaccard similarity clustering)
- Lexical complexity (unique tokens, vocabulary size)
- Error type distribution

---

## Model Training

### Benchmark Pre-trained Models

Before training, benchmark base models to select the best foundation:

```bash
cd src/model/benchmark
python3 nce_benchmark.py
```

**Evaluated Models**:
- `google/gemma-300m` (selected)
- `sentence-transformers/all-MiniLM-L6-v2`
- `microsoft/codebert-base`
- `Salesforce/codet5-small`

**Selection Metric**: Separation margin (distance between positive and negative pairs)

### Training with LoRA

```bash
cd src/model/training
python3 nce_trainer.py
```

**Training Configuration**:
```python
LoRAConfig:
    r=16                    # LoRA rank
    lora_alpha=32           # Scaling factor
    target_modules=["q_proj", "v_proj"]
    task_type=FEATURE_EXTRACTION

TrainingArguments:
    num_train_epochs=5
    per_device_train_batch_size=64
    learning_rate=5e-5
    temperature=0.07        # InfoNCE temperature
    fp16=True              # Mixed precision
```

**Hardware Requirements**:
- Training: NVIDIA A40 (48GB VRAM) or equivalent
- Time: ~2 hours for 5 epochs
- Only 1% of model parameters are trained (LoRA efficiency)

**Output**: `data/models/best_model_final/`

### Push Model to HuggingFace Hub

```bash
cd src/data_engineering/generation
python3 push_to_hub.py
```

---

## Deployment

FFGen provides an interactive Streamlit application for real-time feedback retrieval.

### Launch Application

```bash
cd src/deployment/streamlit_app
streamlit run app.py
```

Access at: `http://localhost:8501`

### Application Features

**Main Page** (`app.py`):
- Submit buggy C code
- Real-time semantic search in feedback database
- Display top-10 most similar feedbacks with confidence scores
- Show associated code examples for each feedback

**Stats Page** (`pages/stats.py`):
- Cache hit/miss rate tracking
- Token consumption monitoring (for cache misses)
- Similarity score distributions
- Error category analytics
- Export statistics (CSV/JSONL)

### Cache System (Online Distillation)

The application implements an intelligent caching mechanism:

1. **Cache Hit** (similarity > 0.3): Return pre-indexed feedback from ChromaDB
2. **Cache Miss** (similarity <= 0.3): Call DeepSeek API and add result to cache

**Configuration** (`config.py`):
```python
SIMILARITY_THRESHOLD = 0.3
CONFIDENCE_THRESHOLD_WARNING = 0.9
TOP_K_RESULTS = 3
```

**Logging**:
- `stats.jsonl` - All queries with hit/miss status and timing
- `cache_miss.jsonl` - New feedbacks in dataset format for retraining

---

## Experiments and Analysis

### Jupyter Notebooks

Located in `experiments/notebooks/`:

1. **`code_clustering_analysis.ipynb`**: Student code clustering analysis
2. **`data_cleaning.ipynb`**: Dataset deduplication pipeline
3. **`push_to_hub.ipynb`**: HuggingFace dataset upload workflow

### Visualizations

- `tsne_visualization.png` - 2D projection of code embeddings
- `similarity_distributions.png` - Similarity score histograms
- `embedding_statistics_comparison.png` - Model comparison charts

### Performance Analysis

**Evolution Across Dataset Versions** (Chapter 6):
```
RAFT Raw (11,806 samples)          → Recall@10: 30.3%
RAFT Deduplicated (10,822 samples) → Recall@10: 50.3% (+20pts)
SYNT Synthetic (7,023 samples)     → Recall@10: 85.0% (+35pts)
```

**Key Finding**: Quality > Quantity. Smaller synthetic dataset with full context outperforms larger real dataset by +35 points.

---

## Reproducibility

### Complete Pipeline Reproduction

```bash
# 1. Generate dataset
cd src/data_engineering/generation
python3 generate_dataset_with_metrics.py

# 2. Analyze quality
cd ../analysis
python3 dataset_quality_analysis.py --datasets local_file.jsonl

# 3. Train model
cd ../../model/training
python3 nce_trainer.py

# 4. Deploy application
cd ../../deployment/streamlit_app
streamlit run app.py
```

### Expected Results

**Training Metrics** (5 epochs):
- Train Loss: 2.34 → 1.01
- Validation Loss: 1.89 → 1.33
- Validation Recall@10: 7.2% → 74.0%

**Inference Performance**:
- Latency: 50ms average (CPU)
- Throughput: 20 queries/second
- Memory: ~1.5GB (model + ChromaDB index)

### Verification Tests

```bash
# Test dataset generation validation
cd src/data_engineering/generation
python3 test_validation.py

# Test deployment system
cd ../../deployment/streamlit_app
python3 test_system.py
```

---

## Project Structure

```
FFGen/
├── src/
│   ├── data_engineering/
│   │   ├── generation/              # Chapter 2: Dataset generation scripts
│   │   │   ├── generate_dataset_with_metrics.py  # Main generator
│   │   │   ├── monitor_generation.py             # Real-time dashboard
│   │   │   ├── analyze_metrics.py                # Post-generation analysis
│   │   │   ├── test_validation.py                # Unit tests
│   │   │   ├── push_to_hub.py                    # Model upload
│   │   │   └── push_clean_dataset.py             # Dataset upload
│   │   └── analysis/                # Chapter 2: Quality analysis
│   │       ├── dataset_quality_analysis.py
│   │       └── compare_datasets.py
│   ├── model/                       # Chapters 3-4: Model architecture & training
│   │   ├── training/
│   │   │   ├── nce_trainer.py       # Main training script
│   │   │   └── nce_trainers.py      # Training utilities
│   │   └── benchmark/
│   │       └── nce_benchmark.py     # Model selection
│   └── deployment/                  # Chapter 5: Production deployment
│       └── streamlit_app/
│           ├── app.py               # Main application
│           ├── pages/
│           │   ├── search.py        # Search interface
│           │   └── stats.py         # Statistics dashboard
│           ├── config.py            # Configuration
│           ├── cache_manager.py     # Hit/miss logic
│           ├── deepseek_caller.py   # API wrapper
│           ├── stats_logger.py      # Logging system
│           └── test_system.py       # Unit tests
├── data/
│   ├── datasets/
│   │   ├── final_dataset/           # RAFT dataset (11,806 samples)
│   │   └── ultra_clean_final_dataset/  # SYNT dataset (7,023 samples)
│   └── models/
│       └── best_model_final/        # Trained model checkpoint
├── experiments/
│   └── notebooks/                   # Chapter 6: Experimental analysis
│       ├── code_clustering_analysis.ipynb
│       ├── data_cleaning.ipynb
│       └── push_to_hub.ipynb
├── legacy/                          # Deprecated code (preserved for reference)
├── tests/                           # Unit tests
├── docs/                            # This documentation
├── main.tex                         # Research paper (LaTeX)
├── pyproject.toml                   # Python project configuration
└── README.md                        # Quick start guide
```

**Module Organization**:
- `src/data_engineering/` - ETL pipeline (Extract, Transform, Load)
- `src/model/` - Neural network training and benchmarking
- `src/deployment/` - Production-ready application
- `experiments/` - Research notebooks and analysis
- `data/` - Datasets and model checkpoints
- `legacy/` - Historical code (not maintained)

---

## Citation

If you use FFGen in your research, please cite:

```bibtex
@misc{codjia2026ffgen,
  title={FFGen: Focused Feedback Generation for Code Education via Neural Retrieval},
  author={Codjia, Matis},
  year={2026},
  institution={IONIS Education Group & LIP6, Sorbonne Université},
  note={5-month research internship project}
}
```

**Contact**: Matis Codjia
**Advisors**: Julien Perez (IONIS CEPIA), Amel Yessad (LIP6, Sorbonne Université)

---

## License

This project is part of academic research. Code and datasets are available for research and educational purposes.

**Note**: Model weights are hosted on HuggingFace Hub under `matis35/gemmaembedding-fgdor`.

---

## Acknowledgments

- **IONIS Education Group** - CEPIA Research Lab
- **LIP6** - Sorbonne Université
- **DeepSeek AI** - API access for dataset generation
- **Google** - Gemma base model
- **HuggingFace** - Infrastructure for model hosting

---

**Last Updated**: January 2026
**Project Duration**: 5 months (2025-2026)
**Status**: Research complete, production-ready prototype deployed
