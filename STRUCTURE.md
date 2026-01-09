# FFGen Project Structure

This document provides a quick reference to the project organization.

## Directory Layout

```
FFGen/
├── src/                           # Source code (organized by paper chapters)
│   ├── data_engineering/         # Chapter 2: Data Engineering
│   │   ├── generation/           # Dataset generation with metrics
│   │   │   ├── generate_dataset_with_metrics.py
│   │   │   ├── monitor_generation.py
│   │   │   ├── analyze_metrics.py
│   │   │   ├── test_validation.py
│   │   │   ├── push_to_hub.py
│   │   │   └── push_clean_dataset.py
│   │   └── analysis/             # Quality analysis and comparison
│   │       ├── dataset_quality_analysis.py
│   │       └── compare_datasets.py
│   │
│   ├── model/                    # Chapters 3-4: Architecture & Training
│   │   ├── training/             # LoRA fine-tuning
│   │   │   ├── nce_trainer.py   # Main training script
│   │   │   └── nce_trainers.py  # Training utilities
│   │   └── benchmark/            # Model selection
│   │       └── nce_benchmark.py
│   │
│   └── deployment/               # Chapter 5: Production Deployment
│       └── streamlit_app/
│           ├── app.py            # Main application
│           ├── pages/
│           │   ├── search.py    # Search interface
│           │   └── stats.py     # Analytics dashboard
│           ├── config.py         # Configuration
│           ├── cache_manager.py  # Hit/miss logic
│           ├── deepseek_caller.py
│           ├── stats_logger.py
│           └── test_system.py
│
├── data/                          # Data artifacts
│   ├── datasets/                 # Training datasets
│   │   ├── final_dataset/       # RAFT (11,806 samples)
│   │   └── ultra_clean_final_dataset/  # SYNT (7,023 samples)
│   ├── models/                   # Model checkpoints
│   │   └── best_model_final/    # Trained Gemma-300M + LoRA
│   ├── raft_dataset.jsonl       # Original RAFT data
│   └── *.csv                     # Analysis results
│
├── experiments/                   # Chapter 6: Experiments & Analysis
│   └── notebooks/
│       ├── code_clustering_analysis.ipynb
│       ├── data_cleaning.ipynb
│       ├── push_to_hub.ipynb
│       └── *.png                 # Visualization outputs
│
├── legacy/                        # Deprecated code (preserved for reference)
│   ├── scripts/                  # Old structure before reorganization
│   ├── training/                 # Original training scripts
│   ├── streamlit_rag_viewer/    # Old app version
│   ├── notebooks/                # Duplicate notebooks
│   └── *.md                      # Old documentation
│
├── tests/                         # Unit tests (to be populated)
│
├── docs/
│   └── README.md                 # Comprehensive documentation
│
├── main.tex                       # Research paper (LaTeX)
├── README.md                      # Project overview
├── pyproject.toml                # Python dependencies
├── uv.lock                       # Dependency lock file
├── Makefile                      # Build commands
├── config.yml                    # Application configuration
└── prompt.txt                    # LLM prompts for generation
```

## Module Mapping to Paper Chapters

| Chapter | Content | Directory |
|---------|---------|-----------|
| 1 | Context and Problem | N/A (paper only) |
| 2 | Data Engineering | `src/data_engineering/` |
| 3 | Model Architecture | `src/model/` |
| 4 | Training & Optimization | `src/model/training/` |
| 5 | Deployment | `src/deployment/` |
| 6 | Experiments & Results | `experiments/` |

## Key File Purposes

### Data Engineering

- **generate_dataset_with_metrics.py**: Main dataset generator (DeepSeek API)
  - Generates synthetic C code with intentional bugs
  - Validates compilation with GCC
  - Tracks tokens, retries, and timing
  - Output: `dataset_c_piscine_semantic_validated.jsonl`

- **monitor_generation.py**: Real-time dashboard
  - Shows progress, success rate, token consumption
  - Updates every 2 seconds during generation

- **analyze_metrics.py**: Post-generation analysis
  - Computes retry distribution, failure reasons
  - Analyzes token efficiency

- **dataset_quality_analysis.py**: Quality metrics
  - Semantic diversity (entropy)
  - Redundancy detection
  - Lexical complexity

### Model Training

- **nce_trainer.py**: Main training loop
  - Loads base model (Gemma-300M)
  - Applies LoRA adapters (rank=16)
  - Trains with InfoNCE loss
  - Saves best checkpoint

- **nce_benchmark.py**: Base model selection
  - Compares pre-trained models
  - Measures separation margin
  - Selects best foundation for fine-tuning

### Deployment

- **app.py**: Streamlit web interface
  - User submits buggy code
  - Semantic search in ChromaDB
  - Returns top-10 feedbacks

- **cache_manager.py**: Online distillation
  - Cache hit if similarity > 0.3
  - Cache miss calls DeepSeek API
  - Adds new feedbacks to ChromaDB

- **stats_logger.py**: Analytics
  - Logs all queries to `stats.jsonl`
  - Logs cache misses to `cache_miss.jsonl`

## Legacy Directory

The `legacy/` folder contains the original project structure before reorganization:

- **Purpose**: Preserve historical code for reference
- **Status**: Not maintained, may contain outdated dependencies
- **Use Case**: Compare old vs new implementations, recover specific scripts

**Note**: All active development happens in `src/`, `data/`, and `experiments/`.

## Development Workflow

1. **Generate Data**: Run `src/data_engineering/generation/generate_dataset_with_metrics.py`
2. **Analyze Quality**: Run `src/data_engineering/analysis/dataset_quality_analysis.py`
3. **Benchmark Models**: Run `src/model/benchmark/nce_benchmark.py`
4. **Train Model**: Run `src/model/training/nce_trainer.py`
5. **Deploy**: Run `streamlit run src/deployment/streamlit_app/app.py`

## Testing

Unit tests are located in `tests/` (currently minimal). Key test files:

- `src/data_engineering/generation/test_validation.py` - Validation tests
- `src/deployment/streamlit_app/test_system.py` - Deployment tests

Run tests with:
```bash
python3 src/data_engineering/generation/test_validation.py
python3 src/deployment/streamlit_app/test_system.py
```

## Configuration Files

- **pyproject.toml**: Python project metadata and dependencies
- **uv.lock**: Locked dependency versions (for reproducibility)
- **config.yml**: Application settings (legacy)
- **prompt.txt**: LLM generation prompts
- **Makefile**: Build automation commands

## Environment Variables

Required for full pipeline:

```bash
export DEEPSEEK_API_KEY="..."      # Dataset generation
export HF_TOKEN="..."               # HuggingFace downloads/uploads
```

## Output Artifacts

Generated during pipeline execution:

```
dataset_c_piscine_semantic_validated.jsonl  # Final dataset
generation_metrics.jsonl                     # Per-sample metrics
generation_summary.json                      # Global statistics
stats.jsonl                                  # Deployment queries
cache_miss.jsonl                             # Cache misses
data/models/best_model_final/                # Trained model
```

## Notes

- All Python code uses English docstrings and comments
- All emojis removed from source code
- Module structure aligns with paper chapters
- Legacy code preserved in separate directory
