# FFGen - Focused Feedback Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Research](https://img.shields.io/badge/license-Research-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-LaTeX-red.svg)](main.tex)

A neural retrieval system for automated pedagogical feedback in programming education.

## Overview

FFGen addresses the feedback bottleneck in large-scale programming courses by using fine-tuned embedding models instead of expensive generative AI. The system retrieves contextually appropriate explanations for buggy code in under 50ms with zero inference cost.

**Key Results**:
- 74% Recall@10 (vs. 7.2% baseline)
- 50ms average latency
- 300M parameter model (50x smaller than GPT-4)
- $0 per inference

**Research Context**: 5-month project conducted at IONIS Education Group (CEPIA Lab) and LIP6 (Sorbonne Université), 2025-2026.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Generate synthetic dataset
cd src/data_engineering/generation
python3 generate_dataset_with_metrics.py

# Train model (requires GPU)
cd ../../model/training
python3 nce_trainer.py

# Launch web application
cd ../../deployment/streamlit_app
streamlit run app.py
```

## Architecture

```
Student Code → Encoder → Vector (768d) ──┐
                                         │→ Cosine Similarity → Retrieve Feedback
Feedback DB → Encoder → Vectors (768d) ──┘
```

- **Base Model**: Google Gemma-300M
- **Fine-Tuning**: LoRA (rank=16, ~1% parameters trained)
- **Loss**: InfoNCE contrastive loss (temperature=0.07)
- **Vector DB**: ChromaDB with HNSW index

## Project Structure

```
src/
├── data_engineering/    # Dataset generation and quality analysis
├── model/              # Training and benchmarking
└── deployment/         # Production Streamlit application

data/
├── datasets/           # RAFT (11.8k) and SYNT (7k) datasets
└── models/            # Trained model checkpoints

experiments/
└── notebooks/         # Analysis and visualizations

docs/
└── README.md         # Complete documentation
```

## Documentation

**Full documentation**: [docs/README.md](docs/README.md)

Includes:
- Complete installation guide
- Data engineering pipeline details
- Training configuration and hyperparameters
- Deployment instructions with cache system
- Reproducibility guide
- Experimental results and analysis

## Research Paper

The complete research report is available in `main.tex` (LaTeX source).

**Chapters**:
1. Context and Problem Statement
2. Data Engineering (3 generation approaches)
3. Model Architecture
4. Training and Optimization
5. Deployment
6. Experiments and Results

## Citation

```bibtex
@misc{codjia2026ffgen,
  title={FFGen: Focused Feedback Generation via Neural Retrieval},
  author={Codjia, Matis},
  year={2026},
  institution={IONIS Education Group & LIP6, Sorbonne Université}
}
```

## Contact

**Author**: Matis Codjia
**Advisors**: Julien Perez (IONIS CEPIA), Amel Yessad (LIP6)
**Period**: 5 months (2025-2026)

## License

Academic research project. Code and datasets available for research and educational purposes.

**Model**: `matis35/gemmaembedding-fgdor` on HuggingFace Hub
**Dataset**: `matis35/SYNT_DATASET` on HuggingFace Hub
