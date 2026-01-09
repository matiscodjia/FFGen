# FFGen Quick Start Guide

## For Reviewers

1. **Read the Paper**: [main.tex](main.tex)
2. **Read Documentation**: [docs/README.md](docs/README.md)
3. **Review Structure**: [STRUCTURE.md](STRUCTURE.md)

## For Reproducibility

```bash
# 1. Install dependencies
pip install -e .

# 2. Set environment variables
export DEEPSEEK_API_KEY="your-key"
export HF_TOKEN="your-token"

# 3. Run complete pipeline
cd src/data_engineering/generation
python3 generate_dataset_with_metrics.py

cd ../../model/training
python3 nce_trainer.py

cd ../../deployment/streamlit_app
streamlit run app.py
```

## For Code Navigation

- **Dataset Generation**: [src/data_engineering/generation/](src/data_engineering/generation/)
- **Model Training**: [src/model/training/](src/model/training/)
- **Web Application**: [src/deployment/streamlit_app/](src/deployment/streamlit_app/)
- **Analysis Notebooks**: [experiments/notebooks/](experiments/notebooks/)

## For Understanding Changes

See [REORGANIZATION_SUMMARY.md](REORGANIZATION_SUMMARY.md) for complete reorganization details.

## Key Files

| File | Purpose |
|------|---------|
| [README.md](README.md) | Project overview |
| [docs/README.md](docs/README.md) | Complete documentation |
| [STRUCTURE.md](STRUCTURE.md) | Directory reference |
| [main.tex](main.tex) | Research paper |
| [pyproject.toml](pyproject.toml) | Dependencies |

## Documentation Map

```
README.md               → Quick overview (you are here)
docs/README.md         → Complete documentation (installation, usage, API)
STRUCTURE.md           → Directory layout and file purposes
REORGANIZATION_SUMMARY.md → What changed and why
main.tex               → Research paper (LaTeX)
```

## Common Tasks

**Generate Dataset**:
```bash
cd src/data_engineering/generation
python3 generate_dataset_with_metrics.py
```

**Train Model**:
```bash
cd src/model/training
python3 nce_trainer.py
```

**Launch App**:
```bash
cd src/deployment/streamlit_app
streamlit run app.py
```

**Analyze Quality**:
```bash
cd src/data_engineering/analysis
python3 dataset_quality_analysis.py --datasets matis35/SYNT_DATASET
```

## Contact

**Author**: Matis Codjia
**Advisors**: Julien Perez (IONIS), Amel Yessad (LIP6)
**Duration**: 5 months (2025-2026)
