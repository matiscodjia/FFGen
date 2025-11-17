# FFGen Makefile - Commandes raccourcies avec uv

.PHONY: help install train viewer generate-neg test clean docker-viewer

# Default target
help:
	@echo "FFGen - Available Commands:"
	@echo ""
	@echo "=== Installation ==="
	@echo "  make install          Install dependencies with uv"
	@echo "  make install-dev      Install with dev dependencies"
	@echo ""
	@echo "=== Data Acquisition ==="
	@echo "  make extract-codes    Extract and merge code from codes/ directory"
	@echo "  make parquet-to-jsonl Convert parquet to JSONL format"
	@echo ""
	@echo "=== Feedbacks Generation ==="
	@echo "  make feedbacks-gen    Launch resumable feedback generation"
	@echo "  make synthetic-gen    Generate dummy dataset with feedbacks"
	@echo ""
	@echo "=== Data Processing ==="
	@echo "  make mine-hybrid      Mine hybrid negatives (hard + random) [RECOMMENDED]"
	@echo "  make mine-random      Mine random negatives only"
	@echo "  make example-hybrid   Example: 2 hard + 3 easy negatives"
	@echo "  make example-random   Example: 5 random negatives"
	@echo ""
	@echo "=== Training ==="
	@echo "  make train            Run training with default config"
	@echo "  make train-mini       Quick train with MiniLM"
	@echo "  make train-gemma      Train with EmbeddingGemma"
	@echo "  make quick-train      Quick training (1 epoch, 100 samples)"
	@echo ""
	@echo "=== Evaluation & Visualization ==="
	@echo "  make evaluate-rag     Evaluate RAG performance"
	@echo "  make viewer           Launch triplet viewer (with RAG + ChromaDB)"
	@echo ""
	@echo "=== Inference Servers ==="
	@echo "  make test-inference   Test inference service with fallback"
	@echo "  make test-fallback    Test fallback mechanism"
	@echo ""
	@echo "=== Development ==="
	@echo "  make clean            Clean cache and temp files"
	@echo "  make deep-clean       Clean everything (venv, models, logs)"
	@echo "  make info             Show environment info"
	@echo ""
	@echo "=== Key Environment Variables ==="
	@echo "  Hybrid negatives:"
	@echo "    INPUT=input.jsonl OUTPUT=output.jsonl"
	@echo "    TOTAL=5 HARD=2 MIN_SIM=0.2 MAX_SIM=0.4"
	@echo "  Training:"
	@echo "    DATA=data.jsonl MODEL=model-name"
	@echo "    EPOCHS=10 BATCH_SIZE=4"
	@echo "-------------------> For more information please read docs/README_MASTER"

# Installation
install:
	uv sync

install-dev:
	uv sync --all-extras

# Data Acquisition
extract-codes:
	@echo "Extracting and merging code from codes/ directory..."
	uv run python utils/extract_and_merge_codes.py

parquet-to-jsonl:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make parquet-to-jsonl INPUT=file.parquet OUTPUT=file.jsonl"; \
		exit 1; \
	fi
	uv run python -c "from utils.parquet_to_jsonl import parquet_to_jsonl; parquet_to_jsonl('$(INPUT)', '$(OUTPUT)')"

# Feedbacks generation
feedbacks-gen:
	uv run python pipeline.py --stage 2
synthetic-gen:
	uv run python 2_data_generation/generate_synthetic_dataset.py

# Training
train:
	uv run python 4_model_training/train_embedding.py \
	  --data $(or $(DATA),data/Exp-002-llama3B_v2_multi_neg.jsonl) \
	  --model $(or $(MODEL),google/embeddinggemma-300m) \
	  --batch-size $(or $(BATCH_SIZE),4) \
	  --epochs $(or $(EPOCHS),10) \
	  --lr 2e-5 \
	  --margin 0.5

# Training avec config spécifique
train-mini:
	@echo "Training with MiniLM (fast)..."
	uv run python 4_model_training/train_embedding.py \
	  --data data/Exp-002-llama3B_v2_multi_neg.jsonl \
	  --model sentence-transformers/all-MiniLM-L6-v2 \
	  --batch-size 8 \
	  --epochs 5

train-gemma:
	@echo "Training with EmbeddingGemma (recommended)..."
	uv run python 4_model_training/train_embedding.py \
	  --data data/Exp-002-llama3B_v2_multi_neg_10.jsonl \
	  --model google/embeddinggemma-300m \
	  --batch-size 16 \
	  --epochs 6

# Viewer (with RAG + ChromaDB cache)
viewer:
	@echo "Launching Triplet Viewer with RAG and ChromaDB cache..."
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run streamlit run 5_triplet_viewer_app/app.py

viewer-port:
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run streamlit run 5_triplet_viewer_app/app.py --server.port $(or $(PORT),8501)

# Data generation - Hybrid negatives (recommended)
mine-hybrid:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make generate-hybrid INPUT=input.jsonl OUTPUT=output.jsonl [TOTAL=5] [HARD=2]"; \
		exit 1; \
	fi
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python 3_data_processing/generate_hybrid_negatives.py \
	  --input $(INPUT) \
	  --output $(OUTPUT) \
	  --total-negatives $(or $(TOTAL),5) \
	  --num-hard $(or $(HARD),2) \
	  --min-similarity $(or $(MIN_SIM),0.2) \
	  --max-similarity $(or $(MAX_SIM),0.4) \
	  --embedding-model $(or $(MODEL),google/embeddinggemma-300m)

# Data generation - Random negatives only
mine-random:
	@if [ -z "$(INPUT)" ] || [ -z "$(OUTPUT)" ]; then \
		echo "Usage: make generate-random INPUT=input.jsonl OUTPUT=output.jsonl [NUM=5]"; \
		exit 1; \
	fi
	uv run python utils/add_multiple_random_negatives.py \
	  --input $(INPUT) \
	  --output $(OUTPUT) \
	  --num-negatives $(or $(NUM),5)


example-hybrid:
	@echo "Example: Generating 5 negatives (2 hard + 3 easy)..."
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python 3_data_processing/generate_hybrid_negatives.py \
	  --input data/Exp-002-llama3B_v2.jsonl \
	  --output data/Exp-002-llama3B_v2_hybrid.jsonl \
	  --total-negatives 5 \
	  --num-hard 2 \
	  --min-similarity 0.2 \
	  --max-similarity 0.4

# Example: 5 random negatives only
example-random:
	@echo "Example: Generating 5 random negatives..."
	uv run python utils/add_multiple_random_negatives.py \
	  --input data/Exp-002-llama3B_v2.jsonl \
	  --output data/Exp-002-llama3B_v2_multi_neg.jsonl \
	  --num-negatives 5

create-requests:
	@echo "Mining request for RAG testing..."
	uv run utils/create_test_queries.py \
    --dataset data/Exp-002-llama3B_v2.jsonl \
    --output data/test_queries.jsonl \
    --num-queries 100
rag-test:
	@echo "Mining request for RAG testing..."
	uv run utils/rag_evaluation.py \
    --model google/embeddinggemma-300m \
    --corpus data/Exp-002-llama3B_v2.jsonl \
    --queries data/test_queries.jsonl \
    --output results/rag_metrics.json
# Cleaning
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage

deep-clean: clean
	rm -rf .venv/
	rm -rf models/
	rm -rf training_logs/

# Docker (viewer only)
docker-viewer-build:
	docker build -f Dockerfile.viewer -t ffgen-viewer .

docker-viewer-run:
	docker run -p 8501:8501 \
	  -v $(PWD)/data:/app/data \
	  ffgen-viewer

docker-viewer: docker-viewer-build docker-viewer-run

# Development
format:
	uv run black src/ tests/
	uv run ruff check --fix src/ tests/

lint:
	uv run ruff check src/ tests/
	uv run mypy src/

# Quick commands
quick-train:
	@echo "Quick training (1 epoch, small batch)..."
	uv run python 4_model_training/train_embedding.py \
	  --data data/Exp-002-llama3B_v2_multi_neg.jsonl \
	  --model sentence-transformers/all-MiniLM-L6-v2 \
	  --batch-size 2 \
	  --epochs 1 \
	  --max-samples 100

# RAG Evaluation
evaluate-rag:
	@if [ -z "$(MODEL)" ] || [ -z "$(CORPUS)" ] || [ -z "$(QUERIES)" ]; then \
		echo "Usage: make evaluate-rag MODEL=path/to/model CORPUS=corpus.jsonl QUERIES=queries.jsonl"; \
		exit 1; \
	fi
	uv run python utils/rag_evaluation.py \
	  --model $(MODEL) \
	  --corpus $(CORPUS) \
	  --queries $(QUERIES) \
	  --output $(or $(OUTPUT),rag_metrics.json)

# Example RAG evaluation
example-evaluate:
	@echo "Running example RAG evaluation..."
	@echo "Note: Requires corpus and queries files. Adjust paths as needed."
	uv run python utils/rag_evaluation.py \
	  --model models/trained_model \
	  --corpus data/Exp-002-llama3B_v2_multi_neg.jsonl \
	  --queries data/test_queries.jsonl \
	  --output results/rag_metrics.json \
	  --k-values 1 5 10

# Inference Server Testing
test-inference:
	@echo "Testing InferenceServer with fallback..."
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python utils/inference_service.py

test-fallback:
	@echo "Testing fallback mechanism (server unavailable)..."
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python test_fallback.py

test-embeddings:
	@echo "Testing embeddings integration..."
	PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python test_embeddings_integration.py

# Environment info
info:
	@echo "=== FFGen Environment Info ==="
	@echo "Python: $(shell python --version)"
	@echo "uv: $(shell uv --version)"
	@echo "Working dir: $(PWD)"
	@echo "Data files: $(shell ls -1 data/*.jsonl 2>/dev/null | wc -l)"
	@echo "Models: $(shell ls -1d models/* 2>/dev/null | wc -l)"
	@uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@uv run python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
	@echo ""
	@echo "=== Inference Servers ==="
	@echo "Embedding server (llama.cpp): http://localhost:8000/v1"
	@echo "LLM server (LM Studio):       http://localhost:1234/v1"
