.PHONY: help install extract generate paraphrase viewer app clean

help:
	@echo "FFGen - Makefile Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make extract     - Extract code snippets"
	@echo "  make generate    - Generate feedbacks with Mistral API"
	@echo "  make paraphrase  - Generate paraphrases for augmentation"
	@echo "  make app         - Launch main showcase app"
	@echo "  make viewer      - Launch triplet viewer app (legacy)"
	@echo "  make clean       - Clean temporary files"
	@echo ""

install:
	uv sync

extract:
	uv run python3 scripts/extract_code.py

generate:
	@if [ -z "$$MISTRAL_API_KEY" ]; then \
		echo "Error: MISTRAL_API_KEY not set"; \
		echo "Set it with: export MISTRAL_API_KEY='your-key'"; \
		exit 1; \
	fi
	uv run python3 scripts/generate_feedbacks.py

paraphrase:
	@if [ -z "$$MISTRAL_API_KEY" ]; then \
		echo "Error: MISTRAL_API_KEY not set"; \
		echo "Set it with: export MISTRAL_API_KEY='your-key'"; \
		exit 1; \
	fi
	uv run python3 scripts/generate_paraphrases.py

app:
	@echo "🚀 Launching FFGen showcase app..."
	@echo "Open your browser at http://localhost:8501"
	@echo ""
	uv run streamlit run app_unified.py

viewer:
	@echo "🚀 Launching triplet viewer app..."
	@echo "Open your browser at http://localhost:8501"
	@echo ""
	uv run streamlit run viewer_app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned temporary files"
