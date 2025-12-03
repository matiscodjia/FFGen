.PHONY: help install extract generate paraphrase viewer clean

help:
	@echo "FFGen - Makefile Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install dependencies"
	@echo "  make extract     - Extract code snippets"
	@echo "  make generate    - Generate feedbacks with Mistral API"
	@echo "  make paraphrase  - Generate paraphrases for augmentation"
	@echo "  make viewer      - Launch viewer app"
	@echo "  make clean       - Clean temporary files"
	@echo ""

install:
	uv sync

extract:
	uv run python3 extract_code.py

generate:
	@if [ -z "$$MISTRAL_API_KEY" ]; then \
		echo "Error: MISTRAL_API_KEY not set"; \
		echo "Set it with: export MISTRAL_API_KEY='your-key'"; \
		exit 1; \
	fi
	uv run python3 generate_feedbacks.py

paraphrase:
	@if [ -z "$$MISTRAL_API_KEY" ]; then \
		echo "Error: MISTRAL_API_KEY not set"; \
		echo "Set it with: export MISTRAL_API_KEY='your-key'"; \
		exit 1; \
	fi
	uv run python3 generate_paraphrases.py

viewer:
	uv run streamlit run viewer_app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned temporary files"
