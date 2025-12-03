# FFGen - Focused Feedback Generation

Production-ready system for code feedback generation using Mistral Large API.

## Quick Start

```bash
# 1. Install dependencies
uv pip install mistralai pyyaml tqdm streamlit

# 2. Set API key
export MISTRAL_API_KEY="your-key-here"

# 3. Extract code
python extract_code.py

# 4. Generate feedbacks
python generate_feedbacks.py

# 5. Generate paraphrases (optional, for better training)
python generate_paraphrases.py

# 6. View results
streamlit run viewer_app/app.py
```

## Core Files

- **config.yml** - Configuration
- **prompt.txt** - Mistral Large prompt for feedbacks
- **prompt_paraphrase.txt** - Prompt for paraphrases
- **extract_code.py** - Extract code snippets
- **generate_feedbacks.py** - Generate feedbacks via API
- **generate_paraphrases.py** - Generate paraphrases for training
- **test_setup.py** - Verify setup

## Apps

- **viewer_app/** - View, edit, analyze feedbacks

## Dataset

- **data/dataset.jsonl** - 14,000+ code-feedback pairs

## Makefile

```bash
make help        # Show commands
make install     # Install dependencies
make extract     # Extract codes
make generate    # Generate feedbacks
make paraphrase  # Generate paraphrases (data augmentation)
make viewer      # Launch viewer
```
