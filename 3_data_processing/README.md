# Stage 2: Data Processing and Preprocessing

This directory contains modules for processing raw data, generating synthetic training examples using LLMs, and mining hard negatives using cosine similarity.

## Modules

### `generate_feedback.py` (Stage 2.1)
Multi-agent LLM feedback generation system that:
- Processes code snippets through a chain of LLM agents
- Generates positive and negative feedback examples
- Supports resumable batch processing
- Outputs training datasets in JSONL format

**Main Function**: `run_feedback_generation(config)`

**Agent Types**:
1. **Tutor**: Generates initial educational feedback
2. **Editor**: Refines and improves feedback quality
3. **Adversary**: Creates negative/incorrect examples
4. **Conceptual**: Extracts high-level concepts (optional)

**Usage**:
```python
from data_processing import run_feedback_generation

config = {
    'generation': {
        'llm_model': 'llama-3.2-3b-instruct',
        'batch_size': 8,
        'agents': [...],
        'final_dataset_file': './data/train.jsonl'
    },
    'paths': {
        'processed_collection': './data/processed.parquet'
    }
}

dataset_path = run_feedback_generation(config)
```

### `preprocess.py`
Data cleaning and validation utilities:
- Text normalization
- Length filtering
- Deduplication
- Record validation

**Functions**:
- `clean_code_snippet(code)`: Normalize code formatting
- `validate_record(record, fields)`: Check required fields
- `filter_by_length(df, column, min, max)`: Filter by text length
- `deduplicate_dataframe(df, column)`: Remove duplicates
- `preprocess_dataset(input, output)`: Complete preprocessing pipeline

