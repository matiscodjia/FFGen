# Stage 1: Data Mining and Acquisition

This directory contains modules for extracting, parsing, and validating source code files.

## Modules

### `ingest_code.py`
Main data acquisition module that:
- Recursively scans directories for source code files
- Extracts code snippets and metadata
- Generates unique identifiers
- Outputs structured data in parquet format

**Main Function**: `run_data_acquisition(config)`

**Usage**:
```python
from data_acquisition import run_data_acquisition

config = {
    'paths': {
        'source_code_dir': './codes',
        'processed_collection': './data/processed.parquet'
    }
}

output_path = run_data_acquisition(config)
```

### `validate_code.py`
Code compilation validation utility that:
- Compiles C code snippets using GCC
- Captures compiler warnings and errors
- Adds compilation results to dataset

**Usage**:
```bash
python validate_code.py input.jsonl output.jsonl
```

## Output Format

Stage 1 produces a parquet file with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `code_id` | string | Unique identifier (UUID) |
| `author_id` | string | Anonymized author identifier |
| `code_snippet` | string | Extracted code content |
| `header`| string | Extracted function signature or instructions|

## Configuration

Configure Stage 1 in your YAML config file:

```yaml
paths:
  source_code_dir: "./codes"  # Input directory
  processed_collection: "./data/processed_collection.parquet"  # Output file
```

## Supported Languages

Currently supports C code files (.c extension).
