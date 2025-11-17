# Stage 2: Data Processing and Preprocessing

This directory contains modules for processing raw data, generating synthetic training examples using LLMs.

## Modules

### `generate_feedback.py` (Stage 2)
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


## Agent Configuration

Each agent is configured with:

```yaml
- agent: "agent_name"
  prompt_file: "./prompts/agent_prompt.txt"
  input_col: "input_field"
  output_col: "output_field"
```

Agents execute sequentially, with each agent's output becoming the next agent's input.

## Output Format

Stage 2 produces JSONL files where each line contains:

```json
{
  "code_id": "uuid",
  "code_snippet": "original code",
  "generated_feedback": "initial feedback",
  "refined_feedback": "improved feedback",
  "negative_feedback": "incorrect feedback",
  "conceptual_feedback": "high-level concepts"
}
```

## Resumable Processing

The pipeline automatically saves progress after each batch. If interrupted:
1. Rerun with same config
2. Pipeline loads existing progress
3. Only processes remaining items

Progress is tracked by `code_id` to prevent duplicates.

## LLM Configuration

Requires an OpenAI-compatible API endpoint:

```python
# Default: localhost:1234 (LM Studio, Ollama, etc.)
async_client = AsyncOpenAI(
    base_url="http://localhost:1234/v1",
    api_key="none"
)
```

To use different LLM providers, modify the client initialization in `generate_feedback.py`.
