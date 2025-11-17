# Configuration Guide

This directory contains configuration files for the FFGen ML pipeline.

## Configuration Structure

A configuration file defines all parameters for the three pipeline stages:

### 1. Experiment Identification
```yaml
run_id: "Exp-001"  # Unique identifier for this experiment
```

### 2. Stage 1: Data Acquisition
```yaml
paths:
  source_code_dir: "./codes"  # Directory containing source code files
  processed_collection: "./data/processed_collection.parquet"  # Output file
```

### 3. Stage 2: Data Processing (Feedback Generation)
```yaml
generation:
  llm_model: "llama-3.2-3b-instruct"  # LLM model for generation
  batch_size: 8  # Batch size for processing

  agents:  # Multi-agent pipeline
    - agent: "tutor"
      prompt_file: "./prompts/agent1_tutor.txt"
      input_col: "code_snippet"
      output_col: "generated_feedback"

    - agent: "editor"
      prompt_file: "./prompts/agent2_editor.txt"
      input_col: "generated_feedback"
      output_col: "refined_feedback"

    - agent: "adversary"
      prompt_file: "./prompts/agent3_adversary.txt"
      input_col: "generated_feedback"
      output_col: "negative_feedback"

  final_dataset_file: "./data/training_dataset.jsonl"
```

### 4. Stage 3: Model Training
```yaml
training:
  mode: "triplet"  # Options: "mnrl" or "triplet"
  base_embedding_model: "google/embeddinggemma-300m"

  data_columns:
    anchor: "code_snippet"
    positive: "refined_feedback"
    negative: "negative_feedback"

  hyperparameters:
    num_epochs: 5
    batch_size: 16
    learning_rate: 0.00032
    warmup_ratio: 0.1
    triplet_margin: 0.5

  evaluator_type: "triplet"  # Options: "ir" or "triplet"
  metric_for_best_model: "validation-set_cosine_accuracy"
  model_output_dir: "./models"
```

### 5. Results Reporting
```yaml
reporting:
  report_file: "./results.csv"
```

## Training Modes

### MNRL (Multiple Negatives Ranking Loss)
- Requires: `sentence1`, `sentence2` columns
- Good for: Learning to match similar pairs
- Use when: You have positive pairs but no explicit negatives

### Triplet Loss
- Requires: `anchor`, `positive`, `negative` columns
- Good for: Learning fine-grained distinctions
- Use when: You have explicit negative examples

## Example Configurations

See `config.yml` for the default configuration.

To create a new experiment:
1. Copy `config.yml` to `configs/experiment_name.yml`
2. Update the `run_id`
3. Modify parameters as needed
4. Run: `python pipeline.py --config configs/experiment_name.yml`
