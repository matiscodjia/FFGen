# RAFT Dataset - Cleaned and Deduplicated

## Dataset Statistics

- **Total samples**: 10,822
  - Train: 8,799 (81.3%)
  - Validation: 1,083 (10.0%)
  - Test: 940 (8.7%)

## Cleaning Process

This dataset has been thoroughly cleaned to maximize feedback diversity and quality:

1. **Original dataset**: 11,806 entries
2. **Exact deduplication**: Removed 745 exact duplicates (6.3%)
3. **Semantic deduplication**: Removed 239 semantically similar entries (2.2%)
4. **Final dataset**: 10,822 entries (91.7% of original)

### Quality Improvements

- ✅ No exact duplicate feedbacks
- ✅ Minimal semantic overlap between feedbacks (< 85% similarity)
- ✅ Stratified splits (no feedback leakage between train/val/test)
- ✅ Diverse code examples for each feedback pattern
- ✅ Balanced distribution across feedback categories

## Data Format

Each entry contains:
- `code`: The code snippet (C code)
- `feedback`: The generated feedback for the code
- `code_id`: Unique identifier for the code
- `author_id`: Identifier for the code author

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("json", data_files={
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl"
})

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]

# Example entry
print(train_data[0])
# {
#   "code": "int my_function() { ... }",
#   "feedback": "Consider the edge case where...",
#   "code_id": "abc-123",
#   "author_id": "xyz-789"
# }
```

## Training Recommendations

For contrastive learning (InfoNCE loss):
- Use batch sizes that allow for sufficient negative examples (128-256)
- Temperature parameter: 0.05-0.1
- Ensure no duplicate feedbacks in a single batch
- Consider using stratified sampling to maintain diversity

## Quality Metrics

- Average feedback length: ~177 characters
- Feedback specificity: 76.5% contain specific technical terms
- Code-feedback correlation: 0.206 (feedbacks are appropriately independent of code length)

---

Generated with FFGen dataset preparation pipeline
