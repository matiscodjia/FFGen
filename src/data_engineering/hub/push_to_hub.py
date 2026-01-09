"""
Push the trained LoRA model to Hugging Face Hub
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login
from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer

# Configuration
LORA_CHECKPOINT = "./checkpoints/lora_contrastive_fixed/best_model"
BASE_MODEL = "Salesforce/SFR-Embedding-Code-400M_R"
HUB_MODEL_NAME = "matis35/SFR-Embedding-Code-400M-LoRA-Feedback"  # Change this to your desired name

print("=" * 80)
print("PUSH LORA MODEL TO HUGGING FACE HUB")
print("=" * 80)

# Step 1: Login to Hugging Face
print("\n1. Logging in to Hugging Face Hub...")
print("   Please enter your HF token when prompted (or set HF_TOKEN env variable)")

try:
    # Try to use environment variable first
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("   ✓ Logged in using HF_TOKEN environment variable")
    else:
        # Interactive login
        login()
        print("   ✓ Logged in successfully")
except Exception as e:
    print(f"   ✗ Login failed: {e}")
    print("\n   To fix this:")
    print("   1. Get your token from: https://huggingface.co/settings/tokens")
    print("   2. Run: huggingface-cli login")
    print("   3. Or set: export HF_TOKEN='your_token_here'")
    exit(1)

# Step 2: Verify checkpoint exists
print(f"\n2. Verifying checkpoint exists...")
checkpoint_path = Path(LORA_CHECKPOINT)
if not checkpoint_path.exists():
    print(f"   ✗ Checkpoint not found at: {LORA_CHECKPOINT}")
    print(f"   Available checkpoints:")
    checkpoints_dir = Path("./checkpoints/lora_contrastive_fixed")
    if checkpoints_dir.exists():
        for item in checkpoints_dir.iterdir():
            if item.is_dir():
                print(f"     - {item}")
    exit(1)

print(f"   ✓ Checkpoint found at: {LORA_CHECKPOINT}")

# Step 3: Load and verify the model
print(f"\n3. Loading LoRA model...")
try:
    # Load config first to verify it's valid
    config = PeftConfig.from_pretrained(LORA_CHECKPOINT)
    print(f"   ✓ LoRA config loaded successfully")
    print(f"     - LoRA r: {config.r}")
    print(f"     - LoRA alpha: {config.lora_alpha}")
    print(f"     - LoRA dropout: {config.lora_dropout}")
    print(f"     - Target modules: {config.target_modules}")

    # Load base model
    print(f"\n   Loading base model: {BASE_MODEL}")
    base_model = AutoModel.from_pretrained(BASE_MODEL)

    # Load LoRA weights
    print(f"   Loading LoRA weights from checkpoint...")
    model = PeftModel.from_pretrained(base_model, LORA_CHECKPOINT)
    print(f"   ✓ Model loaded successfully")

except Exception as e:
    print(f"   ✗ Failed to load model: {e}")
    exit(1)

# Step 4: Load tokenizer
print(f"\n4. Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    print(f"   ✓ Tokenizer loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load tokenizer: {e}")
    exit(1)

# Step 5: Create model card
print(f"\n5. Creating model card...")
model_card = f"""---
language: en
license: mit
library_name: peft
tags:
- code
- embeddings
- lora
- contrastive-learning
- feedback
base_model: {BASE_MODEL}
---

# {HUB_MODEL_NAME}

This is a LoRA fine-tuned version of [{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) for code-feedback alignment.

## Model Description

This model was fine-tuned using **contrastive learning (InfoNCE loss)** to align code snippets with their corresponding feedback.
The model learns to embed code and feedback into a shared semantic space where:
- Matching code-feedback pairs have high cosine similarity
- Non-matching pairs have low similarity

## LoRA Configuration

- **LoRA rank (r)**: {config.r}
- **LoRA alpha**: {config.lora_alpha}
- **LoRA dropout**: {config.lora_dropout}
- **Target modules**: {', '.join(config.target_modules) if hasattr(config, 'target_modules') else 'N/A'}

## Training

The model was trained on the [RAFT dataset](https://huggingface.co/datasets/matis35/RAFT) containing code snippets and their feedback.

**Training strategy:**
- Loss: InfoNCE (contrastive learning)
- Temperature: 0.07
- Negative sampling: In-batch random negatives
- Batch size: 16

## Usage

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

# Load model
base_model = AutoModel.from_pretrained("{BASE_MODEL}")
model = PeftModel.from_pretrained(base_model, "{HUB_MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{BASE_MODEL}")

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def encode_text(text, max_length=512):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings

# Example usage
code = "def factorial(n): return 1 if n == 0 else n * factorial(n-1)"
feedback = "Consider adding error handling for negative inputs."

code_emb = encode_text(code, max_length=512)
feedback_emb = encode_text(feedback, max_length=256)

similarity = torch.matmul(code_emb, feedback_emb.T).item()
print(f"Similarity: {{similarity:.4f}}")
```

## Retrieval Performance

Evaluated on a held-out test set:

| Metric | Validation | Test |
|--------|------------|------|
| Recall@1 | 23.0% | 17.0% |
| Recall@5 | 68.0% | 61.5% |
| Recall@10 | 87.0% | 81.0% |
| MRR | 0.44 | 0.37 |

**Note:** The relatively low Recall@1 is expected because many code snippets can have multiple valid feedback messages.
The model often retrieves semantically relevant feedback even when it's not the exact match from the training set.

## Citation

If you use this model, please cite:

```bibtex
@misc{{sfr-embedding-code-lora-feedback,
  author = {{Matis Codjia}},
  title = {{SFR-Embedding-Code-400M LoRA Fine-tuned for Code-Feedback Alignment}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  howpublished = {{\\url{{https://huggingface.co/{HUB_MODEL_NAME}}}}}
}}
```

## License

MIT License
"""

# Save model card
model_card_path = checkpoint_path / "README.md"
with open(model_card_path, 'w', encoding='utf-8') as f:
    f.write(model_card)
print(f"   ✓ Model card created")

# Step 6: Push to Hub
print(f"\n6. Pushing model to Hub...")
print(f"   Target repository: {HUB_MODEL_NAME}")
print(f"   This may take a few minutes...")

try:
    # Push the model
    model.push_to_hub(
        HUB_MODEL_NAME,
        use_temp_dir=True,
        commit_message="Upload LoRA model for code-feedback alignment"
    )
    print(f"   ✓ Model pushed successfully!")

except Exception as e:
    print(f"   ✗ Failed to push model: {e}")
    print(f"\n   Troubleshooting:")
    print(f"   1. Make sure you have write access to the repository")
    print(f"   2. Check that the repository name is correct: {HUB_MODEL_NAME}")
    print(f"   3. Try pushing manually with: model.push_to_hub('{HUB_MODEL_NAME}')")
    exit(1)

# Step 7: Push tokenizer (optional but recommended)
print(f"\n7. Pushing tokenizer to Hub...")
try:
    tokenizer.push_to_hub(
        HUB_MODEL_NAME,
        use_temp_dir=True,
        commit_message="Upload tokenizer"
    )
    print(f"   ✓ Tokenizer pushed successfully!")
except Exception as e:
    print(f"   ⚠ Tokenizer push failed (non-critical): {e}")

# Step 8: Success message
print("\n" + "=" * 80)
print("✓ SUCCESS!")
print("=" * 80)
print(f"\nYour model is now available at:")
print(f"https://huggingface.co/{HUB_MODEL_NAME}")
print(f"\nTo use it:")
print(f"```python")
print(f"from peft import PeftModel")
print(f"from transformers import AutoModel")
print(f"")
print(f"base_model = AutoModel.from_pretrained('{BASE_MODEL}')")
print(f"model = PeftModel.from_pretrained(base_model, '{HUB_MODEL_NAME}')")
print(f"```")
print("\n" + "=" * 80)
