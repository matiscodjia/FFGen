"""
Push the MERGED model to Hugging Face Hub
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login
from transformers import AutoModel, AutoTokenizer

# --- CONFIGURATION ---
# Le dossier où ton script d'entraînement a sauvegardé le modèle fusionné
LOCAL_MODEL_PATH = "./final_merged_model" 

# Le nom du modèle de base original (juste pour la citation dans le README)
BASE_MODEL_REF = "google/embeddinggemma-300m"

# Le nom final sur le Hub (ex: ton-pseudo/Nom-Du-Modele)
HUB_MODEL_NAME = "matis35/feedbacker" 

print("=" * 80)
print("PUSH FULL MERGED MODEL TO HUGGING FACE HUB")
print("=" * 80)

# Step 1: Login
print("\n1. Logging in to Hugging Face Hub...")
try:
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("   ✓ Logged in using HF_TOKEN")
    else:
        login()
        print("   ✓ Logged in successfully")
except Exception as e:
    print(f"   ✗ Login failed: {e}")
    exit(1)

# Step 2: Verify path
print(f"\n2. Verifying local model path...")
model_path = Path(LOCAL_MODEL_PATH)
if not model_path.exists():
    print(f"   ✗ Directory not found: {LOCAL_MODEL_PATH}")
    print("   Please check where your training script saved the 'merged' model.")
    exit(1)
print(f"   ✓ Found model directory at: {LOCAL_MODEL_PATH}")

# Step 3: Load the merged model
print(f"\n3. Loading model and tokenizer from disk...")
try:
    # On ajoute le paramètre de correction regex pour éviter le warning Mistral
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_MODEL_PATH, 
            trust_remote_code=True, 
            fix_mistral_regex=True
        )
    except TypeError:
        # Fallback pour les anciennes versions de transformers
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        
    model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    print(f"   ✓ Model and Tokenizer loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load: {e}")
    exit(1)

# Step 4: Create Model Card (README.md)
print(f"\n4. Creating README.md...")

# Note : On utilise des doubles accolades {{ }} dans la f-string pour afficher de vraies accolades {} dans le README
readme_text = f"""---
language: c
license: mit
tags:
- code
- embeddings
- contrastive-learning
- retrieval
- feedback
base_model: {BASE_MODEL_REF}
library_name: transformers
pipeline_tag: feature-extraction
---

# {HUB_MODEL_NAME}

This is a **contrastive fine-tuned version** of [{BASE_MODEL_REF}](https://huggingface.co/{BASE_MODEL_REF}).  
It has been trained to align **C code snippets** with their corresponding **feedback/correction instructions**.

## Model Details

- **Architecture**: Bi-Encoder (Embedding Model)
- **Base Model**: {BASE_MODEL_REF}
- **Training Objective**: InfoNCE (Contrastive Loss)
- **Task**: Code-to-Feedback Retrieval

The model maps C code and textual feedback into the same vector space.  
- **Close vectors**: A piece of code and its correct feedback.
- **Distant vectors**: Unrelated code and feedback.

## Usage

This model can be used with `transformers` (and `torch`).

### Using Transformers

```python
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

model_id = "{HUB_MODEL_NAME}"

# 1. Load Model
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

# 2. Define inputs
code = "int main() {{ int x = 0; return x; }}"
feedback = "Good job, the main function correctly returns an integer."

# 3. Helper for pooling
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        # Mean Pooling
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # Normalize (Crucial for Cosine Similarity)
        return F.normalize(embeddings, p=2, dim=1)

# 4. Calculate Similarity
emb_code = get_embedding(code)
emb_feedback = get_embedding(feedback)

similarity = (emb_code @ emb_feedback.T).item()
print(f"Similarity Score: {{similarity:.4f}}")"""