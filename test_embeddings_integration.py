#!/usr/bin/env python3
"""
Test embeddings.py integration with fallback
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "5_triplet_viewer_app"))
from backend.embeddings import load_embedding_model, _encode_with_model

print("="*70)
print("Testing Embeddings Integration with Fallback")
print("="*70)

# Test 1: Load model with server (available)
print("\n1. Loading model with server (localhost:8000)...")
model_with_server = load_embedding_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    server_url="http://localhost:8000/v1",
    use_server=True
)
print(f"   Model type: {type(model_with_server).__name__}")

texts = ["Test embedding 1", "Test embedding 2"]
embeddings = _encode_with_model(model_with_server, texts)
print(f"   Encoded {len(texts)} texts")
print(f"   Embedding shape: {embeddings.shape}")

# Test 2: Load model with unavailable server (fallback)
print("\n2. Loading model with unavailable server (fallback)...")
model_fallback = load_embedding_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    server_url="http://localhost:9999/v1",
    use_server=True
)
print(f"   Model type: {type(model_fallback).__name__}")

embeddings_fallback = _encode_with_model(model_fallback, texts)
print(f"   Encoded {len(texts)} texts")
print(f"   Embedding shape: {embeddings_fallback.shape}")

# Test 3: Load model without server
print("\n3. Loading model without server (direct local)...")
model_local = load_embedding_model(
    "sentence-transformers/all-MiniLM-L6-v2",
    use_server=False
)
print(f"   Model type: {type(model_local).__name__}")

embeddings_local = _encode_with_model(model_local, texts)
print(f"   Encoded {len(texts)} texts")
print(f"   Embedding shape: {embeddings_local.shape}")

print("\n" + "="*70)
print("✓ All embeddings integration tests passed")
print("="*70)
