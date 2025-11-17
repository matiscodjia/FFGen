#!/usr/bin/env python3
"""
Test fallback mechanism with unavailable server
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.inference_service import InferenceServer

async def test_fallback():
    print("="*70)
    print("Testing Fallback Mechanism (Server Unavailable)")
    print("="*70)

    # Test with unavailable server - should fallback to local model
    print("\n1. Testing embeddings with unavailable server...")
    embedder = InferenceServer(
        url="http://localhost:9999/v1",  # Invalid port - will trigger fallback
        fallback_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    texts = ["Hello world", "Bonjour le monde"]
    embeddings = await embedder.encode(texts)
    print(f"   Encoded {len(texts)} texts")
    print(f"   Embedding dimension: {len(embeddings[0])}")
    print(f"   Using fallback: {embedder.is_using_fallback()}")

    print("\n" + "="*70)
    print("✓ Fallback test completed successfully")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_fallback())
