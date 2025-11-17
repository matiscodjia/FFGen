#!/usr/bin/env python3
"""
Test config-based initialization of InferenceServer
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.inference_service import InferenceServer

async def test_config_loading():
    print("="*70)
    print("Testing Config-based InferenceServer Initialization")
    print("="*70)

    # Test 1: Load LLM config
    print("\n1. Testing LLM config loading...")
    try:
        llm = InferenceServer.from_config(
            config_path="./configs/config.yml",
            service_type="llm"
        )
        print(f"   ✓ LLM loaded successfully")
        print(f"   URL: {llm.url}")
        print(f"   Model: {llm.model_name}")
        print(f"   Fallback: {llm.fallback_model}")
        print(f"   Using fallback: {llm.is_using_fallback()}")
    except Exception as e:
        print(f"   ✗ LLM loading failed: {e}")

    # Test 2: Load embeddings config
    print("\n2. Testing Embeddings config loading...")
    try:
        embedder = InferenceServer.from_config(
            config_path="./configs/config.yml",
            service_type="embeddings"
        )
        print(f"   ✓ Embedder loaded successfully")
        print(f"   URL: {embedder.url}")
        print(f"   Model: {embedder.model_name}")
        print(f"   Fallback: {embedder.fallback_model}")
        print(f"   Using fallback: {embedder.is_using_fallback()}")

        # Test encoding
        print("\n3. Testing embeddings encoding...")
        texts = ["Test code snippet", "Test feedback"]
        embeddings = await embedder.encode(texts)
        print(f"   ✓ Encoded {len(texts)} texts")
        print(f"   Embedding dimension: {len(embeddings[0])}")
    except Exception as e:
        print(f"   ✗ Embeddings loading/encoding failed: {e}")

    print("\n" + "="*70)
    print("✓ Config loading test completed")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(test_config_loading())
