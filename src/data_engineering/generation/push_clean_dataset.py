#!/usr/bin/env python3
"""
Push cleaned dataset to HuggingFace Hub
"""

from datasets import load_dataset
from huggingface_hub import HfApi
import os

def main():
    print("="*70)
    print("PUSHING CLEANED DATASET TO HUGGING FACE HUB")
    print("="*70)

    # Load the ultra clean dataset
    dataset = load_dataset('json', data_files={
        'train': 'data/ultra_clean_final_dataset/train.jsonl',
        'validation': 'data/ultra_clean_final_dataset/validation.jsonl',
        'test': 'data/ultra_clean_final_dataset/test.jsonl'
    })

    print(f"\nDataset loaded:")
    print(f"  Train:      {len(dataset['train']):,} samples")
    print(f"  Validation: {len(dataset['validation']):,} samples")
    print(f"  Test:       {len(dataset['test']):,} samples")

    # Push to hub
    repo_id = "matis35/RAFT_CLEAN_V1"

    print(f"\nPushing to {repo_id}...")
    dataset.push_to_hub(repo_id, private=False)

    print(f"\n Dataset pushed successfully!")
    print(f"  URL: https://huggingface.co/datasets/{repo_id}")
    print("="*70)

if __name__ == "__main__":
    main()
