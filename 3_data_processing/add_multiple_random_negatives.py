#!/usr/bin/env python3
"""
Generate multiple random negatives per code snippet for better training diversity.

This script generates N random negatives (default=5) for each code snippet,
providing more diverse training signal for triplet loss.
"""

import json
import random
import sys
from pathlib import Path
from tqdm import tqdm

def load_dataset(input_path):
    """Load JSONL dataset"""
    with open(input_path, 'r') as f:
        return [json.loads(line) for line in f]

def generate_multiple_negatives(examples, num_negatives=5):
    """
    Generate N random negatives for each example.

    Args:
        examples: List of data examples
        num_negatives: Number of negative samples per example

    Returns:
        List of examples with 'negative_feedbacks' field (array)
    """
    print(f"\nGenerating {num_negatives} random negatives per example...")

    # Extract all positive feedbacks
    all_positives = [ex['conceptual_feedback'] for ex in examples]

    for i, example in enumerate(tqdm(examples, desc="Processing")):
        positive = example['conceptual_feedback']

        # Sample N random negatives (different from positive)
        negatives = []
        available_indices = [j for j in range(len(examples)) if j != i]

        # Ensure we don't pick the same negative twice
        selected_indices = random.sample(available_indices, min(num_negatives, len(available_indices)))

        for idx in selected_indices:
            negative = all_positives[idx]
            negatives.append(negative)

        # Add negatives array to example
        example['negative_feedbacks'] = negatives

        # Remove old single negative field if exists
        if 'hard_negative_feedback' in example:
            del example['hard_negative_feedback']
        if 'hard_negative_similarity' in example:
            del example['hard_negative_similarity']
        if 'hard_negative_source_id' in example:
            del example['hard_negative_source_id']
        if 'random_negative' in example:
            del example['random_negative']

    return examples

def save_dataset(examples, output_path):
    """Save dataset to JSONL"""
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')

    print(f"Saved {len(examples)} examples to {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Generate multiple random negatives per code')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--num-negatives', type=int, default=5, help='Number of negatives per example (default: 5)')
    args = parser.parse_args()

    print("="*80)
    print(" MULTIPLE RANDOM NEGATIVES GENERATOR")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Negatives per example: {args.num_negatives}")

    # Load dataset
    examples = load_dataset(args.input)
    print(f"\nLoaded {len(examples)} examples")

    # Generate negatives
    examples_with_negatives = generate_multiple_negatives(examples, args.num_negatives)

    # Save dataset
    save_dataset(examples_with_negatives, args.output)

    # Statistics
    print("\n" + "="*80)
    print(" DATASET STATISTICS")
    print("="*80)
    sample = examples_with_negatives[0]
    print(f"Code snippet: {sample['code_snippet'][:100]}...")
    print(f"Positive feedback: {sample['conceptual_feedback'][:80]}...")
    print(f"Number of negatives: {len(sample['negative_feedbacks'])}")
    print(f"Negative 1: {sample['negative_feedbacks'][0][:80]}...")
    print(f"Negative 2: {sample['negative_feedbacks'][1][:80]}...")

    print("\n Done! Dataset ready for training with multiple negatives.")
    print(f"\nNext step: Train with:")
    print(f"  python 3_model_training/train_embedding.py --data {args.output}")
    print("="*80)

if __name__ == "__main__":
    main()
