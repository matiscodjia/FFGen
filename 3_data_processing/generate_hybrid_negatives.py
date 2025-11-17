#!/usr/bin/env python3
"""
Hybrid Negative Generation - Combine Hard Negatives Mining + Random Sampling

This script generates a mix of hard and easy negatives:
- M hard negatives: Selected based on similarity range (e.g., 0.2-0.4) to positive feedback
- (N-M) easy negatives: Randomly sampled from other examples

This provides both challenging examples (hard) and diversity (random).

Usage:
    python utils/generate_hybrid_negatives.py \
        --input data/input.jsonl \
        --output data/output_hybrid.jsonl \
        --total-negatives 5 \
        --num-hard 2 \
        --min-similarity 0.2 \
        --max-similarity 0.4 \
        --embedding-model google/embeddinggemma-300m
"""

import json
import random
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import argparse


class HybridNegativeGenerator:
    """Generate a hybrid mix of hard and easy negatives."""

    def __init__(
        self,
        embedding_model: str = "google/embeddinggemma-300m",
        batch_size: int = 32
    ):
        """
        Initialize the generator.

        Args:
            embedding_model: Model to use for encoding (for hard negatives)
            batch_size: Batch size for encoding
        """
        self.embedding_model_name = embedding_model
        self.batch_size = batch_size
        self.model = None
        self.device = None

    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            print(f"\n Loading embedding model: {self.embedding_model_name}")
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

            self.model = SentenceTransformer(self.embedding_model_name, device=self.device)
            print(f"   Device: {self.device}")

    def generate_hybrid_negatives(
        self,
        examples: List[Dict],
        total_negatives: int = 5,
        num_hard: int = 2,
        min_similarity: float = 0.2,
        max_similarity: float = 0.4
    ) -> List[Dict]:
        """
        Generate hybrid negatives for all examples.

        Args:
            examples: List of data examples with 'conceptual_feedback'
            total_negatives: Total number of negatives per example (N)
            num_hard: Number of hard negatives among N (M)
            min_similarity: Minimum similarity for hard negatives
            max_similarity: Maximum similarity for hard negatives

        Returns:
            List of examples with 'negative_feedbacks' field containing mixed negatives
        """
        num_easy = total_negatives - num_hard

        print("\n" + "="*80)
        print(" HYBRID NEGATIVE GENERATION")
        print("="*80)
        print(f"Total negatives per example:  {total_negatives}")
        print(f"  - Hard negatives:           {num_hard} (similarity range: {min_similarity:.2f}-{max_similarity:.2f})")
        print(f"  - Easy negatives (random):  {num_easy}")
        print("="*80)

        # Extract all positive feedbacks
        print("\n Extracting positive feedbacks...")
        all_positive_feedbacks = [ex['conceptual_feedback'] for ex in examples]

        # Encode all positive feedbacks (only if hard negatives requested)
        positive_embeddings = None
        if num_hard > 0:
            self.load_model()
            print(f"\n Encoding {len(all_positive_feedbacks)} positive feedbacks...")
            positive_embeddings = self.model.encode(
                all_positive_feedbacks,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            print(f"   Embeddings shape: {positive_embeddings.shape}")

            # Check for NaN values
            nan_mask = np.isnan(positive_embeddings)
            if nan_mask.any():
                num_nan_rows = nan_mask.any(axis=1).sum()
                print(f"   Warning: Found NaN values in {num_nan_rows} embeddings")
                print(f"   Replacing NaN values with zeros...")
                positive_embeddings = np.nan_to_num(positive_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
                print(f"   ✓ NaN values replaced")

        # Statistics
        stats = {
            'total': 0,
            'hard_found_in_range': 0,
            'hard_fallback': 0,
            'hard_similarities': [],
            'easy_negatives': 0
        }

        # Generate negatives for each example
        print(f"\n Generating hybrid negatives...")
        for i, example in enumerate(tqdm(examples, desc="Processing")):
            negatives = []

            # 1. Generate hard negatives (if requested)
            if num_hard > 0 and positive_embeddings is not None:
                hard_negs = self._mine_hard_negatives(
                    i,
                    examples,
                    positive_embeddings,
                    num_hard,
                    min_similarity,
                    max_similarity,
                    stats
                )
                negatives.extend(hard_negs)

            # 2. Generate easy negatives (random)
            if num_easy > 0:
                easy_negs = self._sample_random_negatives(
                    i,
                    all_positive_feedbacks,
                    num_easy,
                    exclude_indices=[j for j in range(i, min(i + num_hard, len(examples)))]
                )
                negatives.extend(easy_negs)
                stats['easy_negatives'] += len(easy_negs)

            # Add negatives array to example
            example['negative_feedbacks'] = negatives

            # Add metadata
            example['negatives_metadata'] = {
                'total': len(negatives),
                'num_hard': min(num_hard, len([n for n in negatives if isinstance(n, dict) and 'similarity' in n])),
                'num_easy': len(negatives) - min(num_hard, len([n for n in negatives if isinstance(n, dict) and 'similarity' in n])),
                'generation_method': 'hybrid'
            }

            # Clean up old fields
            self._cleanup_old_fields(example)

            stats['total'] += 1

        # Print statistics
        self._print_statistics(stats, num_hard, num_easy)

        return examples

    def _mine_hard_negatives(
        self,
        current_idx: int,
        examples: List[Dict],
        positive_embeddings: np.ndarray,
        num_hard: int,
        min_similarity: float,
        max_similarity: float,
        stats: Dict
    ) -> List[str]:
        """
        Mine hard negatives for a single example.

        Returns:
            List of hard negative feedbacks
        """
        hard_negatives = []

        # Current positive embedding
        current_pos_emb = positive_embeddings[current_idx:current_idx+1]

        # Compute similarities to all other positive feedbacks
        similarities = cosine_similarity(current_pos_emb, positive_embeddings)[0]

        # Exclude self
        similarities[current_idx] = -1.0

        # Find candidates in the range
        in_range_mask = (similarities >= min_similarity) & (similarities <= max_similarity)
        in_range_indices = np.where(in_range_mask)[0]

        # Try to get num_hard negatives from the range
        if len(in_range_indices) >= num_hard:
            # Enough candidates in range
            selected_indices = np.random.choice(in_range_indices, size=num_hard, replace=False)
            for idx in selected_indices:
                hard_negatives.append(examples[idx]['conceptual_feedback'])
                stats['hard_similarities'].append(similarities[idx])
            stats['hard_found_in_range'] += num_hard

        elif len(in_range_indices) > 0:
            # Some candidates in range, but not enough
            # Take all in range
            for idx in in_range_indices:
                hard_negatives.append(examples[idx]['conceptual_feedback'])
                stats['hard_similarities'].append(similarities[idx])
            stats['hard_found_in_range'] += len(in_range_indices)

            # Fill remaining with least similar (most dissimilar)
            remaining = num_hard - len(in_range_indices)
            valid_mask = (similarities >= 0) & ~in_range_mask

            if valid_mask.sum() >= remaining:
                # Get indices of least similar
                valid_indices = np.where(valid_mask)[0]
                valid_sims = similarities[valid_indices]
                least_similar_local_indices = np.argsort(valid_sims)[:remaining]
                least_similar_indices = valid_indices[least_similar_local_indices]

                for idx in least_similar_indices:
                    hard_negatives.append(examples[idx]['conceptual_feedback'])
                    stats['hard_similarities'].append(similarities[idx])
                stats['hard_fallback'] += remaining
        else:
            # No candidates in range at all, fallback to least similar
            valid_mask = similarities >= 0
            if valid_mask.any():
                valid_indices = np.where(valid_mask)[0]
                valid_sims = similarities[valid_indices]

                # Get num_hard least similar
                num_to_select = min(num_hard, len(valid_indices))
                least_similar_local_indices = np.argsort(valid_sims)[:num_to_select]
                least_similar_indices = valid_indices[least_similar_local_indices]

                for idx in least_similar_indices:
                    hard_negatives.append(examples[idx]['conceptual_feedback'])
                    stats['hard_similarities'].append(similarities[idx])
                stats['hard_fallback'] += num_to_select

        return hard_negatives

    def _sample_random_negatives(
        self,
        current_idx: int,
        all_feedbacks: List[str],
        num_easy: int,
        exclude_indices: List[int] = None
    ) -> List[str]:
        """
        Sample random (easy) negatives.

        Returns:
            List of random negative feedbacks
        """
        if exclude_indices is None:
            exclude_indices = []

        # Available indices (exclude current and any already selected)
        available_indices = [
            j for j in range(len(all_feedbacks))
            if j != current_idx and j not in exclude_indices
        ]

        # Sample randomly
        num_to_sample = min(num_easy, len(available_indices))
        selected_indices = random.sample(available_indices, num_to_sample)

        return [all_feedbacks[idx] for idx in selected_indices]

    def _cleanup_old_fields(self, example: Dict):
        """Remove old negative fields."""
        old_fields = [
            'hard_negative_feedback',
            'hard_negative_similarity',
            'hard_negative_source_id',
            'random_negative'
        ]
        for field in old_fields:
            if field in example:
                del example[field]

    def _print_statistics(self, stats: Dict, num_hard: int, num_easy: int):
        """Print generation statistics."""
        print("\n" + "="*80)
        print(" GENERATION STATISTICS")
        print("="*80)
        print(f"Total examples processed:  {stats['total']}")

        if num_hard > 0:
            print(f"\nHard Negatives:")
            print(f"  Found in range:          {stats['hard_found_in_range']} ({stats['hard_found_in_range']/(stats['total']*num_hard)*100:.1f}%)")
            print(f"  Fallback (out of range): {stats['hard_fallback']}")

            if stats['hard_similarities']:
                similarities = np.array(stats['hard_similarities'])
                print(f"\n  Similarity statistics (positive vs hard negative):")
                print(f"    Mean:   {similarities.mean():.4f}")
                print(f"    Median: {np.median(similarities):.4f}")
                print(f"    Min:    {similarities.min():.4f}")
                print(f"    Max:    {similarities.max():.4f}")
                print(f"    Std:    {similarities.std():.4f}")

        if num_easy > 0:
            print(f"\nEasy Negatives (Random):")
            print(f"  Generated: {stats['easy_negatives']}")

        print("\n" + "="*80)


def load_dataset(input_path: str) -> List[Dict]:
    """Load JSONL dataset."""
    print(f"\n Loading dataset from: {input_path}")
    with open(input_path, 'r') as f:
        examples = [json.loads(line) for line in f]
    print(f"   Loaded {len(examples)} examples")
    return examples


def save_dataset(examples: List[Dict], output_path: str):
    """Save dataset to JSONL."""
    print(f"\n Saving dataset to: {output_path}")
    with open(output_path, 'w') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    print(f"   Saved {len(examples)} examples")


def main():
    parser = argparse.ArgumentParser(
        description='Generate hybrid negatives (hard + easy/random)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 5 negatives total: 2 hard + 3 random
  python utils/generate_hybrid_negatives.py \\
      --input data/input.jsonl \\
      --output data/output_hybrid.jsonl \\
      --total-negatives 5 \\
      --num-hard 2 \\
      --min-similarity 0.2 \\
      --max-similarity 0.4

  # 10 negatives: 5 hard + 5 random
  python utils/generate_hybrid_negatives.py \\
      --input data/input.jsonl \\
      --output data/output_hybrid.jsonl \\
      --total-negatives 10 \\
      --num-hard 5

  # Only random negatives (num-hard=0)
  python utils/generate_hybrid_negatives.py \\
      --input data/input.jsonl \\
      --output data/output_random.jsonl \\
      --total-negatives 5 \\
      --num-hard 0

  # Only hard negatives (num-hard=total)
  python utils/generate_hybrid_negatives.py \\
      --input data/input.jsonl \\
      --output data/output_hard.jsonl \\
      --total-negatives 5 \\
      --num-hard 5
        """
    )

    # Required arguments
    parser.add_argument('--input', required=True, help='Input JSONL file with conceptual_feedback')
    parser.add_argument('--output', required=True, help='Output JSONL file with negative_feedbacks')

    # Negatives configuration
    parser.add_argument(
        '--total-negatives',
        type=int,
        default=5,
        help='Total number of negatives per example (N) (default: 5)'
    )
    parser.add_argument(
        '--num-hard',
        type=int,
        default=2,
        help='Number of hard negatives among N (M) (default: 2)'
    )

    # Hard negatives parameters
    parser.add_argument(
        '--min-similarity',
        type=float,
        default=0.2,
        help='Minimum similarity for hard negatives (default: 0.2)'
    )
    parser.add_argument(
        '--max-similarity',
        type=float,
        default=0.4,
        help='Maximum similarity for hard negatives (default: 0.4)'
    )
    parser.add_argument(
        '--embedding-model',
        type=str,
        default='google/embeddinggemma-300m',
        help='Embedding model for hard negative mining (default: google/embeddinggemma-300m)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.num_hard > args.total_negatives:
        parser.error(f"--num-hard ({args.num_hard}) cannot be greater than --total-negatives ({args.total_negatives})")

    if args.num_hard < 0:
        parser.error(f"--num-hard must be >= 0")

    if args.min_similarity < 0 or args.max_similarity > 1:
        parser.error(f"Similarity values must be between 0 and 1")

    if args.min_similarity >= args.max_similarity:
        parser.error(f"--min-similarity must be < --max-similarity")

    # Print configuration
    print("="*80)
    print(" HYBRID NEGATIVE GENERATION")
    print("="*80)
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"\nConfiguration:")
    print(f"  Total negatives per example: {args.total_negatives}")
    print(f"  Hard negatives:              {args.num_hard}")
    print(f"  Easy negatives (random):     {args.total_negatives - args.num_hard}")
    if args.num_hard > 0:
        print(f"\nHard Negative Parameters:")
        print(f"  Similarity range:  {args.min_similarity:.2f} - {args.max_similarity:.2f}")
        print(f"  Embedding model:   {args.embedding_model}")
        print(f"  Batch size:        {args.batch_size}")
    print("="*80)

    # Load dataset
    examples = load_dataset(args.input)

    # Generate hybrid negatives
    generator = HybridNegativeGenerator(
        embedding_model=args.embedding_model,
        batch_size=args.batch_size
    )

    examples_with_negatives = generator.generate_hybrid_negatives(
        examples,
        total_negatives=args.total_negatives,
        num_hard=args.num_hard,
        min_similarity=args.min_similarity,
        max_similarity=args.max_similarity
    )

    # Save dataset
    save_dataset(examples_with_negatives, args.output)

    # Show example
    print("\n" + "="*80)
    print(" EXAMPLE OUTPUT")
    print("="*80)
    sample = examples_with_negatives[0]
    print(f"Code: {sample['code_snippet'][:100]}...")
    print(f"\nPositive feedback: {sample['conceptual_feedback'][:100]}...")
    print(f"\nNegative feedbacks ({len(sample['negative_feedbacks'])}):")
    for i, neg in enumerate(sample['negative_feedbacks'][:3], 1):
        print(f"  {i}. {neg[:80]}...")

    print("\n" + "="*80)
    print(" DONE!")
    print("="*80)
    print(f"\nNext step: Train with:")
    print(f"  python 3_model_training/train_embedding.py --data {args.output}")
    print("="*80)


if __name__ == "__main__":
    main()
