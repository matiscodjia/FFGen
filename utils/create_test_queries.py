#!/usr/bin/env python3
"""
Create test queries for RAG evaluation from FFGen dataset.

This script extracts code snippets as queries and marks their corresponding
conceptual feedback as the relevant document to retrieve.

Usage:
    python utils/create_test_queries.py \
        --dataset data/Exp-002-llama3B_v2.jsonl \
        --output data/test_queries.jsonl \
        --num-queries 100
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict


def create_queries_from_dataset(
    dataset_path: str,
    num_queries: int = 100,
    seed: int = 42
) -> List[Dict]:
    """
    Create evaluation queries from FFGen dataset.

    For each selected example:
    - Query = code snippet (what user would search with)
    - Relevant doc = the conceptual_feedback (what should be retrieved)

    Args:
        dataset_path: Path to JSONL dataset
        num_queries: Number of test queries to generate
        seed: Random seed for reproducibility

    Returns:
        List of query dicts with format:
            {"query": str, "relevant_ids": List[str]}
    """
    random.seed(seed)

    # Load dataset
    print(f"Loading dataset from {dataset_path}")
    with open(dataset_path, 'r', encoding='utf-8') as f:
        examples = [json.loads(line) for line in f]

    print(f"Loaded {len(examples)} examples")

    # Sample queries
    if num_queries > len(examples):
        print(f"Warning: Requested {num_queries} queries but only {len(examples)} available")
        num_queries = len(examples)

    selected = random.sample(examples, num_queries)
    print(f"Selected {len(selected)} examples as test queries")

    # Create query objects
    queries = []
    for item in selected:
        # Use code as query (simulates: user has code, wants to find feedback)
        query = item.get('code_snippet', '')

        # The relevant document is the one with this code's ID
        # (In real RAG, we'd retrieve the conceptual_feedback for this code)
        relevant_id = item.get('code_id', str(len(queries)))

        queries.append({
            "query": query,
            "relevant_ids": [relevant_id],
            "metadata": {
                "code_id": relevant_id,
                "expected_feedback": item.get('conceptual_feedback', '')[:100] + "..."
            }
        })

    return queries


def main():
    parser = argparse.ArgumentParser(
        description="Create test queries for RAG evaluation"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to FFGen JSONL dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/test_queries.jsonl",
        help="Output path for queries file"
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=100,
        help="Number of test queries to generate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Create queries
    queries = create_queries_from_dataset(
        dataset_path=args.dataset,
        num_queries=args.num_queries,
        seed=args.seed
    )

    # Save queries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')

    print(f"\n✓ Saved {len(queries)} queries to {output_path}")

    # Show example
    print("\n" + "="*80)
    print("EXAMPLE QUERY")
    print("="*80)
    example = queries[0]
    print(f"Query (code):\n{example['query'][:200]}...")
    print(f"\nRelevant IDs: {example['relevant_ids']}")
    print(f"Expected feedback: {example['metadata']['expected_feedback']}")
    print("="*80)

    print(f"\nNext step: Run RAG evaluation with:")
    print(f"  python utils/rag_evaluation.py \\")
    print(f"    --model models/YOUR_MODEL \\")
    print(f"    --corpus {args.dataset} \\")
    print(f"    --queries {output_path} \\")
    print(f"    --output results/rag_metrics.json")


if __name__ == "__main__":
    main()
