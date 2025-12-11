#!/usr/bin/env python3
"""
clean_dataset_pipeline.py - Complete pipeline for cleaning code-feedback datasets

This script performs the full cleaning process:
1. Exact deduplication (removes identical feedbacks)
2. Semantic deduplication (removes similar feedbacks via Jaccard similarity)
3. Creates stratified train/val/test splits
4. Generates statistics and quality reports

Usage:
    python scripts/clean_dataset_pipeline.py \
        --input heavy_data/raw_dataset.jsonl \
        --output data/cleaned_dataset \
        --threshold 0.40 \
        --push-to-hub matis35/RAFT_CLEAN_V2

Author: FFGen Team
"""

import json
import argparse
import hashlib
import random
from pathlib import Path
from typing import List, Dict, Set
from collections import defaultdict


class DatasetCleaner:
    """Complete dataset cleaning pipeline."""

    def __init__(self, semantic_threshold: float = 0.40, seed: int = 42):
        self.semantic_threshold = semantic_threshold
        self.seed = seed
        random.seed(seed)

        self.stats = {
            'original_size': 0,
            'after_exact_dedup': 0,
            'after_semantic_dedup': 0,
            'final_train': 0,
            'final_val': 0,
            'final_test': 0
        }

    # ==========================================
    # STEP 1: EXACT DEDUPLICATION
    # ==========================================

    def exact_deduplicate(self, entries: List[Dict]) -> List[Dict]:
        """Remove entries with identical feedbacks."""
        print("\n" + "="*70)
        print("STEP 1: EXACT DEDUPLICATION")
        print("="*70)

        self.stats['original_size'] = len(entries)
        print(f"Input: {len(entries):,} entries")

        # Group by feedback
        feedback_groups = defaultdict(list)
        for entry in entries:
            feedback = entry.get('generated_feedback', entry.get('feedback', ''))
            if feedback:
                feedback_groups[feedback].append(entry)

        print(f"Found {len(feedback_groups):,} unique feedbacks")

        # Select best representative for each group
        deduplicated = []
        removed_count = 0

        for feedback, group in feedback_groups.items():
            if len(group) > 1:
                removed_count += len(group) - 1

            # Select best (by code complexity)
            best = self._select_best_code(group)
            deduplicated.append(best)

        self.stats['after_exact_dedup'] = len(deduplicated)

        print(f"Output: {len(deduplicated):,} entries")
        print(f"Removed: {removed_count:,} duplicates ({removed_count/len(entries)*100:.1f}%)")

        return deduplicated

    def _select_best_code(self, codes: List[Dict]) -> Dict:
        """Select code with medium complexity."""
        if len(codes) == 1:
            return codes[0]

        # Score by complexity
        scored = []
        for entry in codes:
            code = entry.get('code_snippet', entry.get('code', ''))
            complexity = self._compute_complexity(code)
            scored.append((entry, complexity))

        # Sort by complexity, take middle
        scored.sort(key=lambda x: x[1])
        return scored[len(scored) // 2][0]

    def _compute_complexity(self, code: str) -> int:
        """Simple code complexity metric."""
        keywords = ['if', 'while', 'for', 'switch', 'case', 'else']
        complexity = sum(code.lower().count(kw) for kw in keywords)
        complexity += code.count('(')
        complexity += len(code) // 100
        return complexity

    # ==========================================
    # STEP 2: SEMANTIC DEDUPLICATION
    # ==========================================

    def semantic_deduplicate(self, entries: List[Dict]) -> List[Dict]:
        """Remove semantically similar feedbacks."""
        print("\n" + "="*70)
        print("STEP 2: SEMANTIC DEDUPLICATION")
        print("="*70)
        print(f"Threshold: {self.semantic_threshold}")
        print(f"Input: {len(entries):,} entries")

        # Build similarity graph
        graph = self._build_similarity_graph(entries)

        # Find clusters (connected components)
        clusters = self._find_connected_components(graph, len(entries))
        print(f"Found {len(clusters):,} similarity clusters")

        # Select representatives
        indices_to_keep = set(range(len(entries)))

        for cluster in clusters:
            if len(cluster) > 1:
                representative = self._select_cluster_representative(cluster, entries)
                for idx in cluster:
                    if idx != representative:
                        indices_to_keep.discard(idx)

        deduplicated = [entries[idx] for idx in sorted(indices_to_keep)]

        self.stats['after_semantic_dedup'] = len(deduplicated)
        removed = len(entries) - len(deduplicated)

        print(f"Output: {len(deduplicated):,} entries")
        print(f"Removed: {removed:,} similar ({removed/len(entries)*100:.1f}%)")

        return deduplicated

    def _build_similarity_graph(self, entries: List[Dict]) -> Dict[int, Set[int]]:
        """Build graph of similar feedbacks."""
        feedbacks = [self._get_feedback(e) for e in entries]
        graph = defaultdict(set)

        # Bucket by length for efficiency
        by_length = defaultdict(list)
        for idx, fb in enumerate(feedbacks):
            bucket = len(fb) // 50
            by_length[bucket].append(idx)

        comparisons = 0
        edges = 0

        for bucket_indices in by_length.values():
            if len(bucket_indices) < 2:
                continue

            for i, idx1 in enumerate(bucket_indices):
                for idx2 in bucket_indices[i+1:]:
                    comparisons += 1

                    if feedbacks[idx1] == feedbacks[idx2]:
                        graph[idx1].add(idx2)
                        graph[idx2].add(idx1)
                        edges += 1
                        continue

                    similarity = self._jaccard_similarity(feedbacks[idx1], feedbacks[idx2])

                    if similarity >= self.semantic_threshold:
                        graph[idx1].add(idx2)
                        graph[idx2].add(idx1)
                        edges += 1

        print(f"  Comparisons: {comparisons:,}, Similar pairs: {edges:,}")
        return graph

    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity (word-level)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def _find_connected_components(self, graph: Dict[int, Set[int]], n: int) -> List[Set[int]]:
        """Find connected components (DFS)."""
        visited = set()
        components = []

        def dfs(node, component):
            visited.add(node)
            component.add(node)
            for neighbor in graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for node in range(n):
            if node not in visited:
                component = set()
                dfs(node, component)
                if len(component) > 1:
                    components.append(component)

        return components

    def _select_cluster_representative(self, cluster: Set[int], entries: List[Dict]) -> int:
        """Select best representative from cluster."""
        scored = []
        for idx in cluster:
            entry = entries[idx]
            code = entry.get('code_snippet', entry.get('code', ''))
            complexity = self._compute_complexity(code)
            scored.append((idx, complexity))

        scored.sort(key=lambda x: -x[1])
        return scored[0][0]

    # ==========================================
    # STEP 3: CREATE SPLITS
    # ==========================================

    def create_splits(self, entries: List[Dict],
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.1,
                     test_ratio: float = 0.1) -> tuple:
        """Create stratified train/val/test splits."""
        print("\n" + "="*70)
        print("STEP 3: CREATE SPLITS")
        print("="*70)
        print(f"Ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")

        # Group by feedback pattern (first 3 words)
        groups = defaultdict(list)
        for entry in entries:
            feedback = self._get_feedback(entry)
            words = feedback.lower().split()[:3]
            key = ' '.join(words) if len(words) >= 3 else feedback[:50]
            groups[key].append(entry)

        print(f"Grouped into {len(groups):,} patterns")

        # Shuffle and split groups
        group_list = list(groups.values())
        random.shuffle(group_list)

        train, val, test = [], [], []
        train_target = int(train_ratio * len(entries))
        val_target = int(val_ratio * len(entries))

        train_count, val_count, test_count = 0, 0, 0

        for group in group_list:
            if train_count < train_target:
                train.extend(group)
                train_count += len(group)
            elif val_count < val_target:
                val.extend(group)
                val_count += len(group)
            else:
                test.extend(group)
                test_count += len(group)

        # Shuffle within splits
        random.shuffle(train)
        random.shuffle(val)
        random.shuffle(test)

        self.stats['final_train'] = len(train)
        self.stats['final_val'] = len(val)
        self.stats['final_test'] = len(test)

        print(f"Train: {len(train):,} ({len(train)/len(entries)*100:.1f}%)")
        print(f"Val:   {len(val):,} ({len(val)/len(entries)*100:.1f}%)")
        print(f"Test:  {len(test):,} ({len(test)/len(entries)*100:.1f}%)")

        return train, val, test

    # ==========================================
    # UTILITIES
    # ==========================================

    def _get_feedback(self, entry: Dict) -> str:
        """Extract feedback from entry."""
        return entry.get('generated_feedback', entry.get('feedback', ''))

    def format_for_hf(self, entry: Dict) -> Dict:
        """Format entry for HuggingFace."""
        return {
            "code": entry.get("code_snippet", entry.get("code", "")),
            "feedback": self._get_feedback(entry),
            "code_id": entry.get("code_id", ""),
            "author_id": entry.get("author_id", "")
        }

    def print_final_stats(self):
        """Print complete pipeline statistics."""
        print("\n" + "="*70)
        print("PIPELINE COMPLETE - FINAL STATISTICS")
        print("="*70)
        print(f"Original size:           {self.stats['original_size']:,}")
        print(f"After exact dedup:       {self.stats['after_exact_dedup']:,} "
              f"(-{self.stats['original_size'] - self.stats['after_exact_dedup']:,})")
        print(f"After semantic dedup:    {self.stats['after_semantic_dedup']:,} "
              f"(-{self.stats['after_exact_dedup'] - self.stats['after_semantic_dedup']:,})")
        print()
        print(f"Final train:             {self.stats['final_train']:,}")
        print(f"Final validation:        {self.stats['final_val']:,}")
        print(f"Final test:              {self.stats['final_test']:,}")
        print()
        total_final = self.stats['final_train'] + self.stats['final_val'] + self.stats['final_test']
        total_removed = self.stats['original_size'] - total_final
        print(f"Total removed:           {total_removed:,} "
              f"({total_removed/self.stats['original_size']*100:.1f}%)")
        print("="*70)


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def save_jsonl(entries: List[Dict], file_path: str):
    """Save JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Complete dataset cleaning pipeline")
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    parser.add_argument('--threshold', type=float, default=0.40, help='Semantic similarity threshold (default: 0.40)')
    parser.add_argument('--push-to-hub', type=str, help='Push to HuggingFace Hub (e.g., matis35/RAFT_CLEAN_V2)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    print("="*70)
    print("DATASET CLEANING PIPELINE")
    print("="*70)
    print(f"Input:     {args.input}")
    print(f"Output:    {args.output}")
    print(f"Threshold: {args.threshold}")
    print(f"Seed:      {args.seed}")

    # Load
    print("\nLoading dataset...")
    entries = load_jsonl(args.input)

    # Clean
    cleaner = DatasetCleaner(semantic_threshold=args.threshold, seed=args.seed)

    entries = cleaner.exact_deduplicate(entries)
    entries = cleaner.semantic_deduplicate(entries)
    train, val, test = cleaner.create_splits(entries)

    # Format
    print("\nFormatting for HuggingFace...")
    train = [cleaner.format_for_hf(e) for e in train]
    val = [cleaner.format_for_hf(e) for e in val]
    test = [cleaner.format_for_hf(e) for e in test]

    # Save
    print(f"\nSaving to {args.output}...")
    save_jsonl(train, f"{args.output}/train.jsonl")
    save_jsonl(val, f"{args.output}/validation.jsonl")
    save_jsonl(test, f"{args.output}/test.jsonl")

    # Stats
    cleaner.print_final_stats()

    # Push to hub
    if args.push_to_hub:
        print(f"\nPushing to HuggingFace Hub: {args.push_to_hub}...")
        from datasets import load_dataset
        dataset = load_dataset('json', data_files={
            'train': f"{args.output}/train.jsonl",
            'validation': f"{args.output}/validation.jsonl",
            'test': f"{args.output}/test.jsonl"
        })
        dataset.push_to_hub(args.push_to_hub, private=False)
        print(f"✓ Pushed to https://huggingface.co/datasets/{args.push_to_hub}")

    print("\n✓ Pipeline complete!")


if __name__ == "__main__":
    main()
