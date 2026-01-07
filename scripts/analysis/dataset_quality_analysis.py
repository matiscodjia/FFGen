"""
Dataset Quality Analysis - Semantic Entropy & Quality Metrics
==============================================================

This script quantifies dataset quality through multiple dimensions:
1. Semantic Diversity (entropy of code/feedback embeddings)
2. Semantic Redundancy (duplicate detection)
3. Code-Feedback Alignment (semantic coherence)
4. Lexical Complexity (vocabulary richness)

Usage:
    python dataset_quality_analysis.py --datasets matis35/RAFT matis35/RAFT_CLEAN_V1
"""

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine, pdist, squareform
from scipy.stats import entropy
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import json
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer

sns.set_style("whitegrid")


class DatasetQualityAnalyzer:
    """Comprehensive dataset quality analysis"""

    def __init__(self, model_name: str = "Salesforce/SFR-Embedding-Code-400M_R"):
        """
        Initialize analyzer with embedding model

        Args:
            model_name: HuggingFace model for semantic embeddings
        """
        print(f"Loading embedding model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding

        Returns:
            Embedding matrix (N, D)
        """
        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
                batch = texts[i:i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**encoded)
                batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def compute_semantic_entropy(self, embeddings: np.ndarray, n_clusters: int = 50) -> float:
        """
        Compute semantic entropy using clustering

        Higher entropy = more diverse dataset

        Args:
            embeddings: Embedding matrix (N, D)
            n_clusters: Number of clusters for discretization

        Returns:
            Entropy value
        """
        # Cluster embeddings
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Compute distribution
        counts = np.bincount(labels, minlength=n_clusters)
        probs = counts / counts.sum()

        # Shannon entropy
        return entropy(probs, base=2)

    def compute_redundancy_score(self, embeddings: np.ndarray, threshold: float = 0.9) -> Dict:
        """
        Compute semantic redundancy (near-duplicates)

        Args:
            embeddings: Embedding matrix (N, D)
            threshold: Cosine similarity threshold for duplicates

        Returns:
            Dict with redundancy metrics
        """
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / (norms + 1e-9)

        # Compute pairwise cosine similarities
        similarities = np.dot(normalized, normalized.T)

        # Count near-duplicates (excluding diagonal)
        np.fill_diagonal(similarities, 0)
        duplicate_pairs = (similarities > threshold).sum() / 2  # Divide by 2 for symmetry

        total_pairs = len(embeddings) * (len(embeddings) - 1) / 2
        redundancy_ratio = duplicate_pairs / total_pairs if total_pairs > 0 else 0

        # Average similarity (excluding self)
        avg_similarity = (similarities.sum() - len(embeddings)) / (len(embeddings) * (len(embeddings) - 1))

        return {
            "duplicate_pairs": int(duplicate_pairs),
            "total_pairs": int(total_pairs),
            "redundancy_ratio": float(redundancy_ratio),
            "avg_pairwise_similarity": float(avg_similarity),
            "max_similarity": float(similarities.max())
        }

    def compute_code_feedback_alignment(
        self,
        code_embeddings: np.ndarray,
        feedback_embeddings: np.ndarray
    ) -> Dict:
        """
        Measure semantic alignment between code and feedback

        Args:
            code_embeddings: Code embedding matrix
            feedback_embeddings: Feedback embedding matrix

        Returns:
            Alignment metrics
        """
        # Normalize
        code_norm = code_embeddings / (np.linalg.norm(code_embeddings, axis=1, keepdims=True) + 1e-9)
        feedback_norm = feedback_embeddings / (np.linalg.norm(feedback_embeddings, axis=1, keepdims=True) + 1e-9)

        # Compute paired similarities
        paired_similarities = (code_norm * feedback_norm).sum(axis=1)

        # Compute cross-similarities (code[i] with all feedbacks)
        cross_similarities = np.dot(code_norm, feedback_norm.T)

        # Ranking: where does the correct feedback rank for each code?
        ranks = []
        for i in range(len(code_embeddings)):
            sorted_indices = np.argsort(-cross_similarities[i])
            rank = np.where(sorted_indices == i)[0][0] + 1
            ranks.append(rank)

        return {
            "mean_paired_similarity": float(paired_similarities.mean()),
            "std_paired_similarity": float(paired_similarities.std()),
            "median_rank": float(np.median(ranks)),
            "mean_reciprocal_rank": float(np.mean(1.0 / np.array(ranks))),
            "recall_at_1": float((np.array(ranks) == 1).mean()),
            "recall_at_5": float((np.array(ranks) <= 5).mean()),
            "recall_at_10": float((np.array(ranks) <= 10).mean())
        }

    def compute_lexical_complexity(self, texts: List[str]) -> Dict:
        """
        Compute lexical complexity metrics

        Args:
            texts: List of text strings

        Returns:
            Lexical metrics
        """
        # Tokenize
        all_tokens = []
        lengths = []

        for text in texts:
            tokens = text.split()
            all_tokens.extend(tokens)
            lengths.append(len(tokens))

        # Vocabulary size
        vocab = set(all_tokens)

        # Type-Token Ratio (vocabulary richness)
        ttr = len(vocab) / len(all_tokens) if all_tokens else 0

        # Average length
        avg_length = np.mean(lengths)
        std_length = np.std(lengths)

        return {
            "vocabulary_size": len(vocab),
            "total_tokens": len(all_tokens),
            "type_token_ratio": float(ttr),
            "avg_length": float(avg_length),
            "std_length": float(std_length),
            "min_length": int(min(lengths)) if lengths else 0,
            "max_length": int(max(lengths)) if lengths else 0
        }

    def analyze_dataset(self, dataset_name: str, split: str = "train") -> Dict:
        """
        Complete quality analysis of a dataset

        Args:
            dataset_name: HuggingFace dataset name
            split: Dataset split to analyze

        Returns:
            Complete quality metrics
        """
        print(f"\n{'='*80}")
        print(f"Analyzing: {dataset_name} ({split})")
        print(f"{'='*80}")

        # Load dataset
        dataset = load_dataset(dataset_name, split=split)

        codes = [item["code"] for item in dataset]
        feedbacks = [item["feedback"] for item in dataset]

        print(f"Dataset size: {len(codes)} examples")

        # 1. Generate embeddings
        print("\n1. Generating embeddings...")
        code_embeddings = self.embed_texts(codes)
        feedback_embeddings = self.embed_texts(feedbacks)

        # 2. Semantic entropy
        print("\n2. Computing semantic entropy...")
        code_entropy = self.compute_semantic_entropy(code_embeddings)
        feedback_entropy = self.compute_semantic_entropy(feedback_embeddings)

        # 3. Redundancy
        print("\n3. Computing redundancy...")
        code_redundancy = self.compute_redundancy_score(code_embeddings)
        feedback_redundancy = self.compute_redundancy_score(feedback_embeddings)

        # 4. Code-Feedback alignment
        print("\n4. Computing code-feedback alignment...")
        alignment = self.compute_code_feedback_alignment(code_embeddings, feedback_embeddings)

        # 5. Lexical complexity
        print("\n5. Computing lexical complexity...")
        code_lexical = self.compute_lexical_complexity(codes)
        feedback_lexical = self.compute_lexical_complexity(feedbacks)

        # Aggregate results
        results = {
            "dataset_name": dataset_name,
            "split": split,
            "size": len(codes),
            "semantic_entropy": {
                "code": code_entropy,
                "feedback": feedback_entropy,
                "mean": (code_entropy + feedback_entropy) / 2
            },
            "redundancy": {
                "code": code_redundancy,
                "feedback": feedback_redundancy
            },
            "alignment": alignment,
            "lexical_complexity": {
                "code": code_lexical,
                "feedback": feedback_lexical
            }
        }

        return results

    def compare_datasets(self, datasets: List[str], output_dir: str = "dataset_quality_analysis"):
        """
        Compare quality metrics across multiple datasets

        Args:
            datasets: List of dataset names
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        all_results = []

        # Analyze each dataset
        for dataset_name in datasets:
            results = self.analyze_dataset(dataset_name)
            all_results.append(results)

            # Save individual results
            result_file = output_path / f"{dataset_name.replace('/', '_')}_quality.json"
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)

        # Generate comparison report
        self.generate_comparison_report(all_results, output_path)

        # Generate visualizations
        self.plot_comparison(all_results, output_path)

        print(f"\n✅ Analysis complete! Results saved to: {output_path}")

    def generate_comparison_report(self, results: List[Dict], output_dir: Path):
        """Generate text comparison report"""
        report_file = output_dir / "comparison_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATASET QUALITY COMPARISON REPORT\n")
            f.write("="*80 + "\n\n")

            for result in results:
                f.write(f"\nDataset: {result['dataset_name']}\n")
                f.write("-"*80 + "\n")
                f.write(f"Size: {result['size']} examples\n\n")

                f.write("Semantic Entropy (Diversity):\n")
                f.write(f"  Code:     {result['semantic_entropy']['code']:.4f}\n")
                f.write(f"  Feedback: {result['semantic_entropy']['feedback']:.4f}\n")
                f.write(f"  Mean:     {result['semantic_entropy']['mean']:.4f}\n\n")

                f.write("Redundancy (Code):\n")
                f.write(f"  Duplicate pairs: {result['redundancy']['code']['duplicate_pairs']}\n")
                f.write(f"  Redundancy ratio: {result['redundancy']['code']['redundancy_ratio']:.4f}\n")
                f.write(f"  Avg similarity: {result['redundancy']['code']['avg_pairwise_similarity']:.4f}\n\n")

                f.write("Code-Feedback Alignment:\n")
                f.write(f"  Mean paired similarity: {result['alignment']['mean_paired_similarity']:.4f}\n")
                f.write(f"  MRR: {result['alignment']['mean_reciprocal_rank']:.4f}\n")
                f.write(f"  Recall@1: {result['alignment']['recall_at_1']:.4f}\n\n")

                f.write("Lexical Complexity (Code):\n")
                f.write(f"  Vocabulary size: {result['lexical_complexity']['code']['vocabulary_size']}\n")
                f.write(f"  Type-Token Ratio: {result['lexical_complexity']['code']['type_token_ratio']:.4f}\n\n")

        print(f"Report saved: {report_file}")

    def plot_comparison(self, results: List[Dict], output_dir: Path):
        """Generate comparison visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Quality Comparison', fontsize=16, fontweight='bold')

        dataset_names = [r['dataset_name'].split('/')[-1] for r in results]

        # 1. Semantic Entropy
        ax = axes[0, 0]
        code_entropy = [r['semantic_entropy']['code'] for r in results]
        feedback_entropy = [r['semantic_entropy']['feedback'] for r in results]

        x = np.arange(len(dataset_names))
        width = 0.35
        ax.bar(x - width/2, code_entropy, width, label='Code', alpha=0.8)
        ax.bar(x + width/2, feedback_entropy, width, label='Feedback', alpha=0.8)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Entropy (bits)')
        ax.set_title('Semantic Diversity (Higher = More Diverse)')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Redundancy Ratio
        ax = axes[0, 1]
        redundancy = [r['redundancy']['code']['redundancy_ratio'] * 100 for r in results]
        ax.bar(dataset_names, redundancy, alpha=0.8, color='coral')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Redundancy Ratio (%)')
        ax.set_title('Semantic Redundancy (Lower = Less Duplicates)')
        ax.set_xticklabels(dataset_names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3)

        # 3. Code-Feedback Alignment
        ax = axes[1, 0]
        mrr = [r['alignment']['mean_reciprocal_rank'] for r in results]
        recall_at_1 = [r['alignment']['recall_at_1'] for r in results]

        x = np.arange(len(dataset_names))
        width = 0.35
        ax.bar(x - width/2, mrr, width, label='MRR', alpha=0.8)
        ax.bar(x + width/2, recall_at_1, width, label='Recall@1', alpha=0.8)
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Score')
        ax.set_title('Code-Feedback Alignment (Higher = Better)')
        ax.set_xticks(x)
        ax.set_xticklabels(dataset_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Lexical Richness
        ax = axes[1, 1]
        ttr = [r['lexical_complexity']['code']['type_token_ratio'] for r in results]
        ax.bar(dataset_names, ttr, alpha=0.8, color='green')
        ax.set_xlabel('Dataset')
        ax.set_ylabel('Type-Token Ratio')
        ax.set_title('Lexical Richness (Higher = More Diverse Vocabulary)')
        ax.set_xticklabels(dataset_names, rotation=15, ha='right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = output_dir / "quality_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {output_file}")
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Dataset Quality Analysis")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["matis35/RAFT", "matis35/RAFT_CLEAN_V1"],
        help="List of dataset names to analyze"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Salesforce/SFR-Embedding-Code-400M_R",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_quality_analysis",
        help="Output directory"
    )

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DatasetQualityAnalyzer(model_name=args.model)

    # Compare datasets
    analyzer.compare_datasets(args.datasets, args.output_dir)


if __name__ == "__main__":
    main()
