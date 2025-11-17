"""
RAG Evaluation Script for FFGen

This script evaluates trained embedding models using a ChromaDB vector store
and computes proper retrieval metrics:
- Recall@k (k=1, 5, 10)
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (NDCG@k)
- Mean Average Precision (MAP)

Usage:
    python utils/rag_evaluation.py \
        --model models/trained_model \
        --corpus data/evaluation_corpus.jsonl \
        --queries data/test_queries.jsonl \
        --output results/rag_metrics.json
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Optional ChromaDB import
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed. Install with: uv pip install chromadb")


class RAGEvaluator:
    """Evaluates retrieval performance of embedding models."""

    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        collection_name: str = "ffgen_eval"
    ):
        """
        Initialize the RAG evaluator.

        Args:
            model_path: Path to the trained model
            device: Device to use (cuda/mps/cpu), auto-detected if None
            collection_name: Name for the ChromaDB collection
        """
        # Device selection
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = device
        print(f"Using device: {device}")

        # Load model
        print(f"Loading model from {model_path}")
        self.model = SentenceTransformer(model_path)
        self.model.to(self.device)

        # Initialize ChromaDB (disabled for performance)
        self.chroma_client = None

        self.corpus = []
        self.corpus_embeddings = None

    def load_corpus(self, corpus_path: str):
        """
        Load corpus from JSONL file.

        Expected format: {"id": str, "text": str, "metadata": dict}
        Or for FFGen: {"code": str, "conceptual_feedback": str, ...}
        """
        print(f"Loading corpus from {corpus_path}")
        self.corpus = []

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                # Support different corpus formats
                if "text" in item:
                    self.corpus.append(item)
                elif "conceptual_feedback" in item:
                    # FFGen format: use conceptual feedback as the text to retrieve
                    self.corpus.append({
                        "id": item.get("code_id", item.get("id", len(self.corpus))),
                        "text": item["conceptual_feedback"],
                        "code": item.get("code_snippet", item.get("code", "")),
                        "metadata": item
                    })
                else:
                    raise ValueError(f"Unsupported corpus format: {item.keys()}")

        print(f"Loaded {len(self.corpus)} documents")

    def index_corpus(self):
        """Index the corpus with embeddings."""
        print("Computing embeddings for corpus...")

        texts = [doc["text"] for doc in self.corpus]

        # Compute embeddings in batches
        self.corpus_embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        # Index in ChromaDB if available
        if CHROMADB_AVAILABLE and self.chroma_client is not None:
            print("Indexing in ChromaDB...")

            # Reset collection if exists
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except:
                pass

            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "FFGen evaluation corpus"}
            )

            # Add documents to collection in batches (ChromaDB has a max batch size)
            batch_size = 5000
            n_docs = len(self.corpus)

            for batch_idx in range(0, n_docs, batch_size):
                end_idx = min(batch_idx + batch_size, n_docs)
                batch_corpus = self.corpus[batch_idx:end_idx]
                batch_embeddings = self.corpus_embeddings[batch_idx:end_idx]
                batch_texts = texts[batch_idx:end_idx]

                self.collection.add(
                    ids=[str(doc.get("id", i)) for i, doc in enumerate(batch_corpus, start=batch_idx)],
                    embeddings=batch_embeddings.tolist(),
                    documents=batch_texts,
                    metadatas=[{"index": i} for i in range(batch_idx, end_idx)]
                )

                print(f"  Added batch {batch_idx//batch_size + 1}: {end_idx - batch_idx} documents")

            print(f"Indexed {len(self.corpus)} documents in ChromaDB")

    def search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for top-k most similar documents.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (doc_index, similarity_score) tuples, sorted by score descending
        """
        # Encode query
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Compute similarities (cosine similarity via dot product with normalized vectors)
        similarities = np.dot(self.corpus_embeddings, query_embedding)

        # Get top-k indices
        top_k_indices = np.argsort(similarities)[::-1][:k]

        # Return (index, score) pairs
        results = [(int(idx), float(similarities[idx])) for idx in top_k_indices]

        return results

    def compute_recall_at_k(
        self,
        queries: List[Dict],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict[int, float]:
        """
        Compute Recall@k: fraction of queries where the correct answer is in top-k.

        Args:
            queries: List of query dicts with "query" and "relevant_ids" keys
            k_values: List of k values to compute recall for

        Returns:
            Dict mapping k -> recall@k
        """
        recall_at_k = {k: 0.0 for k in k_values}

        for query_item in tqdm(queries, desc="Computing Recall@k"):
            query = query_item["query"]
            relevant_ids = set(query_item["relevant_ids"])

            # Get top-k results for max k
            results = self.search(query, k=max(k_values))

            # Check each k value
            for k in k_values:
                retrieved_ids = {self.corpus[idx]["id"] for idx, _ in results[:k]}
                if relevant_ids & retrieved_ids:  # Intersection
                    recall_at_k[k] += 1.0

        # Normalize by number of queries
        n_queries = len(queries)
        recall_at_k = {k: count / n_queries for k, count in recall_at_k.items()}

        return recall_at_k

    def compute_mrr(self, queries: List[Dict]) -> float:
        """
        Compute Mean Reciprocal Rank (MRR).

        MRR = average of (1 / rank_of_first_relevant_item)

        Args:
            queries: List of query dicts with "query" and "relevant_ids" keys

        Returns:
            MRR score
        """
        reciprocal_ranks = []

        for query_item in tqdm(queries, desc="Computing MRR"):
            query = query_item["query"]
            relevant_ids = set(query_item["relevant_ids"])

            # Get top results
            results = self.search(query, k=100)

            # Find rank of first relevant result
            for rank, (idx, _) in enumerate(results, start=1):
                doc_id = self.corpus[idx]["id"]
                if doc_id in relevant_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # No relevant result found
                reciprocal_ranks.append(0.0)

        return np.mean(reciprocal_ranks)

    def compute_ndcg_at_k(
        self,
        queries: List[Dict],
        k_values: List[int] = [5, 10]
    ) -> Dict[int, float]:
        """
        Compute Normalized Discounted Cumulative Gain at k (NDCG@k).

        DCG = sum(rel_i / log2(i+1)) for i in 1..k
        NDCG = DCG / IDCG (ideal DCG)

        Args:
            queries: List of query dicts with "query" and "relevant_ids" keys
                    Can optionally include "relevance_scores" dict {id -> score}
            k_values: List of k values to compute NDCG for

        Returns:
            Dict mapping k -> NDCG@k
        """
        ndcg_at_k = {k: [] for k in k_values}

        for query_item in tqdm(queries, desc="Computing NDCG@k"):
            query = query_item["query"]
            relevant_ids = set(query_item["relevant_ids"])

            # Get relevance scores (1.0 for binary relevance)
            relevance_scores = query_item.get(
                "relevance_scores",
                {doc_id: 1.0 for doc_id in relevant_ids}
            )

            # Get top results
            results = self.search(query, k=max(k_values))

            for k in k_values:
                # Compute DCG
                dcg = 0.0
                for rank, (idx, _) in enumerate(results[:k], start=1):
                    doc_id = self.corpus[idx]["id"]
                    relevance = relevance_scores.get(doc_id, 0.0)
                    dcg += relevance / math.log2(rank + 1)

                # Compute IDCG (ideal DCG)
                ideal_relevances = sorted(relevance_scores.values(), reverse=True)[:k]
                idcg = sum(
                    rel / math.log2(rank + 1)
                    for rank, rel in enumerate(ideal_relevances, start=1)
                )

                # Compute NDCG
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_at_k[k].append(ndcg)

        # Average across queries
        ndcg_at_k = {k: np.mean(scores) for k, scores in ndcg_at_k.items()}

        return ndcg_at_k

    def compute_map(self, queries: List[Dict]) -> float:
        """
        Compute Mean Average Precision (MAP).

        AP = (sum of Precision@k for each relevant doc) / num_relevant
        MAP = mean of AP across all queries

        Args:
            queries: List of query dicts with "query" and "relevant_ids" keys

        Returns:
            MAP score
        """
        average_precisions = []

        for query_item in tqdm(queries, desc="Computing MAP"):
            query = query_item["query"]
            relevant_ids = set(query_item["relevant_ids"])

            if not relevant_ids:
                continue

            # Get top results
            results = self.search(query, k=100)

            # Compute average precision
            num_relevant_found = 0
            precision_sum = 0.0

            for rank, (idx, _) in enumerate(results, start=1):
                doc_id = self.corpus[idx]["id"]
                if doc_id in relevant_ids:
                    num_relevant_found += 1
                    precision_at_k = num_relevant_found / rank
                    precision_sum += precision_at_k

            ap = precision_sum / len(relevant_ids) if relevant_ids else 0.0
            average_precisions.append(ap)

        return np.mean(average_precisions)

    def evaluate(
        self,
        queries_path: str,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        Run full evaluation on a set of queries.

        Args:
            queries_path: Path to queries JSONL file
                Expected format: {"query": str, "relevant_ids": List[str]}
            k_values: List of k values for metrics

        Returns:
            Dict of all computed metrics
        """
        # Load queries
        print(f"Loading queries from {queries_path}")
        queries = []
        with open(queries_path, 'r', encoding='utf-8') as f:
            for line in f:
                queries.append(json.loads(line.strip()))

        print(f"Loaded {len(queries)} queries")

        # Compute all metrics
        print("\nComputing metrics...")
        metrics = {}

        # Recall@k
        recall = self.compute_recall_at_k(queries, k_values)
        metrics["recall"] = recall

        # MRR
        mrr = self.compute_mrr(queries)
        metrics["mrr"] = mrr

        # NDCG@k
        ndcg = self.compute_ndcg_at_k(queries, k_values=[k for k in k_values if k > 1])
        metrics["ndcg"] = ndcg

        # MAP
        map_score = self.compute_map(queries)
        metrics["map"] = map_score

        # Add metadata
        metrics["metadata"] = {
            "n_queries": len(queries),
            "n_corpus": len(self.corpus),
            "model_path": str(self.model),
            "k_values": k_values
        }

        return metrics


def print_metrics(metrics: Dict):
    """Pretty print evaluation metrics."""
    print("\n" + "="*60)
    print("RETRIEVAL EVALUATION RESULTS")
    print("="*60)

    print(f"\nCorpus size: {metrics['metadata']['n_corpus']}")
    print(f"Number of queries: {metrics['metadata']['n_queries']}")

    print("\n--- Recall@k ---")
    for k, score in sorted(metrics["recall"].items()):
        print(f"  Recall@{k:2d}: {score:.4f} ({score*100:.2f}%)")

    print(f"\n--- Mean Reciprocal Rank (MRR) ---")
    print(f"  MRR: {metrics['mrr']:.4f}")

    print("\n--- NDCG@k ---")
    for k, score in sorted(metrics["ndcg"].items()):
        print(f"  NDCG@{k:2d}: {score:.4f}")

    print(f"\n--- Mean Average Precision (MAP) ---")
    print(f"  MAP: {metrics['map']:.4f}")

    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG performance of trained embedding models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to corpus JSONL file"
    )
    parser.add_argument(
        "--queries",
        type=str,
        required=True,
        help="Path to queries JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rag_metrics.json",
        help="Path to save results JSON"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="k values for Recall@k and NDCG@k"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "mps", "cpu"],
        default=None,
        help="Device to use (auto-detected if not specified)"
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = RAGEvaluator(
        model_path=args.model,
        device=args.device
    )

    # Load and index corpus
    evaluator.load_corpus(args.corpus)
    evaluator.index_corpus()

    # Run evaluation
    metrics = evaluator.evaluate(
        queries_path=args.queries,
        k_values=args.k_values
    )

    # Print results
    print_metrics(metrics)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
