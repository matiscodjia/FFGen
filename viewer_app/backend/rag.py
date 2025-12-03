"""
RAG (Retrieval Augmented Generation) testing utilities
MIPS-based retrieval: Code as query, Feedback as response
With persistent ChromaDB caching to avoid recomputing embeddings
"""

import json
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer

# Optional ChromaDB import for embedding caching
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


class RAGTester:
    """MIPS-based RAG: Code as query, Feedback as response with persistent caching"""

    def __init__(self, cache_dir: str = ".chroma_cache", progress_callback=None):
        self.feedbacks = []
        self.feedback_embeddings = None
        self.model = None
        self.collection = None
        self.cache_dir = os.path.join(os.path.dirname(__file__), "..", cache_dir)
        self.progress_callback = progress_callback  # Callback for UI updates

    def _log(self, message: str):
        """Log message - either to callback or print"""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)

    def _get_chromadb_client(self):
        """Get or create persistent ChromaDB client"""
        if not CHROMADB_AVAILABLE:
            return None

        return chromadb.PersistentClient(
            path=self.cache_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

    def _cache_embeddings_chunked(self, feedback_texts: List[str]):
        """
        Cache embeddings in ChromaDB using chunked upserts to avoid SQLite limits.
        SQLite has MAX_VARIABLE_NUMBER limit (~999), so we batch operations.
        """
        client = self._get_chromadb_client()
        if client is None:
            self._log("[ChromaDB] Not available, skipping cache")
            return

        # Get or create collection
        try:
            self.collection = client.get_collection("ffgen_mips_feedback")
            existing_count = self.collection.count()
            self._log(f"[ChromaDB] Found existing collection with {existing_count} items")
        except:
            self.collection = client.create_collection(
                name="ffgen_mips_feedback",
                metadata={"description": "FFGen MIPS Feedback Cache"}
            )
            self._log("[ChromaDB] Created new collection")

        # Upsert embeddings in chunks to avoid SQLite MAX_VARIABLE_NUMBER limit
        # SQLite typically has MAX_VARIABLE_NUMBER of ~32766 on modern systems
        # But ChromaDB implements a max_batch_size check based on compiled limits
        # To be safe, we use chunks that work for most SQLite compilations
        # Empirically, 5000+ items causes issues, so we chunk at 500 for safety
        chunk_size = 500
        total_items = len(self.feedbacks)

        self._log(f"[ChromaDB] Caching {total_items} embeddings in chunks of {chunk_size}...")

        for i in range(0, total_items, chunk_size):
            chunk_end = min(i + chunk_size, total_items)

            chunk_ids = [str(self.feedbacks[j]["id"]) for j in range(i, chunk_end)]
            chunk_embeddings = self.feedback_embeddings[i:chunk_end].tolist()
            chunk_documents = feedback_texts[i:chunk_end]
            chunk_metadatas = [{"index": j, "feedback_len": len(self.feedbacks[j]["feedback"])}
                              for j in range(i, chunk_end)]

            try:
                # Use upsert to handle both new and existing entries
                self.collection.upsert(
                    ids=chunk_ids,
                    embeddings=chunk_embeddings,
                    documents=chunk_documents,
                    metadatas=chunk_metadatas
                )
                self._log(f"[ChromaDB] Cached chunk {i//chunk_size + 1}/{(total_items + chunk_size - 1)//chunk_size}")
            except Exception as e:
                self._log(f"[ChromaDB] Warning: Failed to cache chunk {i}-{chunk_end}: {e}")

        self._log(f"[ChromaDB] ✓ Successfully cached {total_items} feedback embeddings")

    def _load_from_cache(self, feedback_ids: List[str]) -> Optional[np.ndarray]:
        """
        Try to load embeddings from ChromaDB cache.
        Returns embeddings array if all IDs found, None otherwise.
        """
        client = self._get_chromadb_client()
        if client is None:
            return None

        try:
            self.collection = client.get_collection("ffgen_mips_feedback")

            # Query in chunks to avoid SQLite limits
            chunk_size = 500
            all_embeddings = []

            for i in range(0, len(feedback_ids), chunk_size):
                chunk_ids = feedback_ids[i:i+chunk_size]
                results = self.collection.get(
                    ids=[str(fid) for fid in chunk_ids],
                    include=["embeddings"]
                )

                if len(results['ids']) != len(chunk_ids):
                    # Not all IDs found in cache
                    return None

                all_embeddings.extend(results['embeddings'])

            self._log(f"[ChromaDB] ✓ Loaded {len(all_embeddings)} embeddings from cache")
            return np.array(all_embeddings)

        except Exception as e:
            self._log(f"[ChromaDB] Cache miss: {e}")
            return None

    def index_corpus(
        self,
        dataset: List[Dict],
        model: SentenceTransformer,
        use_chromadb: bool = True
    ):
        """
        Index feedback corpus for MIPS retrieval with persistent caching.

        Args:
            dataset: List of dataset items with code_snippet and conceptual_feedback
            model: Loaded sentence transformer model
            use_chromadb: Whether to use ChromaDB for caching (optional)
        """
        self.model = model

        # Prepare feedback corpus
        self.feedbacks = [
            {
                "id": item.get('code_id', f"doc_{i}"),
                "feedback": item.get('conceptual_feedback', ''),
                "code": item.get('code_snippet', ''),
                "metadata": item
            }
            for i, item in enumerate(dataset)
            if item.get('conceptual_feedback', '').strip()
        ]

        self._log(f"[RAG] Indexing {len(self.feedbacks)} feedbacks...")

        # Truncate very long texts to avoid token limit issues
        feedback_texts = [doc["feedback"] for doc in self.feedbacks]
        max_chars = 2000  # Approximately 500 tokens
        feedback_texts = [text[:max_chars] if len(text) > max_chars else text
                         for text in feedback_texts]

        # Try to load from cache first
        if use_chromadb and CHROMADB_AVAILABLE:
            feedback_ids = [doc["id"] for doc in self.feedbacks]
            cached_embeddings = self._load_from_cache(feedback_ids)

            if cached_embeddings is not None:
                self.feedback_embeddings = cached_embeddings
                self._log("[RAG] ✓ Using cached embeddings (no recomputation needed)")
                return

        # Cache miss or disabled - compute embeddings
        self._log("[RAG] Computing embeddings (this may take a while)...")
        self.feedback_embeddings = model.encode(
            feedback_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Cache the computed embeddings for next time
        if use_chromadb and CHROMADB_AVAILABLE:
            self._cache_embeddings_chunked(feedback_texts)

        self._log(f"[RAG] ✓ Indexed {len(self.feedbacks)} feedbacks")

    def index_corpus_from_file(
        self,
        corpus_file,
        model: SentenceTransformer,
        use_chromadb: bool = True
    ):
        """
        Index feedback corpus from uploaded JSONL file.

        Args:
            corpus_file: Uploaded file object
            model: Loaded sentence transformer model
            use_chromadb: Whether to use ChromaDB for caching (optional)
        """
        self.model = model
        self.feedbacks = []

        content = corpus_file.read().decode('utf-8')
        for i, line in enumerate(content.strip().split('\n')):
            if line.strip():
                item = json.loads(line)
                feedback_text = item.get("conceptual_feedback", item.get("feedback", ""))
                if feedback_text.strip():
                    self.feedbacks.append({
                        "id": item.get("code_id", item.get("id", f"doc_{i}")),
                        "feedback": feedback_text,
                        "code": item.get("code_snippet", item.get("code", "")),
                        "metadata": item
                    })

        self._log(f"[RAG] Indexing {len(self.feedbacks)} feedbacks from file...")

        # Truncate very long texts
        feedback_texts = [doc["feedback"] for doc in self.feedbacks]
        max_chars = 2000
        feedback_texts = [text[:max_chars] if len(text) > max_chars else text
                         for text in feedback_texts]

        # Try cache first
        if use_chromadb and CHROMADB_AVAILABLE:
            feedback_ids = [doc["id"] for doc in self.feedbacks]
            cached_embeddings = self._load_from_cache(feedback_ids)

            if cached_embeddings is not None:
                self.feedback_embeddings = cached_embeddings
                self._log("[RAG] ✓ Using cached embeddings")
                return

        # Compute and cache
        self._log("[RAG] Computing embeddings...")
        self.feedback_embeddings = model.encode(
            feedback_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        if use_chromadb and CHROMADB_AVAILABLE:
            self._cache_embeddings_chunked(feedback_texts)

        self._log(f"[RAG] ✓ Indexed {len(self.feedbacks)} feedbacks")

    def search(self, code_query: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """
        MIPS search: Given code query, find most relevant feedbacks.

        Args:
            code_query: Code snippet query
            top_k: Number of results to return

        Returns:
            List of tuples: (rank, similarity_score, feedback_document)
        """
        if self.feedback_embeddings is None or self.model is None:
            raise ValueError("Corpus not indexed. Call index_corpus first.")

        # Truncate query to avoid token limit issues
        max_chars = 2000
        code_query_truncated = code_query[:max_chars] if len(code_query) > max_chars else code_query

        # Encode code query
        query_embedding = self.model.encode(
            code_query_truncated,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        # MIPS: Maximum Inner Product Search
        # Compute similarities between query and all feedbacks
        similarities = np.dot(self.feedback_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return results
        results = []
        for rank, idx in enumerate(top_indices, start=1):
            score = similarities[idx]
            feedback_doc = self.feedbacks[idx]
            results.append((rank, float(score), feedback_doc))

        return results

    def get_corpus_stats(self) -> Optional[Dict]:
        """
        Get statistics about the indexed feedback corpus.

        Returns:
            Dict with corpus statistics or None if not indexed
        """
        if not self.feedbacks or self.feedback_embeddings is None:
            return None

        return {
            "n_feedbacks": len(self.feedbacks),
            "avg_feedback_length": np.mean([len(doc['feedback']) for doc in self.feedbacks]),
            "embedding_dim": self.feedback_embeddings.shape[1],
            "total_embeddings": len(self.feedback_embeddings)
        }

    def clear_cache(self):
        """Clear the ChromaDB cache (useful for testing or reset)"""
        client = self._get_chromadb_client()
        if client is None:
            return

        try:
            client.delete_collection("ffgen_mips_feedback")
            self._log("[ChromaDB] ✓ Cache cleared")
        except Exception as e:
            self._log(f"[ChromaDB] No cache to clear: {e}")
