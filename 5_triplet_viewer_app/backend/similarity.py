"""
Similarity computation utilities
"""

import asyncio
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from utils.inference_service import InferenceServer


def compute_similarities(positive: str, negatives: List[str]) -> List[float]:
    """
    Compute cosine similarities between positive and negatives.

    Args:
        positive: Positive feedback text
        negatives: List of negative feedback texts

    Returns:
        List of cosine similarity scores
    """
    embedder = InferenceServer()
    if not negatives or not positive:
        return []

    try:
        positive_emb = asyncio.run(embedder.encode([positive]))
        negative_embs = asyncio.run(embedder.encode(negatives))
        similarities = cosine_similarity(positive_emb, negative_embs)[0]
        return similarities.tolist()
    except Exception as e:
        raise ValueError(f"Error computing similarities: {e}")


def get_similarity_color(sim: float) -> str:
    """
    Get color class based on similarity value.

    Args:
        sim: Similarity score (0-1)

    Returns:
        CSS class name for color coding
    """
    if sim > 0.6:
        return 'sim-high'
    elif sim > 0.4:
        return 'sim-medium'
    else:
        return 'sim-low'
