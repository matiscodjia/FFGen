"""
Similarity computation utilities
"""

import asyncio
from typing import List, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

try:
    from utils.inference_service import InferenceServer
    INFERENCE_SERVICE_AVAILABLE = True
except ImportError:
    INFERENCE_SERVICE_AVAILABLE = False


def compute_similarities(positive: str, negatives: List[str], model: Union[SentenceTransformer, "InferenceServer"] = None) -> List[float]:
    """
    Compute cosine similarities between positive and negatives.

    Args:
        positive: Positive feedback text
        negatives: List of negative feedback texts
        model: Model instance (SentenceTransformer or InferenceServer)

    Returns:
        List of cosine similarity scores
    """
    if not negatives or not positive:
        return []

    if model is None:
        raise ValueError("Model must be provided. Please load a model first.")

    try:
        # Handle different model types
        if isinstance(model, SentenceTransformer):
            # Standard SentenceTransformers encoding
            positive_emb = model.encode([positive])
            negative_embs = model.encode(negatives)
        elif INFERENCE_SERVICE_AVAILABLE and isinstance(model, InferenceServer):
            # InferenceServer encoding
            positive_emb = asyncio.run(model.encode([positive]))
            negative_embs = asyncio.run(model.encode(negatives))
        else:
            raise TypeError(f"Unsupported model type: {type(model)}")

        similarities = cosine_similarity([positive_emb[0]], negative_embs)[0]
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
