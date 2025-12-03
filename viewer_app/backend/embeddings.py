"""
Embedding model loading and computation utilities with server fallback support
"""

import random
import sys
from pathlib import Path
from typing import Dict, List, Union
from sentence_transformers import SentenceTransformer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from utils.inference_service import InferenceServer
    INFERENCE_SERVICE_AVAILABLE = True
except ImportError:
    INFERENCE_SERVICE_AVAILABLE = False
    print("[Embeddings] Warning: inference_service not available, using local models only")


def load_embedding_model(
    model_name: str = None,
    config_path: str = "./configs/config.yml",
    server_url: str = None,
    use_server: bool = True
) -> Union[SentenceTransformer, "InferenceServer"]:
    """
    Load embedding model with fallback support.

    Priority:
    1. Try loading from config file (if config_path provided)
    2. Try OpenAI-compatible server (if server_url provided and use_server=True)
    3. Fall back to local SentenceTransformers

    Args:
        model_name: Name or path of the model (optional if using config)
        config_path: Path to config YAML file
        server_url: OpenAI-compatible server URL (e.g., "http://localhost:8000/v1")
        use_server: Whether to attempt server connection first

    Returns:
        Loaded model (InferenceServer or SentenceTransformer)
    """
    # Try loading from config first
    if use_server and INFERENCE_SERVICE_AVAILABLE and config_path:
        try:
            return InferenceServer.from_config(
                config_path=config_path,
                service_type="embeddings"
            )
        except Exception as e:
            print(f"[Embeddings] Config loading failed: {e}")

    # Fallback to server_url parameter
    if use_server and server_url and INFERENCE_SERVICE_AVAILABLE:
        try:
            # Try server with fallback to local model
            return InferenceServer(
                url=server_url,
                model_name="default",
                fallback_model=model_name
            )
        except Exception as e:
            print(f"[Embeddings] Server failed, using local model: {e}")
            if model_name:
                return SentenceTransformer(model_name)

    # Direct local model loading
    if model_name:
        return SentenceTransformer(model_name)
    else:
        raise ValueError("Either model_name or valid config_path must be provided")


def _encode_with_model(model: Union[SentenceTransformer, "InferenceServer"], texts: List[str], **kwargs):
    """
    Unified encoding function that works with both SentenceTransformer and InferenceServer.

    Args:
        model: SentenceTransformer or InferenceServer instance
        texts: List of texts to encode
        **kwargs: Additional encoding parameters

    Returns:
        Embeddings array
    """
    if isinstance(model, SentenceTransformer):
        # Standard SentenceTransformers encoding
        return model.encode(texts, **kwargs)
    elif INFERENCE_SERVICE_AVAILABLE and isinstance(model, InferenceServer):
        # InferenceServer encoding (handles both server and fallback)
        import asyncio
        # Extract relevant kwargs for InferenceServer
        batch_size = kwargs.get('batch_size', 32)
        normalize_embeddings = kwargs.get('normalize_embeddings', True)
        show_progress_bar = kwargs.get('show_progress_bar', False)

        embeddings = asyncio.run(model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar
        ))

        import numpy as np
        return np.array(embeddings)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


def compute_embeddings_for_visualization(
    item: Dict,
    model: Union[SentenceTransformer, "InferenceServer"]
) -> Dict:
    """
    Compute embeddings for anchor, positive, and negatives for single triplet visualization.

    Args:
        item: Dataset item containing code_snippet, conceptual_feedback, and negative_feedbacks
        model: Loaded embedding model (SentenceTransformer or InferenceServer)

    Returns:
        Dict with keys:
            - embeddings: numpy array of shape (n_texts, embedding_dim)
            - labels: list of text labels for each embedding
            - types: list of types ('anchor', 'positive', 'negative')
    """
    code = item.get('code_snippet', '')
    positive = item.get('conceptual_feedback', item.get('refined_feedback', ''))
    negatives = item.get('negative_feedbacks', [])

    # Handle legacy format
    if not negatives and 'hard_negative_feedback' in item:
        negatives = [item['hard_negative_feedback']]

    texts = [code, positive] + negatives
    embeddings = _encode_with_model(model, texts)

    return {
        'embeddings': embeddings,
        'labels': ['Anchor (Code)', 'Positive'] + [f'Negative {i+1}' for i in range(len(negatives))],
        'types': ['anchor', 'positive'] + ['negative'] * len(negatives)
    }


def compute_global_embeddings(
    dataset: List[Dict],
    n_samples: int,
    model: Union[SentenceTransformer, "InferenceServer"]
) -> Dict:
    """
    Compute embeddings for N random triplets from dataset for global visualization.

    Args:
        dataset: List of dataset items
        n_samples: Number of triplets to sample
        model: Loaded embedding model (SentenceTransformer or InferenceServer)

    Returns:
        Dict with keys:
            - embeddings: numpy array of all embeddings
            - labels: list of labels for each point
            - types: list of types (anchor/positive/negative)
            - connections: list of tuples (start_idx, end_idx, type) for drawing edges
    """
    # Sample N random examples
    if n_samples > len(dataset):
        n_samples = len(dataset)

    sampled_items = random.sample(dataset, n_samples)

    all_texts = []
    labels = []
    types = []
    connections = []

    current_idx = 0

    for item_idx, item in enumerate(sampled_items):
        code = item.get('code_snippet', '')
        positive = item.get('conceptual_feedback', item.get('refined_feedback', ''))
        negatives = item.get('negative_feedbacks', [])

        # Handle legacy format
        if not negatives and 'hard_negative_feedback' in item:
            negatives = [item['hard_negative_feedback']]

        # Add anchor (code)
        anchor_idx = current_idx
        all_texts.append(code[:100])  # Truncate for display
        labels.append(f"Code {item_idx+1}")
        types.append('anchor')
        current_idx += 1

        # Add positive
        positive_idx = current_idx
        all_texts.append(positive)
        labels.append(f"Pos {item_idx+1}")
        types.append('positive')
        connections.append((anchor_idx, positive_idx, 'positive'))
        current_idx += 1

        # Add negatives
        for neg_idx, neg in enumerate(negatives):
            negative_idx = current_idx
            all_texts.append(neg)
            labels.append(f"Neg {item_idx+1}.{neg_idx+1}")
            types.append('negative')
            connections.append((anchor_idx, negative_idx, 'negative'))
            current_idx += 1

    # Compute embeddings
    embeddings = _encode_with_model(model, all_texts)

    return {
        'embeddings': embeddings,
        'labels': labels,
        'types': types,
        'connections': connections
    }
