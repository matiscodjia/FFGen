"""
Data loading and saving utilities
"""

import json
from typing import List, Dict, Optional


def load_dataset(file) -> Optional[List[Dict]]:
    """
    Load JSONL dataset from uploaded file.

    Args:
        file: Uploaded file object from Streamlit

    Returns:
        List of dataset items or None if error
    """
    try:
        content = file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        dataset = [json.loads(line) for line in lines]
        return dataset
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")


def save_dataset(dataset: List[Dict]) -> str:
    """
    Convert dataset to JSONL format for download.

    Args:
        dataset: List of dataset items

    Returns:
        JSONL formatted string
    """
    return '\n'.join([json.dumps(item) for item in dataset])
