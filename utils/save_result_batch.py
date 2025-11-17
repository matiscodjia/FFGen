import json
import logging
from pathlib import Path
from typing import Any, Dict, List



def save_result_batch(dataset_path: str, batch_items: List[Dict[str, Any]]) -> None:
    """
    Append a batch of processed items to the JSONL dataset file.

    Each item is written as a single JSON line. This function is designed to be
    called repeatedly during batch processing to incrementally save results.

    Args:
        dataset_path: Path to the JSONL dataset file
        batch_items: List of dictionaries to append to the dataset

    Raises:
        IOError: If unable to write to the file
    """
    dataset_file = Path(dataset_path)

    # Create parent directory if it doesn't exist
    dataset_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(dataset_file, 'a', encoding='utf-8') as f:
            for item in batch_items:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

        logger = logging.getLogger(__name__)
        logger.debug(f"Saved batch of {len(batch_items)} items to {dataset_path}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving batch to {dataset_path}: {e}")
        raise IOError(f"Failed to save batch to {dataset_path}: {e}")
