import json
import logging
from pathlib import Path
from typing import Set


def load_progress(dataset_path: str, id_column: str = 'code_id') -> Set[str]:
    """
    Load progress from an existing JSONL dataset file to enable resumable processing.

    Reads the dataset file and extracts all IDs that have already been processed,
    allowing the pipeline to skip these items and resume where it left off.

    Args:
        dataset_path: Path to the JSONL dataset file
        id_column: Name of the column containing unique IDs (default: 'code_id')

    Returns:
        Set of IDs that have already been processed
    """
    dataset_file = Path(dataset_path)

    if not dataset_file.exists():
        logger = logging.getLogger(__name__)
        logger.info(f"No existing dataset found at {dataset_path}. Starting from scratch.")
        return set()

    processed_ids = set()

    try:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    if id_column in item:
                        processed_ids.add(item[id_column])
                    else:
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Line {line_num} in {dataset_path} missing '{id_column}' field"
                        )
                except json.JSONDecodeError as e:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Invalid JSON on line {line_num} in {dataset_path}: {e}")
                    continue

        logger = logging.getLogger(__name__)
        logger.info(f"Loaded {len(processed_ids)} previously processed items from {dataset_path}")

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error reading progress from {dataset_path}: {e}")
        return set()

    return processed_ids