import csv
import logging
from pathlib import Path
from typing import Any, Dict


def init_report_file(config: Dict[str, Any]) -> None:
    """
    Initialize the CSV report file with headers if it doesn't exist.

    Args:
        config: Configuration dictionary containing reporting settings
    """
    report_file = config.get('reporting', {}).get('report_file', './results.csv')
    report_path = Path(report_file)

    # Create parent directory if it doesn't exist
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # If file doesn't exist, create it with headers
    if not report_path.exists():
        headers = [
            'timestamp',
            'run_id',
            'llm_model',
            'base_embedding_model',
            'training_mode',
            'num_epochs',
            'batch_size',
            'learning_rate',
            'warmup_ratio',
            'triplet_margin',
            'evaluator_type',
            'metric_for_best_model',
            'negative_mining_enabled',
            'negative_mining_strategy',
            'negative_mining_model',
            'min_similarity',
            'max_similarity',
            'final_test_loss',
            'final_metric_value',
            'model_output_path',
            'dataset_path',
            'notes'
        ]

        with open(report_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)

        logger = logging.getLogger(__name__)
        logger.info(f"Initialized report file: {report_path}")
