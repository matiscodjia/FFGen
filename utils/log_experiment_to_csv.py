
import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from utils import init_report_file


def log_experiment_to_csv(config: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """
    Log experiment results to CSV report file.

    Args:
        config: Configuration dictionary
        metrics: Dictionary containing training metrics and results
    """
    report_file = config.get('reporting', {}).get('report_file', './results.csv')
    report_path = Path(report_file)

    # Ensure file exists with headers
    init_report_file(config)

    # Extract configuration values
    generation_config = config.get('generation', {})
    training_config = config.get('training', {})
    hyperparams = training_config.get('hyperparameters', {})
    negative_mining = config.get('negative_mining', {})

    # Prepare row data
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # timestamp
        config.get('run_id', 'UNKNOWN'),
        generation_config.get('llm_model', 'N/A'),
        training_config.get('base_embedding_model', 'N/A'),
        training_config.get('mode', 'N/A'),
        hyperparams.get('num_epochs', 'N/A'),
        hyperparams.get('batch_size', 'N/A'),
        hyperparams.get('learning_rate', 'N/A'),
        hyperparams.get('warmup_ratio', 'N/A'),
        hyperparams.get('triplet_margin', 'N/A'),
        training_config.get('evaluator_type', 'N/A'),
        training_config.get('metric_for_best_model', 'N/A'),
        negative_mining.get('enabled', False),
        negative_mining.get('strategy', 'N/A'),
        negative_mining.get('embedding_model', 'N/A'),
        negative_mining.get('min_similarity', 'N/A'),
        negative_mining.get('max_similarity', 'N/A'),
        metrics.get('final_test_loss', 'N/A'),
        metrics.get('final_metric_value', 'N/A'),
        metrics.get('model_output_path', 'N/A'),
        metrics.get('dataset_path', 'N/A'),
        metrics.get('notes', '')
    ]

    # Append to CSV
    with open(report_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(row)

    logger = logging.getLogger(__name__)
    logger.info(f"Logged experiment to: {report_path}")