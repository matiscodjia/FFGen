#!/usr/bin/env python3
"""
FFGen ML Pipeline Orchestrator

Executes the complete ML pipeline for Focused Feedback Generation:
  - Stage 1: Data Mining and Acquisition
  - Stage 2: Data Generation (augmentation)
  - Stage 3: Data Processing
  - Stage 4: Model Training and Finetuning

Usage:
    .venv/bin/python pipeline.py [--config CONFIG_PATH] [--stage STAGE_NUM]
    # Or activate venv first: source .venv/bin/activate && python pipeline.py

Examples:
    # Run complete pipeline
    python pipeline.py

    # Run with custom config
    python pipeline.py --config configs/experiment_001.yml

    # Run specific stage only
    python pipeline.py --stage 1  # Data acquisition only
    python pipeline.py --stage 2  # Data Generation only
    python pipeline.py --stage 3  # Data Processing
    python pipeline.py --stage 4  # Model training only
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


from utils import (
    load_config,
    setup_logging,
    init_report_file,
    log_experiment_to_csv,
    get_device
)

# Import from stage modules (can't use regular import due to module names starting with numbers)
import importlib.util

def import_from_path(module_name: str, file_path: str):
    """Import a module from a file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Import stage modules
stage1 = import_from_path("stage1", project_root / "1_data_acquisition" / "ingest_code.py")
stage2 = import_from_path("stage2", project_root / "2_data_generation" / "generate_feedback.py")
stage3 = import_from_path("stage3", project_root / "3_data_processing" / "generate_hybrid_negatives.py")
stage4 = import_from_path("stage4", project_root / "4_model_training" / "train_embedding.py")

run_data_acquisition = stage1.run_data_acquisition
run_feedback_generation = stage2.run_feedback_generation
run_model_training = stage4.run_model_training



def print_banner(run_id: str, device: str):
    """Print pipeline startup banner"""
    banner = f"""
{'='*70}
              FFGen ML Pipeline - Focused Feedback Generation
{'='*70}
  Run ID: {run_id}
  Device: {device.upper()}
{'='*70}
"""
    print(banner)


def run_stage_1(config: dict) -> Optional[str]:
    """Execute Stage 1: Data Mining and Acquisition"""
    print("\n" + "="*70)
    print("STAGE 1: DATA MINING AND ACQUISITION")
    print("="*70)

    try:
        output_path = run_data_acquisition(config)
        print(f"\n✓ Stage 1 completed successfully")
        return output_path
    except Exception as e:
        print(f"\n✗ Stage 1 failed: {e}")
        raise

def run_stage_2(config: dict) -> Optional[str]:
    """Execute Stage 2: Data Processing and Preprocessing"""
    print("\n" + "="*70)
    print("STAGE 2: DATA GENERATION")
    print("="*70)

    try:
        # Step 2: LLM-based feedback generation
        dataset_path = run_feedback_generation(config)
        print(f"\n✓ Stage 2 completed successfully")
        return dataset_path
    except Exception as e:
        print(f"\n✗ Stage 2 failed: {e}")
        raise
def run_stage_3(config: dict, dataset_path: str) -> Optional[str]:
    """Execute Stage 3: Data Processing (Negative Mining)"""
    print("\n" + "="*70)
    print("STAGE 3: DATA PROCESSING - NEGATIVE MINING")
    print("="*70)

    try:
        # Hybrid Negative Generation (Hard + Easy/Random)
        hnm_config = config.get('negative_mining', {})
        if hnm_config.get('enabled', True):
            print("\n" + "-"*70)
            print("STAGE 2.5: HYBRID NEGATIVE GENERATION")
            print("-"*70)

            dataset = stage3.load_dataset(dataset_path)
            
            generator = stage3.HybridNegativeGenerator(
                embedding_model=hnm_config.get('embedding_model', 'google/embeddinggemma-300m'),
                batch_size=hnm_config.get('batch_size', 32)
            )

            total_neg = hnm_config.get('total_negatives', 5)
            num_hard = hnm_config.get('num_hard', 2)
            min_sim = hnm_config.get('min_similarity', 0.2)
            max_sim = hnm_config.get('max_similarity', 0.4)

            examples_with_negatives = generator.generate_hybrid_negatives(
                dataset,
                total_negatives=total_neg,
                num_hard=num_hard,
                min_similarity=min_sim,
                max_similarity=max_sim
            )

            stage3.save_dataset(examples_with_negatives, dataset_path)

        print(f"\n✓ Stage 3 completed successfully")
        return dataset_path
    except Exception as e:
        print(f"\n✗ Stage 3 failed: {e}")
        raise


def run_stage_4(config: dict, dataset_path: str) -> dict:
    """Execute Stage 3: Model Training and Finetuning"""
    print("\n" + "="*70)
    print("STAGE 3: MODEL TRAINING AND FINETUNING")
    print("="*70)

    try:
        metrics = run_model_training(config, dataset_path)
        print(f"\n✓ Stage 3 completed successfully")
        return metrics
    except Exception as e:
        print(f"\n✗ Stage 3 failed: {e}")
        raise


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(
        description="FFGen ML Pipeline - Focused Feedback Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yml',
        help='Path to configuration file (default: configs/config.yml)'
    )
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run only a specific stage (1, 2, or 3). If omitted, runs all stages.'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(verbose=args.verbose)

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    run_id = config.get('run_id', 'UNKNOWN')
    device = get_device()

    # Print banner
    print_banner(run_id, device)

    # Initialize report file
    init_report_file(config)

    try:
        # Determine which stages to run
        stages_to_run = [args.stage] if args.stage else [1, 2, 3, 4]

        dataset_path = None
        final_metrics = {}

        # Execute requested stages
        if 1 in stages_to_run:
            run_stage_1(config)

        if 2 in stages_to_run:
            dataset_path = run_stage_2(config)

        if 3 in stages_to_run:
            # If we didn't run stage 2, get dataset path from config
            if dataset_path is None:
                dataset_path = config['generation']['final_dataset_file']

            dataset_path = run_stage_3(config, dataset_path)

        if 4 in stages_to_run:
            # If we didn't run stage 2 or 3, get dataset path from config
            if dataset_path is None:
                dataset_path = config['generation']['final_dataset_file']

            final_metrics = run_stage_4(config, dataset_path)

        # Log results if we ran stage 4
        if 4 in stages_to_run and final_metrics:
            log_experiment_to_csv(config, final_metrics)

        # Print success banner
        print("\n" + "="*70)
        print(f"  PIPELINE COMPLETED SUCCESSFULLY - Run ID: {run_id}")
        if final_metrics:
            print(f"  Results logged to: {config.get('reporting', {}).get('report_file', 'results.csv')}")
        print("="*70 + "\n")

    except Exception as e:
        # Print failure banner
        print("\n" + "="*70)
        print(f"  PIPELINE FAILED - Run ID: {run_id}")
        print(f"  Error: {e}")
        print("="*70 + "\n")

        # Log failure
        try:
            error_metrics = {
                "final_test_loss": "FAILED",
                "model_output_path": str(e)
            }
            log_experiment_to_csv(config, error_metrics)
        except:
            pass

        sys.exit(1)


if __name__ == "__main__":
    main()
