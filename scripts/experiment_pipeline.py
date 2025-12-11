"""
Industrial Pipeline for Massive LoRA Adapter Training
======================================================

This script orchestrates the training of multiple LoRA adapters across:
- 2 base models: jinaai/jina-embeddings-v3 (code-embed) & google/gemma-2-2b-it (300M)
- 4 batch sizes: 32, 64, 128, 256
- 2 datasets: RAFT (original) & RAFT Ultra-Clean
Total: 16 adapter configurations

Each training run tracks comprehensive metrics for comparative analysis.
"""

import os
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import shutil

@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment"""
    experiment_id: str
    base_model: str
    batch_size: int
    dataset_name: str
    dataset_path: str
    output_dir: str
    hub_model_id: str

    # Training hyperparameters
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_steps: int = 100
    temperature: float = 0.07
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Tokenization
    code_max_length: int = 512
    feedback_max_length: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class IndustrialPipeline:
    """Main pipeline orchestrator for massive training experiments"""

    # Base model configurations
    BASE_MODELS = {
        "jina-code-embed": {
            "model_name": "jinaai/jina-embeddings-v3",
            "hub_prefix": "matiscodjia/ffgen-jina-code",
        },
        "gemma-embedding-300m": {
            "model_name": "google/gemma-2-2b-it",
            "hub_prefix": "matiscodjia/ffgen-gemma-300m",
        }
    }

    # Dataset configurations
    DATASETS = {
        "raft": {
            "name": "RAFT Original",
            "path": "data/raft_dataset.jsonl",
            "quality": "standard"
        },
        "raft-ultra-clean": {
            "name": "RAFT Ultra Clean",
            "path": "data/ultra_clean_final_dataset/train.jsonl",
            "quality": "high"
        }
    }

    BATCH_SIZES = [32, 64, 128, 256]

    def __init__(self,
                 workspace_dir: str = "experiments",
                 dry_run: bool = False,
                 resume_from: str = None):
        """
        Initialize the pipeline

        Args:
            workspace_dir: Directory to store all experiments
            dry_run: If True, only print what would be executed
            resume_from: Resume from a specific experiment ID
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(exist_ok=True)
        self.dry_run = dry_run
        self.resume_from = resume_from

        # Create subdirectories
        self.runs_dir = self.workspace_dir / "runs"
        self.logs_dir = self.workspace_dir / "logs"
        self.results_dir = self.workspace_dir / "results"
        self.checkpoints_dir = self.workspace_dir / "checkpoints"

        for d in [self.runs_dir, self.logs_dir, self.results_dir, self.checkpoints_dir]:
            d.mkdir(exist_ok=True)

        # Pipeline state
        self.start_time = datetime.now()
        self.experiments: List[ExperimentConfig] = []
        self.completed_experiments: List[str] = []
        self.failed_experiments: List[Dict[str, Any]] = []

        print(f"Pipeline initialized")
        print(f"Workspace: {self.workspace_dir.absolute()}")
        print(f"Dry run: {self.dry_run}")

    def generate_experiments(self) -> List[ExperimentConfig]:
        """Generate all experiment configurations"""
        experiments = []

        for model_key, model_config in self.BASE_MODELS.items():
            for dataset_key, dataset_config in self.DATASETS.items():
                for batch_size in self.BATCH_SIZES:
                    exp_id = f"{model_key}_{dataset_key}_bs{batch_size}"

                    # Check if resuming and this experiment is already done
                    if self.resume_from and self._is_experiment_completed(exp_id):
                        print(f"⏭️  Skipping completed experiment: {exp_id}")
                        self.completed_experiments.append(exp_id)
                        continue

                    hub_model_id = f"{model_config['hub_prefix']}-{dataset_key}-bs{batch_size}"
                    output_dir = str(self.runs_dir / exp_id)

                    config = ExperimentConfig(
                        experiment_id=exp_id,
                        base_model=model_config["model_name"],
                        batch_size=batch_size,
                        dataset_name=dataset_config["name"],
                        dataset_path=dataset_config["path"],
                        output_dir=output_dir,
                        hub_model_id=hub_model_id,
                    )

                    experiments.append(config)

        self.experiments = experiments
        return experiments

    def _is_experiment_completed(self, exp_id: str) -> bool:
        """Check if an experiment has already completed successfully"""
        result_file = self.results_dir / f"{exp_id}_results.json"
        return result_file.exists()

    def save_experiment_plan(self):
        """Save the complete experiment plan to disk"""
        plan_file = self.workspace_dir / "experiment_plan.json"

        plan = {
            "generated_at": self.start_time.isoformat(),
            "total_experiments": len(self.experiments),
            "completed_experiments": len(self.completed_experiments),
            "base_models": list(self.BASE_MODELS.keys()),
            "datasets": list(self.DATASETS.keys()),
            "batch_sizes": self.BATCH_SIZES,
            "experiments": [exp.to_dict() for exp in self.experiments]
        }

        with open(plan_file, 'w') as f:
            json.dump(plan, f, indent=2)

        print(f"Experiment plan saved to {plan_file}")
        print(f"Total experiments: {len(self.experiments)}")
        print(f"Already completed: {len(self.completed_experiments)}")
        print(f"To run: {len(self.experiments)}")

    def run_single_experiment(self, config: ExperimentConfig) -> bool:
        """
        Execute a single training experiment

        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"🚀 Starting Experiment: {config.experiment_id}")
        print(f"{'='*80}")
        print(f"  Model: {config.base_model}")
        print(f"  Dataset: {config.dataset_name}")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Output: {config.output_dir}")
        print(f"  Hub ID: {config.hub_model_id}")

        if self.dry_run:
            print("  [DRY RUN] Would execute training here")
            time.sleep(1)
            return True

        # Create experiment directory
        exp_dir = Path(config.output_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)

        # Prepare training command
        log_file = self.logs_dir / f"{config.experiment_id}.log"

        # Build the training script call
        train_cmd = [
            "python", "scripts/train_single_experiment.py",
            "--config", str(config_file),
            "--output-dir", config.output_dir,
            "--log-file", str(log_file)
        ]

        print(f"Logging to: {log_file}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        start_time = time.time()

        try:
            # Run training
            with open(log_file, 'w') as f:
                process = subprocess.Popen(
                    train_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )

                # Stream output
                for line in process.stdout:
                    print(f"  | {line.rstrip()}")
                    f.write(line)
                    f.flush()

                process.wait()

                if process.returncode != 0:
                    raise subprocess.CalledProcessError(
                        process.returncode, train_cmd
                    )

            duration = time.time() - start_time

            print(f"Experiment completed successfully!")
            print(f"Duration: {duration:.2f}s ({duration/60:.2f}min)")

            # Mark as completed
            self.completed_experiments.append(config.experiment_id)

            # Save completion marker
            completion_file = self.results_dir / f"{config.experiment_id}_results.json"
            with open(completion_file, 'w') as f:
                json.dump({
                    "experiment_id": config.experiment_id,
                    "status": "completed",
                    "duration_seconds": duration,
                    "completed_at": datetime.now().isoformat(),
                    "config": config.to_dict()
                }, f, indent=2)

            return True

        except subprocess.CalledProcessError as e:
            duration = time.time() - start_time
            print(f"Experiment failed with return code {e.returncode}")
            print(f"  Duration before failure: {duration:.2f}s")

            self.failed_experiments.append({
                "experiment_id": config.experiment_id,
                "error": str(e),
                "returncode": e.returncode,
                "duration": duration,
                "failed_at": datetime.now().isoformat()
            })

            return False

        except Exception as e:
            duration = time.time() - start_time
            print(f"Experiment failed with exception: {e}")
            print(f"Duration before failure: {duration:.2f}s")

            self.failed_experiments.append({
                "experiment_id": config.experiment_id,
                "error": str(e),
                "exception_type": type(e).__name__,
                "duration": duration,
                "failed_at": datetime.now().isoformat()
            })

            return False

    def run_all_experiments(self, continue_on_failure: bool = True):
        """
        Run all experiments sequentially

        Args:
            continue_on_failure: If True, continue running even if some experiments fail
        """
        print(f"\n Starting Industrial Training Pipeline")
        print(f" Total experiments to run: {len(self.experiments)}")

        for i, config in enumerate(self.experiments, 1):
            print(f"\nProgress: {i}/{len(self.experiments)}")

            success = self.run_single_experiment(config)

            if not success and not continue_on_failure:
                print(f"\nStopping pipeline due to failure")
                break

            # Brief pause between experiments
            if i < len(self.experiments):
                print(f"\n Pausing 10s before next experiment...")
                time.sleep(10)

        # Final summary
        self.print_summary()
        self.save_final_report()

    def print_summary(self):
        """Print final pipeline summary"""
        print(f"\n{'='*80}")
        print(f"PIPELINE EXECUTION SUMMARY")
        print(f"{'='*80}")

        total = len(self.experiments) + len(self.completed_experiments)
        completed = len(self.completed_experiments)
        failed = len(self.failed_experiments)
        skipped = total - completed - failed

        print(f"Completed: {completed}/{total}")
        print(f"Failed: {failed}/{total}")
        if skipped > 0:
            print(f"Skipped (already done): {skipped}/{total}")

        if self.failed_experiments:
            print(f"\nFailed Experiments:")
            for fail in self.failed_experiments:
                print(f"  - {fail['experiment_id']}: {fail['error']}")

        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"\nTotal pipeline duration: {duration:.2f}s ({duration/3600:.2f}h)")
        print(f"Pipeline finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    def save_final_report(self):
        """Save comprehensive final report"""
        report_file = self.workspace_dir / f"pipeline_report_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "pipeline_start": self.start_time.isoformat(),
            "pipeline_end": datetime.now().isoformat(),
            "total_duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "experiments_total": len(self.experiments) + len(self.completed_experiments),
            "experiments_completed": len(self.completed_experiments),
            "experiments_failed": len(self.failed_experiments),
            "completed_list": self.completed_experiments,
            "failed_list": self.failed_experiments,
            "configuration": {
                "base_models": list(self.BASE_MODELS.keys()),
                "datasets": list(self.DATASETS.keys()),
                "batch_sizes": self.BATCH_SIZES,
            }
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nFinal report saved to: {report_file}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Industrial Pipeline for Massive LoRA Training"
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="experiments",
        help="Workspace directory for all experiments"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate execution without actually training"
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from a previous pipeline run (provide experiment plan file)"
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop pipeline if any experiment fails"
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = IndustrialPipeline(
        workspace_dir=args.workspace,
        dry_run=args.dry_run,
        resume_from=args.resume_from
    )

    # Generate experiment configurations
    experiments = pipeline.generate_experiments()

    # Save plan
    pipeline.save_experiment_plan()

    if args.dry_run:
        print("\nDry run mode - no training will be executed")
        print(f"Generated {len(experiments)} experiment configurations")
        return

    # Confirm before starting
    print(f"\nAbout to start {len(experiments)} training experiments")
    response = input("Continue? [y/N]: ")

    if response.lower() != 'y':
        print("Pipeline cancelled")
        return

    # Run all experiments
    pipeline.run_all_experiments(
        continue_on_failure=not args.stop_on_failure
    )


if __name__ == "__main__":
    main()
