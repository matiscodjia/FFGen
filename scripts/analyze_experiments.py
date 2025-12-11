"""
Experiment Analysis and Visualization
======================================

This script analyzes all experiment results and generates comprehensive
visualizations showing:
- Performance vs batch size
- Performance vs dataset quality
- Model comparison
- Training dynamics
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ExperimentAnalyzer:
    """Analyzer for experiment results"""

    def __init__(self, workspace_dir: str = "experiments"):
        self.workspace_dir = Path(workspace_dir)
        self.results_dir = self.workspace_dir / "results"
        self.runs_dir = self.workspace_dir / "runs"
        self.analysis_dir = self.workspace_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True)

        self.experiments_data = []
        self.metrics_data = []

    def load_all_results(self):
        """Load results from all completed experiments"""
        print("📂 Loading experiment results...")

        result_files = list(self.results_dir.glob("*_results.json"))

        for result_file in result_files:
            with open(result_file) as f:
                result = json.load(f)
                exp_id = result['experiment_id']

                # Load summary metrics
                summary_file = self.runs_dir / exp_id / "summary.json"
                if summary_file.exists():
                    with open(summary_file) as f:
                        summary = json.load(f)
                        result['metrics_summary'] = summary

                # Load step-by-step metrics
                metrics_file = self.runs_dir / exp_id / "metrics.jsonl"
                if metrics_file.exists():
                    step_metrics = []
                    with open(metrics_file) as f:
                        for line in f:
                            step_metrics.append(json.loads(line))
                    result['step_metrics'] = step_metrics

                self.experiments_data.append(result)

        print(f"✅ Loaded {len(self.experiments_data)} experiments")

    def parse_experiment_config(self, exp_id: str) -> Dict[str, Any]:
        """Parse experiment ID to extract configuration"""
        # Format: {model}_{dataset}_bs{batch_size}
        parts = exp_id.split('_')

        # Extract model (can be multi-part)
        if 'jina' in exp_id:
            model = 'jina-code-embed'
            idx = exp_id.index('raft')
        elif 'gemma' in exp_id:
            model = 'gemma-embedding-300m'
            idx = exp_id.index('raft')
        else:
            model = 'unknown'
            idx = 0

        # Extract dataset
        if 'raft-ultra-clean' in exp_id:
            dataset = 'raft-ultra-clean'
            dataset_quality = 'high'
        elif 'raft' in exp_id:
            dataset = 'raft'
            dataset_quality = 'standard'
        else:
            dataset = 'unknown'
            dataset_quality = 'unknown'

        # Extract batch size
        bs_part = [p for p in parts if p.startswith('bs')]
        if bs_part:
            batch_size = int(bs_part[0].replace('bs', ''))
        else:
            batch_size = 0

        return {
            'model': model,
            'dataset': dataset,
            'dataset_quality': dataset_quality,
            'batch_size': batch_size
        }

    def create_summary_dataframe(self) -> pd.DataFrame:
        """Create a DataFrame with all experiment summaries"""
        rows = []

        for exp in self.experiments_data:
            exp_id = exp['experiment_id']
            config = self.parse_experiment_config(exp_id)

            metrics = exp.get('metrics_summary', {})

            row = {
                'experiment_id': exp_id,
                'model': config['model'],
                'dataset': config['dataset'],
                'dataset_quality': config['dataset_quality'],
                'batch_size': config['batch_size'],
                'best_loss': metrics.get('best_loss', None),
                'best_accuracy': metrics.get('best_accuracy', None),
                'final_loss': metrics.get('final_loss', None),
                'final_accuracy': metrics.get('final_accuracy', None),
                'total_steps': metrics.get('total_steps', None),
                'total_epochs': metrics.get('total_epochs', None),
                'duration': exp.get('duration_seconds', None),
            }

            rows.append(row)

        df = pd.DataFrame(rows)
        return df

    def plot_batch_size_effect(self, df: pd.DataFrame):
        """Plot performance vs batch size"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Effect of Batch Size on Model Performance', fontsize=16, fontweight='bold')

        metrics = [
            ('best_loss', 'Best Loss', 'lower is better'),
            ('best_accuracy', 'Best Accuracy', 'higher is better'),
            ('final_loss', 'Final Loss', 'lower is better'),
            ('final_accuracy', 'Final Accuracy', 'higher is better'),
        ]

        for idx, (metric, title, direction) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]

            # Plot for each model and dataset combination
            for model in df['model'].unique():
                for dataset_quality in df['dataset_quality'].unique():
                    subset = df[(df['model'] == model) & (df['dataset_quality'] == dataset_quality)]

                    if len(subset) == 0:
                        continue

                    label = f"{model.replace('jina-code-embed', 'Jina').replace('gemma-embedding-300m', 'Gemma')} - {dataset_quality}"
                    marker = 'o' if model == 'jina-code-embed' else 's'
                    linestyle = '-' if dataset_quality == 'high' else '--'

                    ax.plot(subset['batch_size'], subset[metric],
                            marker=marker, linestyle=linestyle, label=label,
                            linewidth=2, markersize=8)

            ax.set_xlabel('Batch Size', fontweight='bold')
            ax.set_ylabel(title, fontweight='bold')
            ax.set_title(f'{title} vs Batch Size ({direction})', fontweight='bold')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xscale('log', base=2)
            ax.set_xticks([32, 64, 128, 256])
            ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        plt.tight_layout()
        output_file = self.analysis_dir / "batch_size_effect.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.close()

    def plot_dataset_quality_effect(self, df: pd.DataFrame):
        """Plot performance comparison between datasets"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Effect of Dataset Quality on Model Performance', fontsize=16, fontweight='bold')

        metrics = [
            ('best_accuracy', 'Best Accuracy (higher is better)'),
            ('best_loss', 'Best Loss (lower is better)')
        ]

        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx]

            # Prepare data for grouped bar chart
            models = df['model'].unique()
            batch_sizes = sorted(df['batch_size'].unique())

            x = np.arange(len(batch_sizes))
            width = 0.15

            for model_idx, model in enumerate(models):
                for quality_idx, quality in enumerate(['standard', 'high']):
                    subset = df[(df['model'] == model) & (df['dataset_quality'] == quality)]
                    subset = subset.sort_values('batch_size')

                    offset = width * (model_idx * 2 + quality_idx - 1.5)

                    label = f"{model.replace('jina-code-embed', 'Jina').replace('gemma-embedding-300m', 'Gemma')} - {'Clean' if quality == 'high' else 'Original'}"
                    color = f"C{model_idx * 2 + quality_idx}"

                    ax.bar(x + offset, subset[metric], width,
                           label=label, color=color, alpha=0.8)

            ax.set_xlabel('Batch Size', fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
            ax.set_title(title, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(batch_sizes)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.analysis_dir / "dataset_quality_effect.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.close()

    def plot_model_comparison(self, df: pd.DataFrame):
        """Compare models across all configurations"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Comparison Across Configurations', fontsize=16, fontweight='bold')

        # 1. Heatmap: Best Accuracy
        ax = axes[0, 0]
        pivot = df.pivot_table(
            values='best_accuracy',
            index=['model', 'dataset_quality'],
            columns='batch_size',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Accuracy'})
        ax.set_title('Best Accuracy Heatmap', fontweight='bold')
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Model & Dataset', fontweight='bold')

        # 2. Heatmap: Best Loss
        ax = axes[0, 1]
        pivot = df.pivot_table(
            values='best_loss',
            index=['model', 'dataset_quality'],
            columns='batch_size',
            aggfunc='mean'
        )
        sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd_r', ax=ax, cbar_kws={'label': 'Loss'})
        ax.set_title('Best Loss Heatmap', fontweight='bold')
        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Model & Dataset', fontweight='bold')

        # 3. Training duration comparison
        ax = axes[1, 0]
        for model in df['model'].unique():
            subset = df[df['model'] == model]
            model_name = model.replace('jina-code-embed', 'Jina').replace('gemma-embedding-300m', 'Gemma')
            ax.scatter(subset['batch_size'], subset['duration'] / 60,
                      label=model_name, s=100, alpha=0.6)

        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Training Duration (minutes)', fontweight='bold')
        ax.set_title('Training Duration vs Batch Size', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks([32, 64, 128, 256])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        # 4. Overall performance score
        ax = axes[1, 1]
        # Normalize metrics and create composite score
        df_norm = df.copy()
        df_norm['accuracy_score'] = (df_norm['best_accuracy'] - df_norm['best_accuracy'].min()) / (df_norm['best_accuracy'].max() - df_norm['best_accuracy'].min())
        df_norm['loss_score'] = 1 - (df_norm['best_loss'] - df_norm['best_loss'].min()) / (df_norm['best_loss'].max() - df_norm['best_loss'].min())
        df_norm['composite_score'] = (df_norm['accuracy_score'] + df_norm['loss_score']) / 2

        for model in df_norm['model'].unique():
            for quality in df_norm['dataset_quality'].unique():
                subset = df_norm[(df_norm['model'] == model) & (df_norm['dataset_quality'] == quality)]
                label = f"{model.replace('jina-code-embed', 'Jina').replace('gemma-embedding-300m', 'Gemma')} - {quality}"
                marker = 'o' if model == 'jina-code-embed' else 's'
                ax.plot(subset['batch_size'], subset['composite_score'],
                       marker=marker, label=label, linewidth=2, markersize=8)

        ax.set_xlabel('Batch Size', fontweight='bold')
        ax.set_ylabel('Composite Performance Score', fontweight='bold')
        ax.set_title('Overall Performance Score (normalized)', fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks([32, 64, 128, 256])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

        plt.tight_layout()
        output_file = self.analysis_dir / "model_comparison.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.close()

    def plot_training_curves(self):
        """Plot training curves for all experiments"""
        print("📈 Generating training curves...")

        # Group by configuration
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('Training Dynamics Across Experiments', fontsize=16, fontweight='bold')

        metrics = ['loss', 'accuracy']
        comparisons = ['batch_size', 'dataset_quality']

        for metric_idx, metric in enumerate(metrics):
            for comp_idx, comparison in enumerate(comparisons):
                ax = axes[metric_idx, comp_idx]

                # Group experiments by comparison dimension
                groups = defaultdict(list)

                for exp in self.experiments_data:
                    if 'step_metrics' not in exp:
                        continue

                    exp_id = exp['experiment_id']
                    config = self.parse_experiment_config(exp_id)

                    # Create group key
                    if comparison == 'batch_size':
                        key = f"BS={config['batch_size']}"
                    else:
                        key = f"{config['dataset_quality']}"

                    groups[key].append({
                        'config': config,
                        'metrics': exp['step_metrics']
                    })

                # Plot each group
                for group_key, group_exps in groups.items():
                    for exp_data in group_exps:
                        steps = [m['step'] for m in exp_data['metrics']]
                        values = [m[metric] for m in exp_data['metrics']]

                        model_label = exp_data['config']['model'].replace('jina-code-embed', 'J').replace('gemma-embedding-300m', 'G')
                        label = f"{group_key} ({model_label})"

                        ax.plot(steps, values, label=label, alpha=0.7, linewidth=1.5)

                ax.set_xlabel('Training Step', fontweight='bold')
                ax.set_ylabel(metric.title(), fontweight='bold')
                ax.set_title(f'{metric.title()} by {comparison.replace("_", " ").title()}', fontweight='bold')
                ax.legend(loc='best', fontsize=7, ncol=2)
                ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.analysis_dir / "training_curves.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"💾 Saved: {output_file}")
        plt.close()

    def generate_summary_report(self, df: pd.DataFrame):
        """Generate a comprehensive text report"""
        report_file = self.analysis_dir / "summary_report.txt"

        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("INDUSTRIAL TRAINING EXPERIMENT SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")

            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-"*80 + "\n")
            f.write(f"Total Experiments: {len(df)}\n")
            f.write(f"Models: {', '.join(df['model'].unique())}\n")
            f.write(f"Datasets: {', '.join(df['dataset'].unique())}\n")
            f.write(f"Batch Sizes: {', '.join(map(str, sorted(df['batch_size'].unique())))}\n")
            f.write(f"\nTotal Training Time: {df['duration'].sum() / 3600:.2f} hours\n")
            f.write(f"Average Training Time: {df['duration'].mean() / 60:.2f} minutes\n\n")

            # Best performing configurations
            f.write("BEST PERFORMING CONFIGURATIONS\n")
            f.write("-"*80 + "\n")

            best_acc = df.loc[df['best_accuracy'].idxmax()]
            f.write(f"\nHighest Accuracy:\n")
            f.write(f"  Experiment: {best_acc['experiment_id']}\n")
            f.write(f"  Model: {best_acc['model']}\n")
            f.write(f"  Dataset: {best_acc['dataset']} ({best_acc['dataset_quality']})\n")
            f.write(f"  Batch Size: {best_acc['batch_size']}\n")
            f.write(f"  Best Accuracy: {best_acc['best_accuracy']:.4f}\n")
            f.write(f"  Best Loss: {best_acc['best_loss']:.4f}\n")

            best_loss = df.loc[df['best_loss'].idxmin()]
            f.write(f"\nLowest Loss:\n")
            f.write(f"  Experiment: {best_loss['experiment_id']}\n")
            f.write(f"  Model: {best_loss['model']}\n")
            f.write(f"  Dataset: {best_loss['dataset']} ({best_loss['dataset_quality']})\n")
            f.write(f"  Batch Size: {best_loss['batch_size']}\n")
            f.write(f"  Best Loss: {best_loss['best_loss']:.4f}\n")
            f.write(f"  Best Accuracy: {best_loss['best_accuracy']:.4f}\n")

            # Effect of batch size
            f.write("\n\nEFFECT OF BATCH SIZE\n")
            f.write("-"*80 + "\n")
            batch_analysis = df.groupby('batch_size').agg({
                'best_accuracy': ['mean', 'std'],
                'best_loss': ['mean', 'std']
            }).round(4)
            f.write(batch_analysis.to_string())

            # Effect of dataset quality
            f.write("\n\nEFFECT OF DATASET QUALITY\n")
            f.write("-"*80 + "\n")
            quality_analysis = df.groupby('dataset_quality').agg({
                'best_accuracy': ['mean', 'std'],
                'best_loss': ['mean', 'std']
            }).round(4)
            f.write(quality_analysis.to_string())

            # Model comparison
            f.write("\n\nMODEL COMPARISON\n")
            f.write("-"*80 + "\n")
            model_analysis = df.groupby('model').agg({
                'best_accuracy': ['mean', 'std', 'max'],
                'best_loss': ['mean', 'std', 'min']
            }).round(4)
            f.write(model_analysis.to_string())

            # Detailed results table
            f.write("\n\nDETAILED RESULTS TABLE\n")
            f.write("-"*80 + "\n")
            results_table = df[['experiment_id', 'model', 'dataset_quality', 'batch_size',
                               'best_accuracy', 'best_loss', 'final_accuracy', 'final_loss']]
            results_table = results_table.sort_values('best_accuracy', ascending=False)
            f.write(results_table.to_string(index=False))

        print(f"💾 Saved: {report_file}")

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("\n🔬 Starting Experiment Analysis...")

        # Load data
        self.load_all_results()

        if not self.experiments_data:
            print("❌ No experiment results found!")
            return

        # Create summary DataFrame
        df = self.create_summary_dataframe()

        # Save raw data
        csv_file = self.analysis_dir / "all_experiments.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved: {csv_file}")

        # Generate visualizations
        print("\n📊 Generating visualizations...")
        self.plot_batch_size_effect(df)
        self.plot_dataset_quality_effect(df)
        self.plot_model_comparison(df)
        self.plot_training_curves()

        # Generate summary report
        print("\n📝 Generating summary report...")
        self.generate_summary_report(df)

        print("\n✅ Analysis complete!")
        print(f"📁 Results saved to: {self.analysis_dir}")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--workspace",
        type=str,
        default="experiments",
        help="Workspace directory containing experiments"
    )

    args = parser.parse_args()

    analyzer = ExperimentAnalyzer(workspace_dir=args.workspace)
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
