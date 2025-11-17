#!/usr/bin/env python3
"""
Analyze different embedding models to see if tight clustering is model-dependent.

Hypothesis: Maybe current embedding models don't capture enough nuance in C feedback.
A more powerful model might provide better natural separation.

Models tested:
1. Google EmbeddingGemma (300M params)
2. Microsoft GraphCodeBERT (125M params, code-specialized)
3. Salesforce SFR-Embedding-Code (400M params, code-specialized)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_feedbacks(jsonl_path, max_samples=500):
    """Load feedbacks from JSONL dataset."""
    print(f"📂 Loading feedbacks from {jsonl_path}")

    with open(jsonl_path, 'r') as f:
        examples = [json.loads(line) for i, line in enumerate(f) if i < max_samples]

    feedbacks = [ex['conceptual_feedback'] for ex in examples]
    print(f"   Loaded {len(feedbacks)} feedbacks")
    return feedbacks


def analyze_model(model_name, feedbacks):
    """Analyze clustering for a specific embedding model."""
    print(f"\n🤖 Analyzing model: {model_name}")

    # Load model
    print(f"   Loading model...")
    model = SentenceTransformer(model_name, trust_remote_code=True)

    # Encode feedbacks
    print(f"   Encoding {len(feedbacks)} feedbacks...")
    embeddings = model.encode(
        feedbacks,
        show_progress_bar=True,
        convert_to_numpy=True,
        batch_size=32
    )

    # Compute pairwise similarities
    print(f"   Computing similarities...")
    similarities = cosine_similarity(embeddings)

    # Remove diagonal (self-similarity)
    np.fill_diagonal(similarities, np.nan)
    similarities_flat = similarities[~np.isnan(similarities)]

    # Statistics
    stats = {
        'mean': similarities_flat.mean(),
        'median': np.median(similarities_flat),
        'std': similarities_flat.std(),
        'min': similarities_flat.min(),
        'max': similarities_flat.max(),
        'q25': np.percentile(similarities_flat, 25),
        'q75': np.percentile(similarities_flat, 75),
    }

    # t-SNE for visualization
    print(f"   Computing t-SNE projection...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    return {
        'embeddings': embeddings,
        'embeddings_2d': embeddings_2d,
        'similarities': similarities_flat,
        'stats': stats,
    }


def plot_model_analysis(model_name, results, output_path):
    """Create comprehensive visualization for a model."""
    fig = plt.figure(figsize=(20, 6))

    # Clean model name for title
    clean_name = model_name.split('/')[-1]

    # Subplot 1: t-SNE scatter plot
    ax1 = plt.subplot(1, 3, 1)
    scatter = ax1.scatter(
        results['embeddings_2d'][:, 0],
        results['embeddings_2d'][:, 1],
        c=range(len(results['embeddings_2d'])),
        cmap='viridis',
        alpha=0.6,
        s=50,
        edgecolors='white',
        linewidth=0.5
    )
    ax1.set_title(f't-SNE Visualization\n{clean_name}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Dimension 1', fontsize=11)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Feedback Index', fontsize=10)

    # Subplot 2: Similarity distribution histogram
    ax2 = plt.subplot(1, 3, 2)
    stats = results['stats']

    n, bins, patches = ax2.hist(
        results['similarities'],
        bins=50,
        color='steelblue',
        alpha=0.7,
        edgecolor='black',
        linewidth=0.5
    )

    # Add vertical lines for statistics
    ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean: {stats['mean']:.3f}")
    ax2.axvline(stats['median'], color='orange', linestyle='--', linewidth=2, label=f"Median: {stats['median']:.3f}")
    ax2.axvline(0.5, color='green', linestyle=':', linewidth=2, label='Margin: 0.50')

    ax2.set_title(f'Similarity Distribution\n{clean_name}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Cosine Similarity', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Box plot with statistics
    ax3 = plt.subplot(1, 3, 3)

    # Create box plot
    bp = ax3.boxplot(
        [results['similarities']],
        labels=['Similarities'],
        patch_artist=True,
        widths=0.5,
        boxprops=dict(facecolor='lightblue', alpha=0.7),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5)
    )

    # Add statistics text
    text_stats = f"""Statistics:
Mean:   {stats['mean']:.4f}
Median: {stats['median']:.4f}
Std:    {stats['std']:.4f}
Min:    {stats['min']:.4f}
Max:    {stats['max']:.4f}
Q25:    {stats['q25']:.4f}
Q75:    {stats['q75']:.4f}

Margin: 0.5000
Above margin: {(results['similarities'] > 0.5).sum() / len(results['similarities']) * 100:.1f}%
"""

    ax3.text(
        1.5, stats['median'],
        text_stats,
        fontsize=10,
        verticalalignment='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    # Add reference line for margin
    ax3.axhline(0.5, color='green', linestyle=':', linewidth=2, label='Margin (0.5)')

    ax3.set_title(f'Statistics Summary\n{clean_name}', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Cosine Similarity', fontsize=11)
    ax3.set_ylim([0, 1])
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_path}")
    plt.close()


def create_comparison_plot(all_results, output_path):
    """Create comparison plot across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    models_clean = [name.split('/')[-1] for name in all_results.keys()]

    for idx, (model_name, color) in enumerate(zip(all_results.keys(), colors)):
        results = all_results[model_name]
        clean_name = model_name.split('/')[-1]

        # Plot histogram
        axes[idx].hist(
            results['similarities'],
            bins=40,
            color=color,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )

        stats = results['stats']
        axes[idx].axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                         label=f"Mean: {stats['mean']:.3f}")
        axes[idx].axvline(0.5, color='green', linestyle=':', linewidth=2,
                         label='Margin: 0.50')

        axes[idx].set_title(clean_name, fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Cosine Similarity', fontsize=11)
        axes[idx].set_ylabel('Frequency', fontsize=11)
        axes[idx].legend(fontsize=10)
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add percentage above margin
        above_margin = (results['similarities'] > 0.5).sum() / len(results['similarities']) * 100
        axes[idx].text(
            0.05, 0.95,
            f">{0.5:.1f}: {above_margin:.1f}%",
            transform=axes[idx].transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7)
        )

    plt.suptitle('Comparison: Similarity Distributions Across Embedding Models',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {output_path}")
    plt.close()


def print_comparison_table(all_results):
    """Print comparison table of all models."""
    print("\n" + "="*100)
    print("📊 COMPARISON TABLE: Embedding Models")
    print("="*100)
    print(f"{'Model':<40} {'Mean':<8} {'Median':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'>0.5':<8}")
    print("-"*100)

    for model_name, results in all_results.items():
        clean_name = model_name.split('/')[-1]
        stats = results['stats']
        above_margin = (results['similarities'] > 0.5).sum() / len(results['similarities']) * 100

        print(f"{clean_name:<40} {stats['mean']:<8.4f} {stats['median']:<8.4f} "
              f"{stats['std']:<8.4f} {stats['min']:<8.4f} {stats['max']:<8.4f} "
              f"{above_margin:<8.1f}%")

    print("="*100)


def main():
    print("="*100)
    print("🔬 EMBEDDING MODEL ANALYSIS: Testing Clustering Hypothesis")
    print("="*100)
    print("\nHypothesis: Current embedding models may not capture enough nuance.")
    print("Testing if more powerful/specialized models provide better separation.\n")

    # Models to test
    models = {
        'google/embeddinggemma-300m': 'EmbeddingGemma (300M)',
        'microsoft/graphcodebert-base': 'GraphCodeBERT (125M, code-specialized)',
        'Salesforce/SFR-Embedding-Code-400M_R': 'Salesforce SFR-Code (400M, code-specialized)',
    }

    # Load feedbacks
    feedbacks = load_feedbacks('data/Exp-002-llama3B_v2.jsonl', max_samples=500)

    # Analyze each model
    all_results = {}

    for model_name, description in models.items():
        try:
            results = analyze_model(model_name, feedbacks)
            all_results[model_name] = results

            # Create individual plot
            output_path = f"visualization/{model_name.split('/')[-1]}_analysis.png"
            plot_model_analysis(model_name, results, output_path)

        except Exception as e:
            print(f"   ❌ Error with {model_name}: {e}")
            continue

    # Create comparison plot
    if len(all_results) > 1:
        create_comparison_plot(all_results, 'visualization/model_comparison.png')

    # Print comparison table
    print_comparison_table(all_results)

    # Analysis and conclusion
    print("\n" + "="*100)
    print("💡 ANALYSIS & CONCLUSIONS")
    print("="*100)

    # Find best and worst models
    best_model = min(all_results.items(), key=lambda x: x[1]['stats']['mean'])
    worst_model = max(all_results.items(), key=lambda x: x[1]['stats']['mean'])

    print(f"\n✅ BEST Model (lowest mean similarity):")
    print(f"   {best_model[0].split('/')[-1]}")
    print(f"   Mean similarity: {best_model[1]['stats']['mean']:.4f}")
    print(f"   Pairs > 0.5: {(best_model[1]['similarities'] > 0.5).sum() / len(best_model[1]['similarities']) * 100:.1f}%")

    print(f"\n❌ WORST Model (highest mean similarity):")
    print(f"   {worst_model[0].split('/')[-1]}")
    print(f"   Mean similarity: {worst_model[1]['stats']['mean']:.4f}")
    print(f"   Pairs > 0.5: {(worst_model[1]['similarities'] > 0.5).sum() / len(worst_model[1]['similarities']) * 100:.1f}%")

    # Check if hypothesis is validated
    similarity_range = worst_model[1]['stats']['mean'] - best_model[1]['stats']['mean']
    print(f"\n📈 Similarity range across models: {similarity_range:.4f}")

    if similarity_range > 0.1:
        print("\n✅ HYPOTHESIS VALIDATED: Model choice significantly impacts clustering!")
        print(f"   Using {best_model[0].split('/')[-1]} could improve training convergence.")
    else:
        print("\n⚠️  HYPOTHESIS PARTIALLY VALIDATED: Some improvement but limited.")
        print(f"   All models show tight clustering (mean > 0.5).")
        print(f"   Fundamental issue: LLM-generated feedbacks are semantically similar.")
        print(f"   Recommendation: Diversify feedback generation prompts.")

    print("\n" + "="*100)
    print("✅ Analysis complete! Check visualization/ folder for plots.")
    print("="*100)


if __name__ == "__main__":
    main()
