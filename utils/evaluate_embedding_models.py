from typing import List, Dict, Tuple, Union
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Set clean style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Models to evaluate (from config.yml)
MODELS_TO_EVALUATE = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "google/embeddinggemma-300m",
    "Salesforce/SFR-Embedding-Code-400M_R",
    "intfloat/multilingual-e5-large",
]


def load_dataset(file_path: str, max_samples: int = 100) -> List[Dict]:
    """Charge l'ensemble de données JSONL."""
    print(f"Loading dataset from {file_path}...")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if len(data) >= max_samples:
                    break
    print(f"✓ Loaded {len(data)} triplets")
    return data


def compute_embedding_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Calcule la distance cosinus entre deux embeddings (1 - similarité cosinus)."""
    # Reshape est nécessaire si l'entrée est un vecteur 1D
    sim = cosine_similarity(emb1.reshape(1, -1), emb2.reshape(1, -1))[0][0]
    return 1 - sim


def apply_prompt(text: str, model_name: str, input_type: str) -> str:
    """
    Applique le prompt d'alignement spécifique au modèle et au type d'input.
    Input_type est 'query' (Code) ou 'passage' (Feedback).
    """

    # 1. BGE (Général et Code)
    # BGE utilise "Represent this query..." ou "Represent this document..."
    if "bge" in model_name.lower():
        if input_type == 'query':
            return "Represent this query for retrieving relevant documents: " + text
        elif input_type == 'passage':
            return "Represent this document for retrieval: " + text

    # 2. Multilingual E5
    # E5 utilise "query: " et "passage: "
    elif "e5" in model_name.lower():
        if input_type == 'query':
            return "query: " + text
        elif input_type == 'passage':
            return "passage: " + text

    # 3. EmbeddingGemma (et autres modèles Google récents)
    # Utilise souvent "query: " et "passage: "
    elif "gemma" in model_name.lower():
        if input_type == 'query':
            return "query: " + text
        elif input_type == 'passage':
            return "passage: " + text

    # 4. Autres (MiniLM, SFR, GraphCodeBert) : Pas de prompt nécessaire.
    return text


def evaluate_model_on_dataset(
    model: SentenceTransformer,
    dataset: List[Dict],
    model_name: str
) -> Dict:
    """
    Évalue un modèle sur l'ensemble de données.

    Returns:
        - avg_pos_distance: Distance moyenne Code (Query) vs Feedback Positif (Passage)
        - avg_neg_distance: Distance moyenne Code (Query) vs Feedbacks Négatifs (Passage)
        - separation_margin: avg_neg_distance - avg_pos_distance (plus haut est mieux)
    """
    print(f"  Computing embeddings for {len(dataset)} triplets...")

    all_pos_distances = []
    all_neg_distances = []

    for i, item in enumerate(dataset):
        if (i + 1) % 20 == 0:
            print(f"    Progress: {i+1}/{len(dataset)}")

        code = item.get('code_snippet', '')
        positive = item.get('conceptual_feedback', '')
        negatives = item.get('negative_feedbacks', [])

        if not code or not positive or not negatives:
            continue

        # --- CORRECTION : Le CODE est la QUERY, les FEEDBACKS sont les PASSAGES ---
        code_prompted = apply_prompt(code, model_name, 'query')
        
        # Les Feedbacks (Positifs et Négatifs) sont les PASSAGES
        pos_prompted = apply_prompt(positive, model_name, 'passage')
        
        # Encode
        code_emb = model.encode(code_prompted, convert_to_numpy=True)
        pos_emb = model.encode(pos_prompted, convert_to_numpy=True)

        # Distance to positive
        pos_dist = compute_embedding_distance(code_emb, pos_emb)
        all_pos_distances.append(pos_dist)

        # Distance to negatives
        for neg in negatives:
            neg_prompted = apply_prompt(neg, model_name, 'passage') # Prompt négatif
            neg_emb = model.encode(neg_prompted, convert_to_numpy=True)
            neg_dist = compute_embedding_distance(code_emb, neg_emb)
            all_neg_distances.append(neg_dist)

    avg_pos = np.mean(all_pos_distances)
    avg_neg = np.mean(all_neg_distances)
    separation_margin = avg_neg - avg_pos

    results = {
        'avg_pos_distance': avg_pos,
        'avg_neg_distance': avg_neg,
        'separation_margin': separation_margin,
        'all_pos_distances': all_pos_distances,
        'all_neg_distances': all_neg_distances,
        'median_pos_distance': np.median(all_pos_distances),
        'median_neg_distance': np.median(all_neg_distances),
    }

    print(f"  ✓ Positive distance: {avg_pos:.4f} (median: {results['median_pos_distance']:.4f})")
    print(f"  ✓ Negative distance: {avg_neg:.4f} (median: {results['median_neg_distance']:.4f})")
    print(f"  ✓ Separation margin: {separation_margin:.4f}")

    return results


def evaluate_all_models(dataset: List[Dict]) -> Dict[str, Dict]:
    """Évalue tous les modèles sur l'ensemble de données"""
    results = {}

    for model_name in MODELS_TO_EVALUATE:
        print(f"\n{'='*60}")
        print(f"📊 Evaluating: {model_name}")
        print(f"{'='*60}")

        try:
            # Load model
            print(f"⏳ Loading model...")
            # Certains modèles nécessitent trust_remote_code=True
            if "Salesforce" in model_name or "SFR" in model_name:
                model = SentenceTransformer(model_name, trust_remote_code=True)
            else:
                model = SentenceTransformer(model_name)
            print(f"✓ Model loaded")

            # Evaluate
            # Passe le nom du modèle pour permettre l'application des prompts
            model_results = evaluate_model_on_dataset(model, dataset, model_name)
            results[model_name] = model_results

        except Exception as e:
            print(f"❌ Error with {model_name}: {e}")
            continue

    return results


def create_visualizations(results: Dict[str, Dict], output_dir: Path):
    """Crée des visualisations claires et informatives"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data
    model_names = [name.split('/')[-1] for name in results.keys()]  # Noms courts
    full_names = list(results.keys())
    separation_margins = [results[name]['separation_margin'] for name in full_names]
    avg_pos_distances = [results[name]['avg_pos_distance'] for name in full_names]
    avg_neg_distances = [results[name]['avg_neg_distance'] for name in full_names]

    # Figure 1: Separation Margin (la métrique clé)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(model_names, separation_margins, color='steelblue', alpha=0.8)

    # Colore le meilleur modèle en vert
    best_idx = np.argmax(separation_margins)
    bars[best_idx].set_color('forestgreen')
    bars[best_idx].set_alpha(1.0)

    ax.set_xlabel('Marge de Séparation (plus haut est mieux)', fontsize=12, fontweight='bold')
    ax.set_title("Capacité de Séparation Code-Feedback des Modèles d'Embedding",
                 fontsize=14, fontweight='bold', pad=20)
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.grid(axis='x', alpha=0.3)

    # Ajoute les étiquettes de valeur
    for i, (bar, val) in enumerate(zip(bars, separation_margins)):
        label = f'{val:.4f}'
        if i == best_idx:
            label = f'⭐ {label}'
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2, label,
                va='center', fontsize=10, fontweight='bold' if i == best_idx else 'normal')

    plt.tight_layout()
    plt.savefig(output_dir / '1_separation_margin.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '1_separation_margin.png'}")
    plt.close()

    # Figure 2: Distances Positives vs Négatives
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, (model_name, full_name) in enumerate(zip(model_names, full_names)):
        color = 'forestgreen' if i == best_idx else 'steelblue'
        marker = '*' if i == best_idx else 'o'
        size = 300 if i == best_idx else 150
        alpha = 1.0 if i == best_idx else 0.7

        ax.scatter(avg_pos_distances[i], avg_neg_distances[i],
                  s=size, alpha=alpha, color=color, marker=marker,
                  edgecolors='black', linewidths=1.5,
                  label=model_name, zorder=10 if i == best_idx else 5)

    # Ligne diagonale (où pos_dist = neg_dist)
    min_val = min(min(avg_pos_distances), min(avg_neg_distances)) - 0.02
    max_val = max(max(avg_pos_distances), max(avg_neg_distances)) + 0.02
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, linewidth=2,
            label='Pas de séparation')

    ax.set_xlabel('Distance Moyenne: Code → Feedback Positif\n(plus faible est mieux)',
                  fontsize=12, fontweight='bold')
    ax.set_ylabel('Distance Moyenne: Code → Feedbacks Négatifs\n(plus élevée est mieux)',
                  fontsize=12, fontweight='bold')
    ax.set_title("Espace de Distance : Les modèles idéaux sont dans la région 'haut-gauche'",
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # Ajout des étiquettes de région
    ax.text(0.05, 0.95, 'IDÉAL\n(pos faible, neg élevé)',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / '2_distance_space.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '2_distance_space.png'}")
    plt.close()

    # Figure 3: Comparaison de la Distribution (boîtes à moustaches)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Distribution des distances positives
    pos_data = [results[name]['all_pos_distances'] for name in full_names]
    bp1 = ax1.boxplot(pos_data, labels=model_names, patch_artist=True, vert=False)
    for i, patch in enumerate(bp1['boxes']):
        color = 'forestgreen' if i == best_idx else 'steelblue'
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax1.set_xlabel('Distance', fontsize=12, fontweight='bold')
    ax1.set_title('Distance Code → Feedback Positif\n(plus faible est mieux)',
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)

    # Distribution des distances négatives
    neg_data = [results[name]['all_neg_distances'] for name in full_names]
    bp2 = ax2.boxplot(neg_data, labels=model_names, patch_artist=True, vert=False)
    for i, patch in enumerate(bp2['boxes']):
        color = 'forestgreen' if i == best_idx else 'coral'
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xlabel('Distance', fontsize=12, fontweight='bold')
    ax2.set_title('Distance Code → Feedbacks Négatifs\n(plus élevée est mieux)',
                  fontsize=12, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '3_distance_distributions.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '3_distance_distributions.png'}")
    plt.close()

    # Figure 4: Comparaison Récapitulative
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(model_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, avg_pos_distances, width, label='Dist Pos (↓)',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, avg_neg_distances, width, label='Dist Neg (↑)',
                   color='coral', alpha=0.8)

    # Met en évidence le meilleur modèle
    bars1[best_idx].set_color('darkgreen')
    bars2[best_idx].set_color('forestgreen')

    ax.set_ylabel('Distance Moyenne', fontsize=12, fontweight='bold')
    ax.set_title('Comparaison des Modèles : Distances Positives vs Négatives',
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / '4_summary_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / '4_summary_comparison.png'}")
    plt.close()

    # Génération du rapport textuel (inchangée)
    # ...

def generate_report(results: Dict[str, Dict], output_dir: Path):
    """Génère un rapport textuel avec les classements"""

    # Trie par marge de séparation
    sorted_models = sorted(results.items(),
                          key=lambda x: x[1]['separation_margin'],
                          reverse=True)

    report_path = output_dir / 'evaluation_report.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RAPPORT D'ÉVALUATION DES MODÈLES D'EMBEDDING\n")
        f.write("="*80 + "\n\n")

        f.write("OBJECTIF:\n")
        f.write("Évaluer la capacité naturelle des modèles (sans fine-tuning) à :\n")
        f.write("  1. Rapprocher le code du feedback positif (distance FAIBLE)\n")
        f.write("  2. Séparer le code des feedbacks négatifs (distance ÉLEVÉE)\n")
        f.write("  3. Maximiser la marge de séparation = distance_neg - distance_pos\n\n")

        f.write("-"*80 + "\n")
        f.write("CLASSEMENTS (par Marge de Séparation):\n")
        f.write("-"*80 + "\n\n")

        for rank, (model_name, model_results) in enumerate(sorted_models, 1):
            short_name = model_name.split('/')[-1]
            f.write(f"#{rank} {short_name}\n")
            f.write(f"   Nom complet: {model_name}\n")
            f.write(f"   Marge de Séparation: {model_results['separation_margin']:.6f}\n")
            f.write(f"   Distance Positive Moyenne: {model_results['avg_pos_distance']:.6f}\n")
            f.write(f"   Distance Négative Moyenne: {model_results['avg_neg_distance']:.6f}\n")
            f.write(f"   Médiane Distance Positive: {model_results['median_pos_distance']:.6f}\n")
            f.write(f"   Médiane Distance Négative: {model_results['median_neg_distance']:.6f}\n")

            if rank == 1:
                f.write(f"   ⭐ MEILLEUR MODÈLE - Recommandé pour le fine-tuning\n")
            f.write("\n")

        f.write("-"*80 + "\n")
        f.write("RECOMMANDATION:\n")
        f.write("-"*80 + "\n")
        best_model = sorted_models[0][0]
        best_results = sorted_models[0][1]
        f.write(f"\n✓ Meilleur Modèle: {best_model}\n")
        f.write(f"\nCe modèle démontre la meilleure séparation naturelle entre le code et les\n")
        f.write(f"feedbacks négatifs tout en maintenant la proximité avec le feedback positif.\n")
        f.write(f"\nMarge de Séparation: {best_results['separation_margin']:.6f}\n")
        f.write(f"Ce modèle est le meilleur candidat pour le fine-tuning.\n\n")

    print(f"\n✓ Saved: {report_path}")

    # Affiche également dans la console
    print("\n" + "="*80)
    print("CLASSEMENTS FINALS:")
    print("="*80)
    for rank, (model_name, model_results) in enumerate(sorted_models, 1):
        short_name = model_name.split('/')[-1]
        marker = "⭐" if rank == 1 else f"#{rank}"
        print(f"{marker} {short_name}")
        print(f"   Marge de Séparation: {model_results['separation_margin']:.6f}")
        print()

    print("\n🏆 MODÈLE RECOMMANDÉ POUR LE FINE-TUNING:")
    print(f"   {sorted_models[0][0]}")
    print(f"   Marge de Séparation: {sorted_models[0][1]['separation_margin']:.6f}")


def main():
    """Pipeline d'évaluation principal"""
    print("="*80)
    print("ÉVALUATION DES MODÈLES D'EMBEDDING")
    print("="*80)
    print("\nObjectif : Évaluer la capacité de séparation naturelle des modèles (sans fine-tuning)")
    print(f"\nModèles à évaluer : {len(MODELS_TO_EVALUATE)}")
    for model in MODELS_TO_EVALUATE:
        print(f"  • {model}")

    # Chemins
    project_root = Path(__file__).parent.parent
    dataset_path = project_root / "data" / "test.jsonl"
    output_dir = project_root / "data" / "model_evaluation"

    # Vérifie si l'ensemble de données existe
    if not dataset_path.exists():
        print(f"\n❌ Ensemble de données non trouvé : {dataset_path}")
        print("Veuillez fournir un chemin d'ensemble de données valide avec des triplets code/feedback.")
        return

    # Charge l'ensemble de données (augmenté à 500 échantillons pour une évaluation plus robuste)
    print("\n" + "-"*80)
    dataset = load_dataset(str(dataset_path), max_samples=500)

    if not dataset:
        print("❌ Aucune donnée chargée. Sortie.")
        return

    # Évalue tous les modèles
    print("\n" + "-"*80)
    print("Démarrage de l'évaluation...")
    results = evaluate_all_models(dataset)

    if not results:
        print("\n❌ Aucun modèle évalué avec succès. Sortie.")
        return

    # Crée les visualisations
    print("\n" + "-"*80)
    print("Génération des visualisations...")
    create_visualizations(results, output_dir)

    # Génère le rapport
    print("\n" + "-"*80)
    print("Génération du rapport...")
    generate_report(results, output_dir)

    print("\n" + "="*80)
    print("✓ ÉVALUATION TERMINÉE")
    print("="*80)
    print(f"\nTous les résultats enregistrés dans : {output_dir}")
    print("\nFichiers générés:")
    print("  • 1_separation_margin.png")
    print("  • 2_distance_space.png")
    print("  • 3_distance_distributions.png")
    print("  • 4_summary_comparison.png")
    print("  • evaluation_report.txt")


if __name__ == "__main__":
    main()