import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from tqdm import tqdm
from collections import Counter

# Téléchargement des ressources NLTK nécessaires pour le BLEU score
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# ==========================================
# CONFIGURATION
# ==========================================
DATASETS_TO_TEST = [
    "matis35/RAFT",
    "matis35/SYNT_V2",
    "matis35/SYNT_V4",
    "matis35/cf-synt"
]

# Colonnes à analyser séparément
CODE_COLUMN = "code"
FEEDBACK_COLUMN = "feedback" 

# Taille de l'échantillon pour les calculs (pour éviter d'exploser la RAM/CPU)
# 1000 est statistiquement suffisant pour voir les tendances
SAMPLE_SIZE = 1000 
RANDOM_SEED = 42

MODEL_NAME = 'google/embeddinggemma-300m'
#'Qwen/Qwen3-Embedding-4B'
# ==========================================
# FONCTIONS DE MÉTRIQUES
# ==========================================

def compute_semantic_redundancy(embeddings):
    """
    Calcule la similarité cosinus moyenne entre toutes les paires.
    Plus c'est haut, plus c'est redondant (mauvais).
    """
    # Calcul de la matrice de similarité
    cos_matrix = cosine_similarity(embeddings)
    
    # On masque la diagonale (similarité avec soi-même = 1) pour ne pas fausser la moyenne
    np.fill_diagonal(cos_matrix, np.nan)
    
    # Moyenne des valeurs restantes
    mean_redundancy = np.nanmean(cos_matrix)
    return mean_redundancy

def compute_embedding_variance(embeddings):
    """
    Calcule la variance totale ("l'étalement") du nuage de points.
    Plus c'est haut, plus le dataset couvre de concepts différents (bon).
    """
    # Variance calculée sur chaque dimension, puis sommée
    total_variance = np.var(embeddings, axis=0).sum()
    return total_variance

def compute_self_bleu(texts, sample_n=500):
    """
    Calcule le Self-BLEU-4 score.
    Mesure la diversité lexicale. Si une phrase ressemble trop aux autres.
    Plus c'est haut, moins c'est diversifié (mauvais).
    NOTE : C'est un calcul lourd (N*N), on réduit l'échantillon pour ce test spécifique.
    """
    bleu_scores = []
    smoothing = SmoothingFunction().method1

    # On tokenise tout d'avance
    tokenized_texts = [nltk.word_tokenize(t.lower()) for t in texts[:sample_n]]

    for i in range(len(tokenized_texts)):
        hypothesis = tokenized_texts[i]
        # Les références sont toutes les autres phrases SAUF celle-ci
        references = tokenized_texts[:i] + tokenized_texts[i+1:]

        # On calcule le BLEU-4
        score = sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        bleu_scores.append(score)

    return np.mean(bleu_scores)

def compute_thematic_diversity(dataset, sample):
    """
    Calcule la diversité thématique du dataset.
    Plus il y a de thèmes distincts, mieux c'est.
    """
    # Pour SYNT_V4, on a la colonne 'theme'
    if 'theme' in dataset.column_names:
        themes = Counter([dataset[i]['theme'] for i in sample])
        num_unique_themes = len(themes)
        # Distribution équilibrée ? (entropie normalisée)
        theme_counts = np.array(list(themes.values()))
        probs = theme_counts / theme_counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(themes))
        balance = entropy / max_entropy if max_entropy > 0 else 0
        return num_unique_themes, balance
    else:
        return None, None

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================

def main():
    print(f"Chargement du modèle d'embedding : {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)

    results = []

    for ds_name in tqdm(DATASETS_TO_TEST, desc="Benchmarking Datasets"):
        try:
            # 1. Chargement du dataset
            dataset = load_dataset(ds_name, split="train")

            # Vérification des colonnes
            if CODE_COLUMN not in dataset.column_names or FEEDBACK_COLUMN not in dataset.column_names:
                print(f"  Colonnes manquantes dans {ds_name}. Colonnes dispos: {dataset.column_names}")
                continue

            # 2. Échantillonnage (Shuffle + Select)
            if len(dataset) > SAMPLE_SIZE:
                sample_indices = list(range(SAMPLE_SIZE))
                dataset_sample = dataset.shuffle(seed=RANDOM_SEED).select(sample_indices)
            else:
                sample_indices = list(range(len(dataset)))
                dataset_sample = dataset

            codes = dataset_sample[CODE_COLUMN]
            feedbacks = dataset_sample[FEEDBACK_COLUMN]

            # Nettoyage basique (supprimer les None ou vides)
            codes = [c for c in codes if c and len(c.strip()) > 0]
            feedbacks = [f for f in feedbacks if f and len(f.strip()) > 0]

            # 3. Encodage Vectoriel SÉPARÉ pour code et feedback
            print(f"\n  Encodage de {len(codes)} codes...")
            code_embeddings = model.encode(codes, show_progress_bar=False)
            print(f"  Encodage de {len(feedbacks)} feedbacks...")
            feedback_embeddings = model.encode(feedbacks, show_progress_bar=False)

            # 4. Métriques SÉPARÉES
            # --- CODE METRICS ---
            code_redundancy = compute_semantic_redundancy(code_embeddings)
            code_variance = compute_embedding_variance(code_embeddings)

            # --- FEEDBACK METRICS ---
            feedback_redundancy = compute_semantic_redundancy(feedback_embeddings)
            feedback_variance = compute_embedding_variance(feedback_embeddings)
            feedback_self_bleu = compute_self_bleu(feedbacks, sample_n=300)

            # --- THEMATIC DIVERSITY ---
            num_themes, theme_balance = compute_thematic_diversity(dataset, sample_indices)

            results.append({
                "Dataset": ds_name.replace("matis35/", ""),
                "Nb Exemples": len(dataset),
                "Code Diversity (↑)": round(code_variance, 3),
                "Code Redundancy (↓)": round(code_redundancy, 3),
                "Feedback Diversity (↑)": round(feedback_variance, 3),
                "Feedback Redundancy (↓)": round(feedback_redundancy, 3),
                "Feedback Self-BLEU (↓)": round(feedback_self_bleu, 3),
                "Num Themes": num_themes if num_themes else "-",
                "Theme Balance (↑)": round(theme_balance, 3) if theme_balance else "-"
            })

        except Exception as e:
            print(f" Erreur sur {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    # ==========================================
    # AFFICHAGE & EXPORT
    # ==========================================
    df = pd.DataFrame(results)

    print("\n\n" + "="*80)
    print(" RÉSULTATS DU BENCHMARK COMPARATIF - CODE vs FEEDBACK")
    print("="*80)
    print("Légende :")
    print(" - Code Diversity (↑)      : Variance des embeddings de code (+ = meilleur)")
    print(" - Code Redundancy (↓)     : Similarité moyenne entre codes (- = meilleur)")
    print(" - Feedback Diversity (↑)  : Variance des feedbacks (+ = meilleur)")
    print(" - Feedback Redundancy (↓) : Cohérence structurelle (optimal ~0.45-0.52)")
    print(" - Feedback Self-BLEU (↓)  : Répétition lexicale (- = meilleur)")
    print(" - Num Themes              : Nombre de thèmes algorithmiques distincts")
    print(" - Theme Balance (↑)       : Équilibre de distribution (1.0 = parfait)")
    print("-" * 80)
    print(df.to_string(index=False))

    # Export CSV pour ton rapport LaTeX
    df.to_csv("benchmark_results.csv", index=False)
    print("\n Résultats sauvegardés dans 'benchmark_results.csv'")

    # Analyse qualitative
    print("\n" + "="*80)
    print(" INTERPRÉTATION POUR L'APPRENTISSAGE CONTRASTIF")
    print("="*80)
    print("Pour l'entraînement contrastif, l'idéal est :")
    print("   HAUTE diversité de CODE (variance élevée)")
    print("   COHÉRENCE de FEEDBACK (redondance modérée ~0.45-0.52)")
    print("   DIVERSITÉ thématique (nombre de thèmes élevé)")
    print("\nUne redondance de feedback MODÉRÉE est BÉNÉFIQUE car elle garantit")
    print("une structure homogène qui facilite l'apprentissage contrastif.")

if __name__ == "__main__":
    main()