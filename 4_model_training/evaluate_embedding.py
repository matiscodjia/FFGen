from openai import OpenAI
import numpy as np
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer, util
import json
import torch 
import os # Pour une meilleure gestion de la clé API OpenAI

# --- 1. Définition des Fonctions et des Codes ---

# Vos deux fragments de code en langage C
code_1 = "int my_compute_factorial_rec(int nb){   if (nb == 0)        return 1;    if (nb < 0)        return 0;    return my_compute_factorial_rec(nb - 1) * nb;}"
code_4 = "int my_compute_power( int nb, int p){    int result = nb;    int i = 0;    if (p < 0) {        return 0;    } else if (p == 0) {        return 1;    }    while (p > 1) {        result = result * (nb);        p --;    }    return result;}"
codes = [code_1, code_4]

# Liste des modèles Hugging Face à tester
models = [
    "Salesforce/SFR-Embedding-Code-400M_R",
    "microsoft/graphcodebert-base",
    "jinaai/jina-embeddings-v2-base-code",
    "nomic-ai/nomic-embed-text-v1.5", 
    "google/embeddinggemma-300m",
    "BAAI/bge-code-v1"
]

# --- 2. Fonction de Calcul de Similarité (Sentence Transformers) ---

def compute_similarity(model_name: str, code_pairs: list) -> float:
    """
    Charge un modèle SentenceTransformer et calcule la similarité cosinus 
    entre les deux fragments de code. Retourne une valeur de similarité.
    """
    try:
        # Détection du périphérique (CUDA > MPS > CPU)
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Charger le modèle
        model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        print(f"Chargement réussi du modèle sur {device}.")
        
        # Encoder les deux fragments de code
        embeddings = model.encode(code_pairs, convert_to_tensor=True, show_progress_bar=False)
        
        # Calculer la similarité cosinus (util.cos_sim)
        similarity_matrix = util.cos_sim(embeddings, embeddings)
        
        # Récupérer la similarité entre le code 1 et le code 4
        similarity = similarity_matrix[0][1].item()
        
        return similarity
    
    except Exception as e:
        print(f"Erreur lors du traitement du modèle {model_name}: {e}")
        return None

# --- 3. Fonction pour Tester Tous les Modèles et Stocker les Résultats ---

def sensitivity_test(models: list, code_pairs: list) -> dict:
    """
    Teste tous les modèles Hugging Face et stocke les scores de similarité/distance.
    """
    results_dict = {}
    print("--- Début des Tests de Sensibilité Hugging Face ---")
    for model in models:
        print(f"\n🚀 Modèle d'embedding: {model}")
        similarity_score = compute_similarity(model, code_pairs)
        
        if similarity_score is not None:
            # Convertir la similarité en distance (Distance = 1 - Similarité)
            distance_score = 1 - similarity_score
            results_dict[model] = {
                "similarité_cosinus": similarity_score,
                "distance_cosinus": distance_score
            }
            print(f"   -> Distance Cosinus: {distance_score:.4f}")
            
    print("\n--- Tests Terminés (Hugging Face) ---")
    return results_dict

# --- 4. Fonction de Calcul pour le Modèle OpenAI (API Externe) ---

def compute_openai_distance(c1: str, c2: str) -> float:
    """
    Calcule la distance cosinus en utilisant l'API OpenAI et text-embedding-3-small.
    """
    print("\nCalcul de référence (OpenAI - text-embedding-3-small) :")
    try:
        if not os.getenv("OPENAI_API_KEY"):
            print("Erreur: La variable d'environnement OPENAI_API_KEY n'est pas définie.")
            return None
        
        client = OpenAI()
        
        response_1 = client.embeddings.create(input=c1, model="text-embedding-3-small")
        embedding_1_list = response_1.data[0].embedding
        
        response_2 = client.embeddings.create(input=c2, model="text-embedding-3-small")
        embedding_2_list = response_2.data[0].embedding

        # Conversion en NumPy et Reshape (La correction nécessaire)
        embedding_1 = np.array(embedding_1_list).reshape(1, -1) 
        embedding_2 = np.array(embedding_2_list).reshape(1, -1) 
        
        # Calcul de la distance avec sklearn
        distance = pairwise_distances(embedding_1, embedding_2, metric="cosine")[0][0]
        
        print(f"   -> Distance Cosinus (OpenAI) : {distance:.4f}")
        return distance
    
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API OpenAI : {e}")
        return None

# --- 5. Exécution Principale ---

if __name__ == "__main__":
    
    # 5.1 Exécution des tests Hugging Face
    hf_results = sensitivity_test(models, codes)
    
    # 5.2 Exécution du test OpenAI
    openai_distance = compute_openai_distance(code_1, code_4)
    
    # --- Consolidation des Résultats ---
    
    # Ajouter le résultat OpenAI aux résultats finaux
    if openai_distance is not None:
        hf_results["openai/text-embedding-3-small"] = {
            "similarité_cosinus": 1 - openai_distance,
            "distance_cosinus": openai_distance
        }

    # 5.3 Sauvegarde des Résultats
    file_name = 'embedding_sensitivity_results.json'
    with open(file_name, 'w') as fp:
        json.dump(hf_results, fp, indent=4)
    print(f"\nRésultats complets sauvegardés dans '{file_name}'")
    
    # 5.4 Affichage du résumé UNIFIÉ et TRIÉ
    
    print("\n" + "="*70)
    print("Résumé des Distances Cosinus (OpenAI et Hugging Face)")
    print("  -> Plus la distance est proche de 1, plus les codes sont DIFFÉRENTS.")
    print("="*70)
    
    # Préparer les données pour le tri
    sorted_results = []
    for model, scores in hf_results.items():
        sorted_results.append((scores['distance_cosinus'], model))

    # Trier par distance décroissante (le meilleur est celui qui distingue le mieux)
    sorted_results.sort(key=lambda item: item[0], reverse=True)
    
    # Affichage final
    for distance, model in sorted_results:
        # Affichage du nom du modèle et de sa distance alignés
        print(f"{model:<40}: {distance:.4f}")
        
    print("="*70)