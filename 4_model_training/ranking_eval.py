import pandas as pd
import numpy as np
import json
import random
import torch
import os
from tqdm import tqdm # Importation de tqdm

from openai import OpenAI
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import pairwise_distances, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- 1. CONFIGURATION ---

# Définition des modèles à tester
MODELS = [
    "Salesforce/SFR-Embedding-Code-400M_R",
    "microsoft/graphcodebert-base",
    "jinaai/jina-embeddings-v2-base-code",
    "nomic-ai/nomic-embed-text-v1.5",
    "google/embeddinggemma-300m",
    "BAAI/bge-code-v1"
]
OPENAI_MODEL = "text-embedding-3-small"

# Les seuils de similarité pour la classification binaire (Distance Cosinus)
DISTANCE_THRESHOLDS = {
    "Seuil_0.50": 0.50,
    "Seuil_0.70": 0.70,
    "Seuil_0.80": 0.80, 
}

# Chemin d'accès à votre fichier JSONL (À MODIFIER PAR LE CHEMIN RÉEL)
JSONL_FILE_PATH = "/Users/matiscodjia/Dev/06_research/FFGen/data/Exp-001_llama3B_v1.jsonl" 
# --- FIN CONFIGURATION ---


# --- 2. FONCTIONS DE TRAITEMENT DES DONNÉES ---

def load_jsonl(file_path: str) -> List[Dict[str, str]]:
    """
    Charge les données d'un fichier JSONL valide (une ligne = un objet JSON).
    """
    data = []
    print(f"Tentative de lecture du fichier : {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"⚠️ Avertissement : Ligne ignorée due à une erreur JSON : {e}")
                        
    except FileNotFoundError:
        print(f"ERREUR FATALE : Fichier non trouvé à l'emplacement : {file_path}")
        return []
        
    return data

def generate_similarity_dataframe(codes: List[Dict[str, str]]) -> pd.DataFrame:
    """
    Génère un DataFrame de paires de code et de labels (1=Similaire, 0=Différent)
    basé sur l'heuristique du nom de l'exercice.
    """
    pairs: List[Tuple[str, str, int]] = []
    grouped_codes: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    
    for code_entry in codes:
        key = (code_entry['exercise'], code_entry['cpoolday'])
        if key not in grouped_codes:
            grouped_codes[key] = []
        grouped_codes[key].append(code_entry)

    all_codes_flat = codes
    
    # Paires Positives (Même exercice)
    for key, group in grouped_codes.items():
        if len(group) >= 2:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    pairs.append((group[i]['code_snippet'], group[j]['code_snippet'], 1))

    # Paires Négatives (Exercices différents)
    num_positive_pairs = len([p for p in pairs if p[2] == 1])
    num_negative_pairs = 0
    max_attempts = num_positive_pairs * 10
    attempts = 0

    while num_negative_pairs < num_positive_pairs and attempts < max_attempts:
        code_A = random.choice(all_codes_flat)
        code_B = random.choice(all_codes_flat)
        
        if code_A['code_id'] != code_B['code_id'] and code_A['exercise'] != code_B['exercise']:
            pairs.append((code_A['code_snippet'], code_B['code_snippet'], 0))
            num_negative_pairs += 1
            
        attempts += 1
    
    df = pd.DataFrame(pairs, columns=['code_A', 'code_B', 'true_label'])
    return df

# --- 3. FONCTIONS D'EMBEDDING ET DE CALCUL DE DISTANCE ---

EMBEDDING_CACHE = {} 
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device : {DEVICE}")
def get_hf_embedding(model_name: str, code: str) -> np.ndarray:
    """Calcule l'embedding pour un code donné en utilisant un modèle Hugging Face."""
    if (model_name, code) in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[(model_name, code)]
        
    try:
        # NOTE : On ne charge pas le modèle à chaque appel, mais on le récupère du cache si possible
        if model_name not in EMBEDDING_CACHE:
             model = SentenceTransformer(model_name, trust_remote_code=True, device=DEVICE)
             EMBEDDING_CACHE[model_name] = model

        model = EMBEDDING_CACHE[model_name]
        embedding = model.encode(code, convert_to_numpy=True).flatten()
        EMBEDDING_CACHE[(model_name, code)] = embedding
        return embedding
    except Exception as e:
        # print(f"Erreur HF pour {model_name}: {e}") # Désactiver les prints dans la boucle TQDM
        return np.zeros(768)

def get_openai_embedding(code: str) -> np.ndarray:
    """Calcule l'embedding pour un code donné en utilisant l'API OpenAI."""
    if (OPENAI_MODEL, code) in EMBEDDING_CACHE:
        return EMBEDDING_CACHE[(OPENAI_MODEL, code)]
        
    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY non définie.")
        
        client = OpenAI()
        response = client.embeddings.create(input=code, model=OPENAI_MODEL)
        embedding = np.array(response.data[0].embedding, dtype=np.float32)
        EMBEDDING_CACHE[(OPENAI_MODEL, code)] = embedding
        return embedding
    except Exception as e:
        # print(f"Erreur OpenAI: {e}") # Désactiver les prints dans la boucle TQDM
        return np.zeros(1536) 

def calculate_distance(df: pd.DataFrame, model_name: str, is_openai: bool = False) -> pd.Series:
    """Calcule la distance Cosinus pour toutes les paires du DataFrame avec barre de progression."""
    distances = []
    
    # Utilisation de tqdm pour suivre la progression
    desc = f"{'OpenAI' if is_openai else 'HF'} - {model_name}"
    
    for index, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        code_A, code_B = row['code_A'], row['code_B']
        
        if is_openai:
            emb_A = get_openai_embedding(code_A)
            emb_B = get_openai_embedding(code_B)
        else:
            emb_A = get_hf_embedding(model_name, code_A)
            emb_B = get_hf_embedding(model_name, code_B)

        # Assurer que les vecteurs ne sont pas des vecteurs nuls (en cas d'erreur d'embedding)
        if np.all(emb_A == 0) or np.all(emb_B == 0):
             dist = 1.0 # Distance maximale si l'embedding a échoué
        else:
             dist = pairwise_distances(
                 emb_A.reshape(1, -1), 
                 emb_B.reshape(1, -1), 
                 metric="cosine"
             )[0][0]
             
        distances.append(dist)
        
    return pd.Series(distances)

# --- 4. CLASSIFICATION ET ÉVALUATION ---

def justify_thresholds(thresholds: Dict[str, float]):
    """Justifie le choix des seuils de distance."""
    print("\n" + "="*80)
    print("📢 Justification des Seuils de Distance Cosinus")
    print("   (Distance proche de 0 = Similaire ; Distance proche de 1 = Différent)")
    print("="*80)
    
    print(f"**Seuil {thresholds['Seuil_0.50']:.2f} (Conservateur/Fort)** : Favorise la **Précision** (grande confiance dans la prédiction 'Similaire').")
    print(f"**Seuil {thresholds['Seuil_0.70']:.2f} (Équilibré/Modéré)** : Compromis, souvent bon pour le **F1-Score**.")
    print(f"**Seuil {thresholds['Seuil_0.80']:.2f} (Aggressif/Faible)** : Favorise le **Rappel** (trouve plus de paires similaires), au détriment de la Précision.")
    print("="*80)

def evaluate_model(y_true: pd.Series, distances: pd.Series, thresholds: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Calcule l'AUC et les métriques de classification pour chaque seuil."""
    metrics_results = {}
    
    # 1. AUC (Area Under the Curve)
    try:
        # AUC doit être calculé sur la similarité (1-Distance), où 1.0 est le meilleur score.
        auc = roc_auc_score(y_true, 1 - distances)
    except ValueError:
        auc = np.nan 

    metrics_results['AUC'] = {'Score': auc}

    # 2. Métriques dépendantes du Seuil
    for threshold_name, threshold_val in thresholds.items():
        # Prédit 1 (Similaire) si Distance < seuil, sinon 0 (Différent)
        y_pred = (distances < threshold_val).astype(int)
        
        metrics_results[threshold_name] = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        }
    
    return metrics_results

# --- 5. EXÉCUTION PRINCIPALE ---

if __name__ == "__main__":
    
    # 5.1 Chargement et préparation des données
    print("1. Préparation des données...")
    all_codes = load_jsonl(JSONL_FILE_PATH)
    
    if len(all_codes) < 2:
         print("ERREUR: Données insuffisantes pour créer des paires. Arrêt.")
         exit()
         
    df_pairs = generate_similarity_dataframe(all_codes)
    y_true = df_pairs['true_label']
    
    print(f"   -> Nombre total de paires générées : {len(df_pairs)}")
    print(f"   -> Paires Similaires (label=1) : {y_true.sum()}")
    print(f"   -> Paires Différentes (label=0) : {len(y_true) - y_true.sum()}")
    
    # 5.2 Calcul des Embeddings et Évaluation pour tous les modèles
    
    ALL_RESULTS = {}
    
    print("\n2. Exécution des modèles d'embeddings et calcul des métriques...")
    
    # Itération sur les modèles Hugging Face
    for model_name in MODELS:
        print(f"\n[HF] Démarrage du modèle: {model_name}")
        df_pairs[f'{model_name}_dist'] = calculate_distance(df_pairs, model_name, is_openai=False)
        distances = df_pairs[f'{model_name}_dist']
        ALL_RESULTS[model_name] = evaluate_model(y_true, distances, DISTANCE_THRESHOLDS)
        
    # Test du modèle OpenAI
    #print(f"\n[OpenAI] Démarrage du modèle: {OPENAI_MODEL}")
    #df_pairs[f'{OPENAI_MODEL}_dist'] = calculate_distance(df_pairs, OPENAI_MODEL, is_openai=True)
    #distances_openai = df_pairs[f'{OPENAI_MODEL}_dist']
    #ALL_RESULTS[OPENAI_MODEL] = evaluate_model(y_true, distances_openai, DISTANCE_THRESHOLDS)
    
    # 5.3 Justification des seuils
    justify_thresholds(DISTANCE_THRESHOLDS)
    
    # 5.4 Affichage et Classement Final
    
    ranking_data = []
    
    for model, metrics in ALL_RESULTS.items():
        auc_score = metrics['AUC']['Score']
        f1_070 = metrics['Seuil_0.70']['F1-Score']
        
        ranking_data.append({
            "Modèle": model,
            "AUC_Score": auc_score,
            "F1_Seuil_0.70": f1_070,
        })
        
    df_ranking = pd.DataFrame(ranking_data).sort_values(by="AUC_Score", ascending=False).reset_index(drop=True)
    df_ranking.index = df_ranking.index + 1 
    
    # Affichage
    print("CLASSEMENT FINAL DES CANDIDATS AU FINETUNING (Basé sur l'AUC)")
    print("#"*80)
    print(df_ranking)
    df_ranking.to_json("Embeddings_ranking.json")
    
    
    