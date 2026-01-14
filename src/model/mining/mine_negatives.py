import torch
import json
import os
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from collections import Counter
import re

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# On utilise un modèle "juge impartial" déjà entrainé sur des millions de phrases
# pour avoir une vraie mesure de similarité sémantiqu
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" 
DATASET_ID = "matis35/cf-synt_V2" # Ton dataset
OUTPUT_FILE = "train_hard_negatives_cleaned_three.json"

BATCH_SIZE = 128
CANDIDATES_TO_FETCH = 30   # On regarde large pour pouvoir filtrer
NUM_NEGATIVES_NEEDED = 3   # On veut 2 négatifs finaux
HUB_THRESHOLD_PCT = 0.02   # Si un feedback apparait dans > 2% des tops candidats, il est banni (trop générique)

# ==============================================================================
# SCRIPT
# ==============================================================================


# ==========================================
# 1. NETTOYAGE CODE C
# ==========================================
def clean_c_code(code_string):
    if not code_string: return ""
    
    # 1. Supprimer les commentaires blocs /* ... */
    # [\s\S] permet de matcher aussi les sauts de ligne
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    
    # 2. Supprimer les commentaires ligne // ...
    code_string = re.sub(r'//.*', '', code_string)
    
    # 3. Supprimer la fonction main et tout ce qui suit
    # On cherche "int main(...){" ou "void main(...){" et on coupe tout jusqu'à la fin
    # C'est une heuristique robuste car le main sert souvent de runner de test à la fin du fichier
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    
    # 4. (Optionnel) Supprimer les directives #include si tu veux vraiment juste la logique
    # code_string = re.sub(r'#include.*', '', code_string)
    
    # 5. Nettoyage des espaces vides excessifs
    return code_string.strip()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Démarrage du Mining sur {device}...")

    # 1. Chargement des données
    print("Chargement du dataset...")
    dataset = load_dataset(DATASET_ID, split="train")
    
    # Extraction propre
    data_list = []
    unique_feedbacks_set = set()
    
    for item in tqdm(dataset, desc="Parsing"):
        # On garde code et feedback pour référence
        code = clean_c_code(item["code"])
        feedback = item["feedback"]
        data_list.append({"code": code, "positive": feedback})
        unique_feedbacks_set.add(feedback)

    unique_feedbacks = list(unique_feedbacks_set)
    print(f"Stats: {len(data_list)} exemples, {len(unique_feedbacks)} feedbacks uniques.")

    # 2. Encodage (Vectorisation)
    print(f"Chargement du modèle {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    print("Embeddings des Feedbacks (Corpus)...")
    feedback_embeddings = model.encode(unique_feedbacks, batch_size=BATCH_SIZE, convert_to_tensor=True, show_progress_bar=True)
    
    print("Embeddings des Codes (Queries)...")
    # Note: MiniLM marche étonnamment bien sur du code brut pour de la similarité structurelle simple
    code_texts = [d["code"] for d in data_list]
    code_embeddings = model.encode(code_texts, batch_size=BATCH_SIZE, convert_to_tensor=True, show_progress_bar=True)

    # 3. Mining Phase 1 : Scouting (Recherche Large)
    print("🔍 Recherche des voisins sémantiques (Semantic Search)...")
    # semantic_search gère les batches automatiquement pour éviter l'OOM
    hits = util.semantic_search(
        code_embeddings, 
        feedback_embeddings, 
        top_k=CANDIDATES_TO_FETCH, 
        score_function=util.cos_sim
    )

    # 4. Détection des "Hubs" (Feedbacks toxiques/génériques)
    print("🛡️ Analyse statistique pour détecter les Feedbacks 'Passe-Partout'...")
    all_hits_indices = []
    for hit_list in hits:
        for item in hit_list:
            all_hits_indices.append(item['corpus_id'])
    
    # On compte combien de fois chaque feedback a été suggéré
    counts = Counter(all_hits_indices)
    total_queries = len(data_list)
    threshold_count = total_queries * HUB_THRESHOLD_PCT
    
    blacklist_indices = set()
    for idx, count in counts.items():
        if count > threshold_count:
            blacklist_indices.add(idx)
    
    print(f"{len(blacklist_indices)} feedbacks bannis (apparaissent trop souvent).")
    print(f"   Exemple de seuil: banni si apparaît > {int(threshold_count)} fois.")
    
    # Petit check visuel des bannis pour toi
    if len(blacklist_indices) > 0:
        print("   Exemples de feedbacks bannis (Top 3):")
        top_banned = counts.most_common(3)
        for idx, cnt in top_banned:
            print(f"   - ({cnt}x) : {unique_feedbacks[idx][:100]}...")

    # 5. Sélection Finale (Filtrage)
    print("Construction des triplets finaux...")
    final_dataset = []
    
    skipped_count = 0
    
    for i, hit_list in enumerate(tqdm(hits)):
        anchor_code = data_list[i]["code"]
        positive_feedback = data_list[i]["positive"]
        
        selected_negatives = []
        
        for hit in hit_list:
            feedback_idx = hit['corpus_id']
            candidate_feedback = unique_feedbacks[feedback_idx]
            
            # CRITÈRES DE SÉLECTION STRICTS :
            # 1. Ce n'est pas le vrai positif (évident)
            # 2. Ce n'est pas un feedback banni (HUB)
            # 3. Il est assez long (évite les "Syntax Error" de 2 mots)
            if (candidate_feedback != positive_feedback and 
                feedback_idx not in blacklist_indices and 
                len(candidate_feedback) > 20):
                
                selected_negatives.append(candidate_feedback)
            
            if len(selected_negatives) >= NUM_NEGATIVES_NEEDED:
                break
        
        # Si après filtrage on a trouvé nos négatifs, on sauvegarde
        if len(selected_negatives) >= NUM_NEGATIVES_NEEDED:
            final_dataset.append({
                "code": anchor_code,
                "positive": positive_feedback,
                "negatives": selected_negatives
            })
        else:
            skipped_count += 1

    # 6. Sauvegarde
    print(f"Sauvegarde de {len(final_dataset)} triplets propres dans {OUTPUT_FILE}...")
    print(f"{skipped_count} exemples ignorés (pas assez de négatifs valides trouvés).")
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)
        
    print("Terminé.")

if __name__ == "__main__":
    main()