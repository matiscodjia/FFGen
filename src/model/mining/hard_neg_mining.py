import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import json
import os
import re

# CONFIGURATION
MODEL_PATH = "matis35/feedbacker-2" # Ton modèle actuel (entrainé avec random negatives)
DATASET_ID = "matis35/cf-synt_V2"
OUTPUT_FILE = "train_hard_negatives.json"
BATCH_SIZE = 64
NUM_HARD_NEGATIVES = 2 # Combien de pièges on garde par code ?

# NETTOYAGE
def clean_c_code(code_string):
    if not code_string: return ""
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    code_string = re.sub(r'//.*', '', code_string)
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    return code_string.strip()

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def main():
    device = "cuda" if torch.cuda.is_available() else "mps"
    print(f"⛏️ Mining Hard Negatives sur {device}...")

    # 1. Chargement Modèle & Data
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(device)
    model.eval()

    dataset = load_dataset(DATASET_ID)
    train_data = dataset["train"].to_list()
    
    # On indexe UNIQUEMENT le Train set (on ne mine pas sur le test !)
    feedbacks = list(set([item["feedback"] for item in train_data]))
    print(f"Indexation de {len(feedbacks)} feedbacks uniques du TRAIN set...")

    # 2. Encodage de tous les feedbacks (L'Index)
    feedback_embs = []
    for i in tqdm(range(0, len(feedbacks), BATCH_SIZE), desc="Encoding Feedbacks"):
        batch = feedbacks[i : i+BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            emb = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
            feedback_embs.append(F.normalize(emb, p=2, dim=1))
    
    feedback_index = torch.cat(feedback_embs, dim=0) # [N_Feedbacks, Dim]

    # 3. Mining : Pour chaque code d'entrainement, on cherche les pièges
    triplets = []
    
    print("🔍 Recherche des Hard Negatives...")
    # On traite les codes par batch pour aller vite
    for i in tqdm(range(0, len(train_data), BATCH_SIZE)):
        batch_items = train_data[i : i+BATCH_SIZE]
        batch_codes = [clean_c_code(x["code"]) for x in batch_items]
        batch_positives = [x["feedback"] for x in batch_items]
        
        # Encoder les codes
        inputs = tokenizer(batch_codes, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
            q_emb = mean_pooling(out.last_hidden_state, inputs["attention_mask"])
            q_emb = F.normalize(q_emb, p=2, dim=1)
        
        # Calculer similarité avec TOUS les feedbacks
        scores = torch.matmul(q_emb, feedback_index.T) # [Batch, N_Feedbacks]
        
        # Récupérer les Top-K scores (ex: Top 10)
        # On prend large pour pouvoir filtrer la vraie réponse
        top_scores, top_indices = torch.topk(scores, k=10, dim=1)
        
        top_indices = top_indices.cpu().numpy()
        
        for j, (code, positive) in enumerate(zip(batch_codes, batch_positives)):
            found_negatives = []
            
            # On parcourt les candidats proposés par le modèle
            for candidate_idx in top_indices[j]:
                candidate_text = feedbacks[candidate_idx]
                
                # CRITIQUE : On ne garde le négatif QUE s'il est différent du positif
                # (Évite de miner la bonne réponse comme un négatif)
                if candidate_text != positive:
                    found_negatives.append(candidate_text)
                
                if len(found_negatives) >= NUM_HARD_NEGATIVES:
                    break
            
            # On sauvegarde le triplet (Code, Vrai, [Faux1, Faux2])
            if found_negatives:
                triplets.append({
                    "code": code,
                    "positive": positive,
                    "negatives": found_negatives
                })

    # 4. Sauvegarde
    print(f"Sauvegarde de {len(triplets)} triplets d'entrainement dans {OUTPUT_FILE}")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(triplets, f, indent=4)

if __name__ == "__main__":
    main()