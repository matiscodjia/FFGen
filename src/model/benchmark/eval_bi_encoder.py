import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

MODEL_PATH = "matis35/feedbacker-2" 
#Salesforce/SFR-Embedding-Code-400M_R

DATASET_ID = "matis35/cf-synt_V2"
BATCH_SIZE = 64
MAX_LENGTH_CODE = 512     
MAX_LENGTH_FEEDBACK = 128

# Détection Périphérique
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Accélérateur : CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Accélérateur : MPS (Apple Silicon)")
else:
    DEVICE = torch.device("cpu")
    print("Attention : CPU utilisé (Lent)")

# ==========================================
# 2. FONCTIONS UTILITAIRES
# ==========================================
def clean_c_code(code_string):
    """Nettoyage identique à l'entraînement"""
    if not code_string: return ""
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    code_string = re.sub(r'//.*', '', code_string)
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    return code_string.strip()

def mean_pooling(token_embeddings, attention_mask):
    """Pooling standard pour transformer les tokens en un seul vecteur"""
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def encode_batch(model, tokenizer, texts, max_length):
    """Encode une liste de textes en tenseurs normalisés"""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1) # Normalisation L2 critique pour Cosine Sim
    return emb

# ==========================================
# 3. MAIN EVALUATION
# ==========================================
def main():
    print(f"📦 Chargement du modèle depuis {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model.to(DEVICE)
    model.eval()

    print(f"📚 Chargement du dataset {DATASET_ID}...")
    dataset = load_dataset(DATASET_ID)

    # ---------------------------------------------------------
    # ÉTAPE A : CONSTRUIRE L'INDEX (LA BOTTE DE FOIN)
    # On prend TOUS les feedbacks uniques du dataset (Train + Val + Test)
    # ---------------------------------------------------------
    all_data = []
    for split in dataset.keys():
        all_data.extend(dataset[split].to_list())
    
    unique_feedbacks = list(set([item["feedback"] for item in all_data]))
    print(f"   -> Indexation de {len(unique_feedbacks)} feedbacks uniques (Search Space)...")

    feedback_embeddings = []
    for i in tqdm(range(0, len(unique_feedbacks), BATCH_SIZE), desc="Encoding Index"):
        batch = unique_feedbacks[i : i + BATCH_SIZE]
        emb = encode_batch(model, tokenizer, batch, MAX_LENGTH_FEEDBACK)
        feedback_embeddings.append(emb.cpu()) # On garde sur CPU pour ne pas saturer la VRAM
    
    # Matrice Index [N_Docs, Hidden_Dim]
    feedback_index = torch.cat(feedback_embeddings, dim=0).to(DEVICE)

    # ---------------------------------------------------------
    # ÉTAPE B : PRÉPARER LES REQUÊTES (LES AIGUILLES)
    # On utilise UNIQUEMENT le Test Set
    # ---------------------------------------------------------
    test_data = dataset["test"].to_list() if "test" in dataset else dataset["validation"].to_list()
    print(f"   -> Évaluation sur {len(test_data)} requêtes (Test Set)...")
    
    codes = [clean_c_code(x["code"]) for x in test_data]
    targets = [x["feedback"] for x in test_data]

    # ---------------------------------------------------------
    # ÉTAPE C : RECHERCHE & MÉTRIQUES
    # ---------------------------------------------------------
    ranks = []
    
    # On traite les requêtes par batch
    for i in tqdm(range(0, len(codes), BATCH_SIZE), desc="Retrieving"):
        batch_codes = codes[i : i + BATCH_SIZE]
        batch_targets = targets[i : i + BATCH_SIZE]
        
        # 1. Encoder les requêtes
        q_emb = encode_batch(model, tokenizer, batch_codes, MAX_LENGTH_CODE)
        
        # 2. Produit Scalaire (Similarity Search)
        # [Batch, Dim] x [Dim, Index_Size] -> [Batch, Index_Size]
        scores = torch.matmul(q_emb, feedback_index.T)
        
        # 3. Calcul du Rang pour chaque requête
        for j, true_feedback in enumerate(batch_targets):
            try:
                # Où est le vrai feedback dans notre liste unique ?
                target_idx = unique_feedbacks.index(true_feedback)
                
                # Quel score a-t-il obtenu ?
                true_score = scores[j, target_idx].item()
                
                # Combien de feedbacks ont un meilleur score que lui ?
                # (C'est la définition du rang)
                rank = (scores[j] > true_score).sum().item() + 1
                ranks.append(rank)
                
            except ValueError:
                # Cas théorique où le feedback du test n'est pas dans l'index global
                ranks.append(10000)

    # ---------------------------------------------------------
    # ÉTAPE D : RÉSULTATS
    # ---------------------------------------------------------
    ranks = np.array(ranks)
    metrics = {
        "mrr": np.mean(1 / ranks),
        "recall_at_1": np.mean(ranks <= 1),
        "recall_at_5": np.mean(ranks <= 5),
        "recall_at_10": np.mean(ranks <= 10),
        "median_rank": np.median(ranks)
    }

    print("\n📊 RÉSULTATS GLOBAUX (Full Retrieval) :")
    print(f"   MRR        : {metrics['mrr']:.4f}")
    print(f"   Recall@1   : {metrics['recall_at_1']:.4f} ({metrics['recall_at_1']*100:.2f}%)")
    print(f"   Recall@5   : {metrics['recall_at_5']:.4f} ({metrics['recall_at_5']*100:.2f}%)")
    print(f"   Recall@10  : {metrics['recall_at_10']:.4f} ({metrics['recall_at_10']*100:.2f}%)")
    print(f"   Median Rank: {metrics['median_rank']}")

    # Sauvegarde JSON
    with open("global_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print("\n💾 Métriques sauvegardées dans 'global_metrics.json'")

if __name__ == "__main__":
    main()