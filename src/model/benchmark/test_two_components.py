import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Chemins vers TES modèles sauvegardés
BI_ENCODER_PATH = "./final_merged_model"            # Ton modèle Gemma/LoRA fusionné
CROSS_ENCODER_PATH = "./cross_encoder_codebert_reranker" # Ton modèle CodeBERT

DATASET_ID = "matis35/cf-synt_V2"
TOP_K_RETRIEVAL = 200  # Le Bi-Encoder récupère les 20 meilleurs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🚀 Benchmarking sur {DEVICE}...")

# ==========================================
# 2. FONCTIONS UTILITAIRES
# ==========================================
def clean_c_code(code_string):
    if not code_string: return ""
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    code_string = re.sub(r'//.*', '', code_string)
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    return code_string.strip()

def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# ==========================================
# 3. CHARGEMENT DES MODÈLES
# ==========================================
print("📦 Chargement des modèles...")

# A. BI-ENCODER (Retriever)
bi_tokenizer = AutoTokenizer.from_pretrained(BI_ENCODER_PATH)
bi_model = AutoModel.from_pretrained(BI_ENCODER_PATH, torch_dtype=torch.float32)
bi_model.to(DEVICE)
bi_model.eval()

# B. CROSS-ENCODER (Reranker)
cross_tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_PATH)
cross_model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_PATH)
cross_model.to(DEVICE)
cross_model.eval()

# ==========================================
# 4. PRÉPARATION DE LA BASE DE DOCUMENT (INDEX)
# ==========================================
print("📚 Indexation de la base de connaissances (Feedbacks)...")
dataset = load_dataset(DATASET_ID)

# 1. Définition des Requêtes de Test (Ce qu'on va chercher)
# On prend le set de test s'il existe, sinon validation
test_queries = dataset["test"].to_list() if "test" in dataset else dataset["validation"].to_list()

# Optionnel : Décommenter pour tester vite fait sur 500 exemples
# test_queries = test_queries[:500]

# 2. Construction de l'Index Global (Où on cherche)
# On fusionne Train + Val + Test pour être sûr que la réponse existe
all_data = []
for split in dataset.keys():
    all_data.extend(dataset[split].to_list())

# 3. Extraction des feedbacks UNIQUES
unique_feedbacks = list(set([item["feedback"] for item in all_data]))
print(f"   -> {len(unique_feedbacks)} feedbacks uniques dans l'index global.")
print(f"   -> {len(test_queries)} requêtes à évaluer.")

# 4. Encodage de la base (inchangé)
feedback_embeddings = []
batch_size = 32

for i in tqdm(range(0, len(unique_feedbacks), batch_size), desc="Encoding KB"):
    batch_texts = unique_feedbacks[i : i + batch_size]
    inputs = bi_tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = bi_model(**inputs)
        emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        feedback_embeddings.append(emb.cpu())

feedback_index = torch.cat(feedback_embeddings, dim=0)
# ==========================================
# 5. BOUCLE DE TEST
# ==========================================
print("\n🔥 Démarrage du Benchmark...")

metrics = {
    "bi_encoder": {"mrr": [], "r1": [], "r5": [], "r10": []},
    "reranked":   {"mrr": [], "r1": [], "r5": [], "r10": []}
}

for item in tqdm(test_queries, desc="Evaluating"):
    query_code = clean_c_code(item["code"])
    true_feedback = item["feedback"]
    
    # --- ETAPE 1 : RETRIEVAL (Bi-Encoder) ---
    # Encoder la requête
    inputs = bi_tokenizer(query_code, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        q_emb = mean_pooling(bi_model(**inputs).last_hidden_state, inputs["attention_mask"])
        q_emb = F.normalize(q_emb, p=2, dim=1).cpu() # [1, Hidden_Dim]
    
    # Recherche vectorielle (Dot product)
    scores = torch.matmul(q_emb, feedback_index.T).squeeze(0) # [N_feedbacks]
    top_k_scores, top_k_indices = torch.topk(scores, k=TOP_K_RETRIEVAL)
    
    # Récupérer les textes des candidats
    candidates = [unique_feedbacks[idx] for idx in top_k_indices.tolist()]
    
    # --- METRIQUES BI-ENCODER SEUL ---
    try:
        rank = candidates.index(true_feedback) + 1
    except ValueError:
        rank = 1000 # Pas trouvé dans le Top K
    
    metrics["bi_encoder"]["mrr"].append(1.0/rank if rank <= TOP_K_RETRIEVAL else 0)
    metrics["bi_encoder"]["r1"].append(1 if rank == 1 else 0)
    metrics["bi_encoder"]["r5"].append(1 if rank <= 5 else 0)
    metrics["bi_encoder"]["r10"].append(1 if rank <= 10 else 0)
    
    # Si le Bi-Encoder a raté le bon feedback dans le Top-20, le Reranker ne peut pas le sauver
    if rank > TOP_K_RETRIEVAL:
        metrics["reranked"]["mrr"].append(0)
        metrics["reranked"]["r1"].append(0)
        metrics["reranked"]["r5"].append(0)
        metrics["reranked"]["r10"].append(0)
        continue

    # --- ETAPE 2 : RERANKING (Cross-Encoder) ---
    # On prépare les paires [Code, Candidat] pour les 20 candidats
    pairs = [[query_code, cand] for cand in candidates]
    
    cross_inputs = cross_tokenizer(
        [p[0] for p in pairs], 
        [p[1] for p in pairs], 
        padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        logits = cross_model(**cross_inputs).logits.squeeze(-1)
        rerank_scores = torch.sigmoid(logits).cpu().numpy()
    
    # Trier les 20 candidats selon le nouveau score
    sorted_indices_local = np.argsort(rerank_scores)[::-1] # Indices de 0 à 19
    reranked_candidates = [candidates[i] for i in sorted_indices_local]
    
    # --- METRIQUES APRES RERANKING ---
    try:
        new_rank = reranked_candidates.index(true_feedback) + 1
    except ValueError:
        new_rank = 1000 # Should not happen if it was in Top K
        
    metrics["reranked"]["mrr"].append(1.0/new_rank)
    metrics["reranked"]["r1"].append(1 if new_rank == 1 else 0)
    metrics["reranked"]["r5"].append(1 if new_rank <= 5 else 0)
    metrics["reranked"]["r10"].append(1 if new_rank <= 10 else 0)

# ==========================================
# 6. AFFICHAGE DES RÉSULTATS
# ==========================================
def print_res(name, m):
    print(f"\n📊 --- {name} ---")
    print(f"   MRR        : {np.mean(m['mrr']):.4f}")
    print(f"   Recall@1   : {np.mean(m['r1']):.4f}")
    print(f"   Recall@5   : {np.mean(m['r5']):.4f}")
    print(f"   Recall@10  : {np.mean(m['r10']):.4f}")

print_res("Bi-Encoder (Sans Reranker)", metrics["bi_encoder"])
print_res("Pipeline Complet (Avec Reranker)", metrics["reranked"])

# Calcul du Gain
gain_r1 = (np.mean(metrics["reranked"]["r1"]) - np.mean(metrics["bi_encoder"]["r1"])) * 100
print(f"\n🚀 GAIN net sur la Top-1 Accuracy : +{gain_r1:.2f}%")