import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import PeftModel
import chromadb
from datasets import load_dataset
import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Le dossier contenant adapter_config.json et adapter_model.safetensors
ADAPTER_PATH = "./best_model_gemma_final_adapter_1" 
BASE_MODEL = "google/embeddinggemma-300m"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Paramètres du Pipeline
TOP_K_RETRIEVAL = 15   # Le Bi-Encoder ratisse large
RERANK_BATCH_SIZE = 10 # Pour ne pas saturer la mémoire du Reranker

# Détection Hardware
if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("🚀 Mode: Apple Metal (MPS)")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("🚀 Mode: NVIDIA CUDA")
else:
    DEVICE = "cpu"
    print("⚠️ Mode: CPU (Lent)")

# ==========================================
# 1. CHARGEMENT DES MODÈLES
# ==========================================
print("\n🔄 Chargement du Bi-Encoder (Gemma + LoRA)...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base = AutoModel.from_pretrained(BASE_MODEL, trust_remote_code=True)
embedder = PeftModel.from_pretrained(base, ADAPTER_PATH)
embedder.to(DEVICE)
embedder.eval()

print("🔄 Chargement du Reranker (BGE-M3)...")
rerank_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
rerank_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL).to(DEVICE)
rerank_model.eval()

# ==========================================
# 2. FONCTIONS D'INFÉRENCE
# ==========================================
def get_embedding(text):
    with torch.no_grad():
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        outputs = embedder.base_model(**inputs)
        emb = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        mask = inputs.attention_mask.unsqueeze(-1).expand(emb.size()).float()
        vec = torch.sum(emb * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(vec, p=2, dim=1).cpu().numpy().flatten().tolist()

def rerank_candidates(query, candidates):
    if not candidates: return []
    pairs = [[query, doc] for doc in candidates]
    
    # Batching pour éviter OOM sur le reranker
    all_scores = []
    for i in range(0, len(pairs), RERANK_BATCH_SIZE):
        batch = pairs[i : i + RERANK_BATCH_SIZE]
        with torch.no_grad():
            inputs = rerank_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
            scores = rerank_model(**inputs, return_dict=True).logits.view(-1).float()
            all_scores.extend(torch.sigmoid(scores).cpu().numpy())
            
    return all_scores

# ==========================================
# 3. PRÉPARATION DES DONNÉES & INDEX
# ==========================================
print("\n📥 Chargement du TEST SET...")
dataset = load_dataset('matis35/RAFT_CLEAN_V1', split='test') # On analyse tout le test set

# ChromaDB en mémoire (éphémère pour l'analyse)
client = chromadb.Client()
try:
    client.delete_collection("eval_db")
except: pass
collection = client.create_collection(name="eval_db")

print("building Index Vectoriel (Knowledge Base)...")
docs_content = []
docs_ids = []

# On indexe tous les feedbacks uniques du test set pour créer l'espace de recherche
# Note : Dans la vraie vie, on indexerait TOUTE la base de feedbacks connue.
# Ici, on assume que la réponse est quelque part dans le test set.
unique_feedbacks = list(set([item['feedback'] for item in dataset]))
batch_size = 16

for i in tqdm(range(0, len(unique_feedbacks), batch_size), desc="Indexing"):
    batch_docs = unique_feedbacks[i : i + batch_size]
    batch_vecs = [get_embedding(d) for d in batch_docs]
    batch_ids = [str(hash(d)) for d in batch_docs] # ID unique basé sur le contenu
    
    collection.add(documents=batch_docs, embeddings=batch_vecs, ids=batch_ids)

# ==========================================
# 4. BOUCLE D'ÉVALUATION
# ==========================================
print(f"\n🔬 Démarrage de l'analyse sur {len(dataset)} requêtes...")

metrics = {"mrr": [], "recall_1": [], "recall_5": [], "recall_10": []}
failures = [] # Pour stocker les cas difficiles

for idx, item in tqdm(enumerate(dataset), total=len(dataset), desc="Evaluating"):
    query_code = item['code']
    target_feedback = item['feedback']
    
    # --- A. RETRIEVAL (Bi-Encoder) ---
    q_vec = get_embedding(query_code)
    results = collection.query(query_embeddings=[q_vec], n_results=TOP_K_RETRIEVAL)
    
    retrieved_docs = results['documents'][0]
    
    # --- B. RERANKING (Cross-Encoder) ---
    scores = rerank_candidates(query_code, retrieved_docs)
    
    # Tri par score Cross-Encoder
    ranked_results = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
    sorted_docs = [doc for doc, score in ranked_results]
    
    # --- C. CALCUL DES MÉTRIQUES ---
    try:
        # On cherche la position de la bonne réponse
        rank = sorted_docs.index(target_feedback) + 1
    except ValueError:
        rank = float('inf') # Pas trouvé dans le Top K
    
    # Stockage
    metrics["mrr"].append(1.0 / rank if rank <= TOP_K_RETRIEVAL else 0.0)
    metrics["recall_1"].append(1 if rank == 1 else 0)
    metrics["recall_5"].append(1 if rank <= 5 else 0)
    metrics["recall_10"].append(1 if rank <= 10 else 0)
    
    # Analyse d'erreur : Si la réponse n'est même pas dans le Top 10 après rerank
    if rank > 10:
        failures.append({
            "query_id": idx,
            "rank": rank if rank != float('inf') else f"> {TOP_K_RETRIEVAL}",
            "code_snippet": query_code[:100] + "...",
            "target_feedback": target_feedback,
            "top_predicted": sorted_docs[0] if sorted_docs else "None",
            "top_score": ranked_results[0][1] if ranked_results else 0.0
        })

# ==========================================
# 5. RÉSULTATS & RAPPORT
# ==========================================
print("\n" + "="*40)
print("📊 RÉSULTATS GLOBAUX DU PIPELINE RAG")
print("="*40)

final_mrr = np.mean(metrics["mrr"])
final_r1 = np.mean(metrics["recall_1"])
final_r5 = np.mean(metrics["recall_5"])
final_r10 = np.mean(metrics["recall_10"])

print(f"✅ MRR        : {final_mrr:.4f}")
print(f"✅ Recall@1   : {final_r1:.4f}")
print(f"✅ Recall@5   : {final_r5:.4f}")
print(f"✅ Recall@10  : {final_r10:.4f}")
print("-" * 40)
print(f"📉 Nombre d'échecs (hors Top 10) : {len(failures)} / {len(dataset)}")

# Sauvegarde des échecs pour analyse humaine
if failures:
    df_fail = pd.DataFrame(failures)
    df_fail.to_csv("analyse_echecs_rag.csv", index=False)
    print("📝 Les détails des échecs ont été sauvegardés dans 'analyse_echecs_rag.csv'")
    print("\nExemple d'échec :")
    print(df_fail.iloc[0])