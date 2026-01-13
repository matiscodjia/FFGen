import os
import torch
import torch.nn.functional as F
import numpy as np
import json
import re
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# ==========================================
# CONFIGURATION
# ==========================================
# Le chemin vers ton modèle sauvegardé (celui que tu veux évaluer)
MODEL_PATH = "./final_model_hard_negatives_lora" 
DATASET_ID = "matis35/cf-synt_V2"
BATCH_SIZE = 64
# Détection automatique du périphérique
if torch.cuda.is_available():
    DEVICE = torch.device("cpu")
    print(f"✅ Accélérateur détecté : CUDA ({torch.cuda.get_device_name(0)})")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✅ Accélérateur détecté : MPS (Apple Silicon)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ Aucun accélérateur détecté : CPU utilisé (Lent)")

print(f"🚀 Évaluation Full-Corpus sur {DEVICE}...")


base_model = AutoModel.from_pretrained("matis35/feedbacker-2")
model = PeftModel.from_pretrained(base_model, "./final_model_hard_negatives_lora")

# ==========================================
# FONCTIONS UTILITAIRES
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
# 1. CHARGEMENT
# ==========================================
print("📦 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.float16 if DEVICE=="cuda" else torch.float32)
model.to(DEVICE)
model.eval()

print("📚 Chargement du dataset...")
dataset = load_dataset(DATASET_ID)

# ==========================================
# 2. CONSTRUCTION DE L'INDEX (La Botte de Foin)
# ==========================================
# On prend TOUT (Train + Val + Test) pour les feedbacks
# C'est ton "Search Space" complet.
all_data = []
for split in dataset.keys():
    all_data.extend(dataset[split].to_list())

# On dédoublonne les feedbacks pour l'index (inutile d'avoir 2x le même texte exact)
unique_feedbacks = list(set([item["feedback"] for item in all_data]))
print(f"   -> Index : {len(unique_feedbacks)} feedbacks uniques (Train + Val + Test).")

# Encodage de l'Index
feedback_embeddings = []
for i in tqdm(range(0, len(unique_feedbacks), BATCH_SIZE), desc="Encoding Index"):
    batch_texts = unique_feedbacks[i : i + BATCH_SIZE]
    inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        feedback_embeddings.append(emb.cpu())

feedback_index = torch.cat(feedback_embeddings, dim=0).to(DEVICE) # [N_Docs, Dim]

# ==========================================
# 3. DÉFINITION DES REQUÊTES (Les Aiguilles)
# ==========================================
# On utilise UNIQUEMENT le Test Set pour les requêtes
# C'est la seule façon de mesurer la généralisation.
test_data = dataset["test"].to_list() if "test" in dataset else dataset["validation"].to_list()
print(f"   -> Requêtes : {len(test_data)} codes inconnus (Test Set).")

# ==========================================
# 4. ÉVALUATION
# ==========================================
ranks = []
hits_at_1 = 0
hits_at_5 = 0
hits_at_10 = 0

print("\n🔥 Lancement de la recherche...")

for i in tqdm(range(0, len(test_data), BATCH_SIZE), desc="Searching"):
    # Batching des requêtes pour aller plus vite
    batch_items = test_data[i : i + BATCH_SIZE]
    batch_codes = [clean_c_code(item["code"]) for item in batch_items]
    batch_targets = [item["feedback"] for item in batch_items]
    
    # Encodage Requêtes
    inputs = tokenizer(batch_codes, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        q_emb = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
        q_emb = F.normalize(q_emb, p=2, dim=1) # [Batch, Dim]
    
    # Recherche Vectorielle (Matrice Batch x Index)
    # Scores = Cosine Similarity
    scores = torch.matmul(q_emb, feedback_index.T) # [Batch, N_Docs]
    
    # Pour chaque requête du batch
    for j, true_feedback in enumerate(batch_targets):
        # On trouve l'index du vrai feedback dans la liste unique
        try:
            target_idx = unique_feedbacks.index(true_feedback)
        except ValueError:
            # Cas rare : le feedback du test n'est pas dans l'index (nettoyage ?)
            # On considère ça comme un échec (rank = max)
            ranks.append(10000)
            continue
            
        # Score de la bonne réponse
        true_score = scores[j, target_idx].item()
        
        # Combien de documents ont un meilleur score que la bonne réponse ?
        # C'est le rang (Rank)
        better_scores_count = (scores[j] > true_score).sum().item()
        rank = better_scores_count + 1
        
        ranks.append(rank)
        
        if rank <= 1: hits_at_1 += 1
        if rank <= 5: hits_at_5 += 1
        if rank <= 10: hits_at_10 += 1

# ==========================================
# 5. RÉSULTATS
# ==========================================
ranks = np.array(ranks)
metrics = {
    "mrr": np.mean(1 / ranks),
    "recall_at_1": hits_at_1 / len(test_data),
    "recall_at_5": hits_at_5 / len(test_data),
    "recall_at_10": hits_at_10 / len(test_data),
    "median_rank": np.median(ranks)
}

print("\n📊 RÉSULTATS SUR DATASET COMPLET (15k docs) :")
print(f"   MRR        : {metrics['mrr']:.4f}")
print(f"   Recall@1   : {metrics['recall_at_1']:.4f}")
print(f"   Recall@5   : {metrics['recall_at_5']:.4f}")
print(f"   Recall@10  : {metrics['recall_at_10']:.4f}")
print(f"   Median Rank: {metrics['median_rank']}")

# Sauvegarde
with open("evaluation_full_corpus.json", "w") as f:
    json.dump(metrics, f, indent=4)