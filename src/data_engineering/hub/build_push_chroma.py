import os
import shutil
import logging
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import re

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import chromadb
from huggingface_hub import HfApi, create_repo, login

# ==========================================
# CONFIGURATION
# ==========================================

# 1. Modèle d'Embedding
MODEL_ID = "matis35/feedbacker-2"

# 2. Dataset Source
SOURCE_DATASET_ID = "matis35/cf-synt_V2"

# 3. Dataset Cible (Stockage ChromaDB)
TARGET_REPO_ID = "matis35/chroma-rag-storage"
CHROMA_DB_DIR = "chroma_db_storage"

# 4. Paramètres
COLLECTION_NAME = "feedbacks"
BATCH_SIZE = 64
DISTANCE_METRIC = "cosine"

# Token HF
HF_TOKEN = os.environ.get("HF_TOKEN")

# Logs
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 1. NETTOYAGE CODE C (Ta fonction)
# ==========================================
def clean_c_code(code_string):
    """Nettoie le code C pour ne garder que la logique essentielle."""
    if not code_string: return ""
    
    # 1. Supprimer les commentaires blocs /* ... */
    code_string = re.sub(r'/\*[\s\S]*?\*/', '', code_string)
    
    # 2. Supprimer les commentaires ligne // ...
    code_string = re.sub(r'//.*', '', code_string)
    
    # 3. Supprimer la fonction main et tout ce qui suit
    code_string = re.sub(r'(int|void)\s+main\s*\(.*?\)\s*\{[\s\S]*', '', code_string)
    
    # 4. Nettoyage des espaces vides excessifs
    return code_string.strip()

# ==========================================
# FONCTIONS UTILITAIRES
# ==========================================

def setup_environment():
    if HF_TOKEN:
        login(token=HF_TOKEN)
        logger.info("✅ Connecté à Hugging Face.")
    
    if os.path.exists(CHROMA_DB_DIR):
        logger.warning(f"🧹 Suppression de l'ancien dossier '{CHROMA_DB_DIR}'...")
        shutil.rmtree(CHROMA_DB_DIR)
    os.makedirs(CHROMA_DB_DIR, exist_ok=True)

def load_model_and_tokenizer():
    logger.info(f"📥 Chargement du modèle : {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available(): device = "mps" # Support Mac
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer, device

def encode_batch(texts, model, tokenizer, device):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy().tolist()

# ==========================================
# PIPELINE PRINCIPAL
# ==========================================

def main():
    setup_environment()
    
    # --- 1. TÉLÉCHARGEMENT ---
    logger.info(f"📚 Téléchargement du dataset : {SOURCE_DATASET_ID}")
    dataset = load_dataset(SOURCE_DATASET_ID)
    
    all_data = []
    for split in dataset.keys():
        all_data.extend(dataset[split].to_list())
    
    logger.info(f"✅ Dataset chargé : {len(all_data)} entrées.")

    # --- 2. CHROMADB ---
    logger.info(f"💽 Initialisation ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.create_collection(
        name=COLLECTION_NAME, 
        metadata={"hnsw:space": DISTANCE_METRIC}
    )

    # --- 3. INDEXATION ---
    model, tokenizer, device = load_model_and_tokenizer()
    logger.info(f"🚀 Indexation sur {device}...")
    
    total_items = len(all_data)
    
    with tqdm(total=total_items, desc="Indexation") as pbar:
        for i in range(0, total_items, BATCH_SIZE):
            batch = all_data[i : i + BATCH_SIZE]
            
            feedbacks = [item.get("feedback", item.get("generated_feedback", "")) for item in batch]
            
            # APPLICATION DU NETTOYAGE ICI
            raw_codes = [item.get("code", "") for item in batch]
            clean_codes = [clean_c_code(c) for c in raw_codes]
            
            # Encodage (Feedback)
            embeddings = encode_batch(feedbacks, model, tokenizer, device)
            
            # Métadonnées (Avec le code nettoyé)
            metadatas = []
            for item, c_code in zip(batch, clean_codes):
                meta = {
                    "code": c_code,  # <-- On stocke le code propre
                    "theme": str(item.get("theme", "N/A")),
                    "difficulty": str(item.get("difficulty", "N/A")),
                    "error_category": str(item.get("error_category", "N/A"))
                }
                metadatas.append(meta)
                
            ids = [f"id_{i+j}" for j in range(len(batch))]
            
            collection.add(
                embeddings=embeddings,
                documents=feedbacks,
                metadatas=metadatas,
                ids=ids
            )
            pbar.update(len(batch))
            
    logger.info(f"✅ Indexation terminée ! {collection.count()} items.")
    
    # Nettoyage mémoire
    del model, tokenizer
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 4. UPLOAD ---
    logger.info(f"Upload vers {TARGET_REPO_ID}...")
    api = HfApi()
    
    create_repo(repo_id=TARGET_REPO_ID, repo_type="dataset", exist_ok=True, private=True)
    
    api.upload_folder(
        folder_path=CHROMA_DB_DIR,
        path_in_repo=CHROMA_DB_DIR,
        repo_id=TARGET_REPO_ID,
        repo_type="dataset",
        commit_message=f"Update index with {len(all_data)} cleaned items"
    )
    
    logger.info(f"🎉 Terminé ! Lien : https://huggingface.co/datasets/{TARGET_REPO_ID}/tree/main/{CHROMA_DB_DIR}")

if __name__ == "__main__":
    main()