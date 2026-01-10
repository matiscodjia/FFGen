import os
import shutil
import chromadb
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, create_repo
import json
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_SOURCE = "matis35/SYNT_V4" # Ton dataset de feedback
MODEL_NAME = "matis35/gemmaembedding-fgdor"
OUTPUT_DIR = "chroma_db_storage"
REPO_ID = "matis35/chroma-rag-storage" # Là où on stocke la DB

# --- 1. SETUP MODEL ---
print("🚀 Chargement du modèle...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, device_map="auto")

def encode_text(text_list):
    inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        # NORMALISATION IMPORTANTE POUR LE COSINE
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().tolist()

# --- 2. CLEAN & INIT DB ---
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR) # On supprime pour être sûr de repartir à zéro

print("🛠️ Création de la base Chroma (Mode COSINE)...")
client = chromadb.PersistentClient(path=OUTPUT_DIR)

# C'EST ICI QUE TOUT SE JOUE : on force "cosine"
collection = client.create_collection(
    name="feedbacks",
    metadata={"hnsw:space": "cosine"} 
)

# --- 3. INDEXATION ---
print(f"📥 Chargement des données depuis {DATASET_SOURCE}...")
dataset = load_dataset(DATASET_SOURCE)
data = []
for split in dataset.keys():
    data.extend(dataset[split].to_list())

print(f"⚙️ Indexation de {len(data)} documents...")

BATCH_SIZE = 32
for i in tqdm(range(0, len(data), BATCH_SIZE)):
    batch = data[i:i+BATCH_SIZE]
    
    # On indexe le FEEDBACK (pour que la recherche code -> feedback fonctionne)
    feedbacks = [item.get("feedback", "") for item in batch]
    codes = [item.get("code", "") for item in batch]
    
    # Encodage
    embeddings = encode_text(feedbacks)
    
    # IDs uniques
    ids = [f"id_{i+j}" for j in range(len(batch))]
    
    # Metadatas (on stocke le code pour pouvoir le comparer après)
    metadatas = [{"code": c, "theme": item.get("theme", "")} for c, item in zip(codes, batch)]
    
    collection.add(
        embeddings=embeddings,
        documents=feedbacks,
        metadatas=metadatas,
        ids=ids
    )

print("✅ Indexation terminée.")

# --- 4. VÉRIFICATION (Sanity Check) ---
print("\n🔎 TEST DE DISTANCE (Doit être < 1.0) :")
test_query = "Error in loop"
test_emb = encode_text([test_query])
results = collection.query(query_embeddings=test_emb, n_results=1)
dist = results['distances'][0][0]
print(f"Distance trouvée : {dist}")

if dist > 1.1:
    print("❌ ATTENTION : La distance est encore > 1. Quelque chose ne va pas.")
else:
    print("✅ VALIDÉ : La distance est cohérente pour du Cosine.")

    # --- 5. UPLOAD ---
    print("\n⬆️ Upload vers Hugging Face Hub...")
    api = HfApi()
    
    # Création du repo s'il n'existe pas
    create_repo(REPO_ID, repo_type="dataset", exist_ok=True, private=True)
    
    api.upload_folder(
        folder_path=OUTPUT_DIR,
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo="chroma_db_storage"
    )
    print("🎉 TERMINÉ ! Base de données mise à jour sur le Hub.")