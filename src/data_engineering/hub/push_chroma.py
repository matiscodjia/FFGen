import os
from huggingface_hub import HfApi, create_repo

# --- CONFIGURATION ---
# Remplace par ton pseudo et le nom que tu veux donner à ton dataset
REPO_ID = "matis35/chroma-rag-storage"  
# Le chemin exact du dossier que tu m'as montré dans la capture
LOCAL_FOLDER_PATH = "/Users/matiscodjia/Dev/06_research/FFGen/src/deployment/streamlit_app/streamlit_rag_viewer/chroma_db_storage" 

def upload_chroma_db():
    print(f"Préparation de l'upload pour {REPO_ID}...")
    
    api = HfApi()
    
    # 1. Création du Dataset (s'il n'existe pas déjà)
    # On le met en privé par sécurité
    repo_url = create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True, 
        private=True
    )
    print(f"Dataset prêt : {repo_url}")

    # 2. Upload du dossier complet
    # L'argument 'path_in_repo' définit le nom du dossier sur le Hub.
    # On garde le même nom pour ne pas te perdre.
    print(f"Envoi du dossier '{LOCAL_FOLDER_PATH}' vers le Hub (ça peut prendre un moment)...")
    
    api.upload_folder(
        folder_path=LOCAL_FOLDER_PATH,
        repo_id=REPO_ID,
        repo_type="dataset",
        path_in_repo="chroma_db_storage" 
    )
    
    print("------------------------------------------------")
    print("SUCCÈS ! Ta base de données est sauvegardée.")
    print(f"Lien : https://huggingface.co/datasets/{REPO_ID}/tree/main/chroma_db_storage")
    print("------------------------------------------------")

if __name__ == "__main__":
    # Vérification simple que le dossier existe avant de lancer
    if os.path.exists(LOCAL_FOLDER_PATH):
        upload_chroma_db()
    else:
        print(f"Erreur : Le dossier '{LOCAL_FOLDER_PATH}' est introuvable ici.")