import os
from transformers import AutoModel, AutoTokenizer

# ================= CONFIGURATION =================
# 1. Le chemin de votre dossier fusionné sur votre ordi
LOCAL_MODEL_PATH = "ready_model"

# 2. Le nom que vous voulez donner au repo sur le Hub
# Remplacer 'matis35' par votre pseudo exact s'il est différent
HF_REPO_ID = "matis35/gemmaembedding-fgdor"
# =================================================

def push_to_hub():
    print(f"🔍 Vérification du modèle local : {LOCAL_MODEL_PATH}...")
    
    try:
        # On charge pour vérifier l'intégrité
        model = AutoModel.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
        print("✅ Chargement local réussi.")
    except Exception as e:
        print(f"❌ Erreur critique : Impossible de charger le modèle local.\n{e}")
        return

    print(f"🚀 Envoi vers Hugging Face Hub : {HF_REPO_ID}...")
    
    try:
        # On ne passe plus de token, il utilise celui de 'huggingface-cli login'
        # token=True force l'utilisation du token local
        model.push_to_hub(HF_REPO_ID, private=True, token=True)
        tokenizer.push_to_hub(HF_REPO_ID, private=True, token=True)
        
        print("\n🎉 TERMINÉ !")
        print(f"👉 Vérifiez votre modèle ici : https://huggingface.co/{HF_REPO_ID}")
        
    except Exception as e:
        print(f"❌ Erreur lors de l'upload : {e}")
        print("👉 Avez-vous bien fait 'huggingface-cli login' dans votre terminal ?")

if __name__ == "__main__":
    push_to_hub()