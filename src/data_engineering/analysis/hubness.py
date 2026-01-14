import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = "matis35/feedbacker-2"
DATASET_ID = "matis35/cf-synt_V2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
TOP_K = 10  # On regarde les 10 voisins les plus proches

def main():
    print(f"🚀 Visualisation Finale (Mode FP32 Safe) sur {DEVICE}...")
    
    # 1. Chargement & Nettoyage
    print("🧹 Chargement du dataset...")
    ds = load_dataset(DATASET_ID, split="train")
    raw_feedbacks = list(set([item["feedback"] for item in ds]))
    # Filtre anti-NaN (chaînes vides)
    unique_feedbacks = [f for f in raw_feedbacks if f and len(f.strip()) > 5]
    print(f"   -> {len(unique_feedbacks)} feedbacks uniques à analyser.")

    # 2. Encodage avec Sécurité FP32
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    embeddings = []
    print("📚 Encodage en cours...")
    
    for i in tqdm(range(0, len(unique_feedbacks), BATCH_SIZE)):
        batch = unique_feedbacks[i : i + BATCH_SIZE]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inputs)
            
            # --- SAFE CASTING FLOAT32 (CRUCIAL) ---
            # On passe tout en FP32 avant de faire des maths
            last_hidden = out.last_hidden_state.to(torch.float32)
            mask = inputs["attention_mask"].to(torch.float32)

            # Pooling
            sum_mask = torch.clamp(mask.sum(1, keepdim=True), min=1e-9)
            emb = torch.sum(last_hidden * mask.unsqueeze(-1), 1) / sum_mask
            
            # Normalisation
            emb = F.normalize(emb, p=2, dim=1)
            embeddings.append(emb.cpu()) # On stocke sur CPU pour économiser VRAM
            
    matrix = torch.cat(embeddings, dim=0) # [N, 1024]

    # 3. Calcul de la Matrice de Similarité
    print("🧮 Calcul des distances (Cosine Similarity)...")
    # Produit scalaire (A . B) = Cosine Similarity car vecteurs normalisés
    sim_matrix = torch.mm(matrix, matrix.T)
    
    # On s'ignore soi-même (diagonale = -1)
    sim_matrix.fill_diagonal_(-1)
    
    # 4. Extraction des Voisins (k-NN)
    values, top_indices = torch.topk(sim_matrix, k=TOP_K, dim=1)
    
    # 5. Calcul des Stats de Hubness
    flat_indices = top_indices.view(-1).numpy()
    # counts[i] = combien de fois le feedback i a été cité comme voisin
    counts = np.bincount(flat_indices, minlength=len(unique_feedbacks))
    
    # Stats Textuelles
    print("-" * 30)
    print(f"Max citations (Le plus gros Hub): {max(counts)}")
    print(f"Moyenne citations: {np.mean(counts):.2f} (Attendu: ~{TOP_K})")
    print(f"Feedbacks isolés (0 citation): {np.sum(counts == 0)}")
    print("-" * 30)

    # ==========================================
    # VISUALISATION 1 : L'HISTOGRAMME (Zoomé)
    # ==========================================
    print("📊 Génération de l'histogramme...")
    plt.figure(figsize=(10, 6))
    
    # On force les 'bins' pour qu'ils soient lisibles entre 0 et le Max
    max_val = max(counts)
    bins = range(0, max_val + 2) # Bins de taille 1
    
    sns.histplot(counts, bins=bins, kde=False, color="skyblue", edgecolor="black")
    
    plt.title(f"Distribution de Hubness (k={TOP_K})")
    plt.xlabel("Nombre de fois où un feedback est cité comme voisin")
    plt.ylabel("Nombre de Feedbacks")
    plt.yscale('log') # Log scale pour voir les petites barres des Hubs
    plt.xlim(0, max_val + 5) # On zoome sur la zone utile
    
    plt.grid(axis='y', alpha=0.3)
    plt.savefig("viz_histogram_final.png")
    print("✅ viz_histogram_final.png sauvegardé.")

    # ==========================================
    # VISUALISATION 2 : LA HEATMAP (Triée)
    # ==========================================
    print("🔥 Génération de la Heatmap structurelle...")
    
    # A. On trie les feedbacks par popularité (Hubness)
    sorted_indices_by_hubness = np.argsort(counts)[::-1] # Décroissant
    
    # B. On sélectionne les échantillons pour la visualisation
    # - 50 "Stars" (Les plus cités à gauche)
    # - 50 "Normaux" (Pris au milieu du classement)
    top_hubs = sorted_indices_by_hubness[:50]
    mid_idx = len(unique_feedbacks) // 2
    normal_feedbacks = sorted_indices_by_hubness[mid_idx : mid_idx + 50]
    
    # L'axe X sera composé de [Hubs ... Normaux]
    x_indices = np.concatenate([top_hubs, normal_feedbacks])
    
    # C. On prend 100 requêtes aléatoires (Axe Y)
    y_indices = np.random.choice(len(unique_feedbacks), 100, replace=False)
    
    # D. Extraction de la sous-matrice
    # On doit repasser par numpy pour le slicing avancé
    sim_numpy = sim_matrix.numpy()
    heatmap_data = sim_numpy[np.ix_(y_indices, x_indices)]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, cmap="viridis", vmin=0.0, vmax=1.0)
    
    plt.title("Carte de Chaleur : Hubs (Gauche) vs Feedbacks Moyens (Droite)")
    plt.xlabel("Feedbacks Cibles (50 Plus Populaires | 50 Moyens)")
    plt.ylabel("100 Feedbacks (Requêtes aléatoires)")
    
    # Trait blanc pour séparer les deux zones
    plt.axvline(x=50, color='white', linestyle='--', linewidth=2)
    
    plt.savefig("viz_heatmap_final.png")
    print("✅ viz_heatmap_final.png sauvegardé.")

if __name__ == "__main__":
    main()