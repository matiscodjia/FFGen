import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# 1. DONNÉES BRUTES
csv_data = """name,test_mrr,test_recall_at_5,test_recall_at_10
embeddinggemma-300m_BS64,0.4055,0.5798,0.7395
embeddinggemma-300m_BS128,0.3867,0.5551,0.7275
embeddinggemma-300m_BS256,0.3900,0.5498,0.7216
jina-embeddings-v2-base-code_BS64,0.3806,0.5312,0.6805
SFR-Embedding-Code-400M_R_BS256,0.3282,0.4776,0.6735
jina-embeddings-v2-base-code_BS128,0.3314,0.5068,0.6689
gte-large-en-v1.5_BS256,0.3361,0.4845,0.6563
graphcodebert-base_BS128,0.3059,0.4241,0.6551
gte-large-en-v1.5_BS64,0.3158,0.4756,0.6527
SFR-Embedding-Code-400M_R_BS128,0.3339,0.4896,0.6448
gte-large-en-v1.5_BS128,0.3342,0.4793,0.6379
SFR-Embedding-Code-400M_R_BS64,0.3069,0.4826,0.6354
mxbai-embed-large-v1_BS256,0.2883,0.4329,0.6323
mxbai-embed-large-v1_BS128,0.3052,0.4448,0.6310
mxbai-embed-large-v1_BS64,0.3153,0.4548,0.6250
graphcodebert-base_BS256,0.2809,0.4226,0.6185
snowflake-arctic-embed-m_BS64,0.3082,0.4861,0.6145
snowflake-arctic-embed-m_BS256,0.2965,0.4810,0.5979
bge-base-en-v1.5_BS128,0.2688,0.4275,0.5931
bge-base-en-v1.5_BS256,0.2911,0.4054,0.5876
bge-base-en-v1.5_BS64,0.2873,0.4027,0.5833
snowflake-arctic-embed-m_BS128,0.2859,0.4689,0.5793
graphcodebert-base_BS64,0.2869,0.4131,0.5659"""

# 2. PRÉPARATION
df = pd.read_csv(io.StringIO(csv_data))

# Fonction pour nettoyer les noms (pour la lisibilité)
def clean_label(name):
    name = name.replace("google/", "").replace("Salesforce/", "").replace("microsoft/", "")
    name = name.replace("embeddinggemma-300m", "Gemma-300M")
    name = name.replace("SFR-Embedding-Code-400M_R", "SFR-Code")
    name = name.replace("jina-embeddings-v2-base-code", "Jina-v2")
    name = name.replace("graphcodebert-base", "GraphCodeBERT")
    name = name.replace("snowflake-arctic-embed-m", "Arctic-M")
    name = name.replace("bge-base-en-v1.5", "BGE-Base")
    name = name.replace("gte-large-en-v1.5", "GTE-Large")
    name = name.replace("mxbai-embed-large-v1", "MxBai-Large")
    return name.replace("_", " ") # Remplace le underscore par espace

df['name'] = df['name'].apply(clean_label)
df = df.set_index('name')

# Tri décroissant sur la métrique principale (Recall@10)
df = df.sort_values(by='test_recall_at_10', ascending=False)

# Renommage des colonnes pour l'affichage
df.columns = ['MRR', 'Recall@5', 'Recall@10']

# 3. GÉNÉRATION DE LA HEATMAP
plt.figure(figsize=(10, 12), dpi=150) # Format vertical

# sns.heatmap fait tout le travail :
# - annot=True : écrit les chiffres
# - fmt=".1%" : format pourcentage
# - cmap="YlGn" : dégradé Jaune -> Vert (Vert = Meilleur)
ax = sns.heatmap(df, annot=True, fmt=".1%", cmap="YlGn", 
                 cbar_kws={'label': 'Performance Score'}, 
                 linewidths=.5, linecolor='gray')

# Esthétique
plt.title("Leaderboard Final: Comparaison des Modèles", fontsize=14, fontweight='bold', pad=20)
plt.ylabel("") # Pas besoin de label Y, les noms suffisent
plt.xlabel("")
plt.tight_layout()

# Sauvegarde
plt.savefig("leaderboard_heatmap.png")
print("✅ Heatmap générée : leaderboard_heatmap.png")
plt.show()