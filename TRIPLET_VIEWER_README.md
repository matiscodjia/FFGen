# FFGen Triplet Viewer

Application web pour visualiser, éditer et analyser les triplets d'entraînement avec visualisations UMAP 3D interactives.

## Fonctionnalités

### Interface
- Design minimal et épuré
- Navigation fluide entre les exemples
- Édition en temps réel de tous les champs
- Visualisations 3D interactives avec UMAP

### Analyse
- Calcul automatique des similarités (positive vs negatives)
- Métriques détaillées : mean, min, max, std
- Visualisation par couleur : rouge (>0.6), orange (0.4-0.6), vert (<0.4)
- Histogrammes et statistiques du dataset

### Visualisation UMAP
- **Single triplet** : Visualiser un triplet individuel en 3D
- **Global dataset** : Visualiser N triplets ensemble avec toutes leurs connexions
- Arêtes vertes épaisses : code → positive
- Arêtes rouges fines : code → negatives
- Interactive 3D avec Plotly (rotation, zoom, hover)

### Édition
- Modifier le code snippet, positive feedback, negatives
- Ajouter/supprimer des negatives
- Télécharger le dataset modifié en JSONL

## Lancement

```bash
# Depuis le dossier FFGen
./start_viewer.sh

# Ou directement
streamlit run triplet_viewer.py
```

L'application s'ouvre automatiquement sur **http://localhost:8501**

## Guide d'Utilisation

### 1. Charger un Dataset

**Sidebar → Upload JSONL file**

Formats supportés :
```json
{
  "id": "example_001",
  "code_snippet": "int main() { ... }",
  "conceptual_feedback": "...",
  "negative_feedbacks": ["...", "...", "..."]
}
```

Format legacy :
```json
{
  "code_snippet": "...",
  "refined_feedback": "...",
  "hard_negative_feedback": "..."
}
```

### 2. Charger un Modèle d'Embedding

**Sidebar → Select model → Load Model**

Modèles disponibles :
- `sentence-transformers/all-MiniLM-L6-v2` : Rapide, léger (défaut)
- `google/embeddinggemma-300m` : Meilleure séparation (recommandé)
- `microsoft/graphcodebert-base` : Spécialisé code

### 3. Naviguer et Éditer

- **Navigation** : Boutons Previous/Next ou jump direct
- **Édition** : Modifier code, positive, negatives inline
- **Sauvegarder** : Bouton "Save Changes"
- **Export** : Sidebar → "Download Dataset" (télécharge le JSONL modifié)

### 4. Calculer les Similarités

**Bouton "Compute Similarities"** sur chaque exemple

Interprétation :
- `sim > 0.6` 🔴 : Negative trop similaire (problème)
- `0.4 < sim < 0.6` 🟠 : Moyennement similaire
- `sim < 0.4` 🟢 : Bien séparé (optimal)

Règle d'or : Pour triplet loss avec margin=0.5, on veut `mean(similarities) < 0.5`

### 5. Visualiser en 3D UMAP

#### Single Triplet (Tab "3D Visualization")
1. Click "Generate Visualization"
2. Explore l'espace 3D interactif
3. Hover sur les points pour voir les détails

#### Global Dataset (Section en bas)
1. Choisir le nombre d'exemples (1-100)
2. Click "Generate Global UMAP"
3. Visualiser tous les triplets ensemble avec leurs connexions
4. Analyse visuelle de la distribution et des clusters

**Légende :**
- 🔵 Bleu : Anchor (code)
- 🟢 Vert : Positive feedback
- 🔴 Rouge : Negative feedbacks
- Ligne verte épaisse : anchor → positive
- Ligne rouge fine : anchor → negative

### 6. Analyser les Statistiques

**Tab "Current Example"** : Stats de l'exemple courant
**Tab "Dataset Stats"** : Distributions globales (code length, negatives count)
**Tab "Similarity Analysis"** : Métriques de qualité, histogrammes

## Use Cases

### Valider la qualité des Hard Negatives
```
1. Charger dataset HNM
2. Load embedding model (embeddinggemma-300m)
3. Compute similarities sur plusieurs exemples
4. Tab "Similarity Analysis" → Vérifier mean
5. Si mean > 0.6 → HNM trop similaires, utiliser random negatives
```

### Comparer Random vs Hard Negatives
```
1. Charger dataset_hnm.jsonl
2. Generate Global UMAP (20-30 exemples)
3. Observer les distances et clusters
4. Répéter avec dataset_random.jsonl
5. Comparer les distributions visuelles
```

### Nettoyer le Dataset
```
1. Parcourir les exemples
2. Compute similarities
3. Supprimer negatives avec sim > 0.7
4. Corriger feedbacks mal générés
5. Download edited dataset
```

### Analyser Avant Training
```
1. Load dataset + model
2. Tab "Dataset Stats" → Vérifier distributions
3. Generate Global UMAP → Identifier clusters problématiques
4. Compute similarities sur échantillon
5. Tab "Similarity Analysis" → Quality assessment
6. Si OK → Lancer training, sinon régénérer negatives
```

## Workflow Complet

```bash
# 1. Générer dataset avec negatives
python utils/add_multiple_random_negatives.py \
  --input data/Exp-002-llama3B_v2.jsonl \
  --output data/Exp-002-llama3B_v2_multi_neg.jsonl \
  --num-negatives 5

# 2. Lancer viewer
./start_viewer.sh

# 3. Dans l'app:
#    - Upload le dataset
#    - Load model: embeddinggemma-300m
#    - Generate Global UMAP (30 exemples)
#    - Compute similarities sur échantillon
#    - Vérifier quality metrics

# 4. Si qualité OK, lancer training
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python 3_model_training/train_embedding.py \
  --data data/Exp-002-llama3B_v2_multi_neg.jsonl \
  --model google/embeddinggemma-300m \
  --batch-size 4 \
  --epochs 10 \
  --lr 2e-5 \
  --margin 0.5
```

## Configuration Technique

### Métriques de Similarité

Cosine similarity entre :
- **Positive** : conceptual_feedback
- **Negative** : chaque negative_feedback

Plus la similarité est élevée, plus les embeddings sont proches dans l'espace vectoriel.

### UMAP (Uniform Manifold Approximation and Projection)

- Réduction de dimensionnalité non-linéaire
- Preserve la structure locale et globale
- Paramètres : `n_neighbors=15`, `n_components=3`
- Plus robuste que t-SNE pour les grandes distances

### Quality Assessment

L'app évalue automatiquement :
- Mean similarity < 0.5 : ✅ Good
- Mean similarity 0.5-0.6 : ⚠️ Medium
- Mean similarity > 0.6 : 🚨 Bad (negatives trop similaires)

## Troubleshooting

### L'app ne démarre pas
```bash
uv pip install streamlit plotly sentence-transformers umap-learn
```

### Modèle ne charge pas
Le modèle se télécharge au premier chargement. Patience lors de la première utilisation.

### UMAP génération lente
Normal pour >50 exemples. Réduire le nombre d'exemples ou utiliser un modèle plus rapide (all-MiniLM-L6-v2).

### Dataset ne charge pas
- Vérifier format JSONL (une ligne = un JSON valide)
- Champs requis : `code_snippet`, `conceptual_feedback` ou `refined_feedback`, `negative_feedbacks` ou `hard_negative_feedback`

## Performance

- **Chargement dataset** : < 1s pour 6000 exemples
- **Calcul similarité** : ~0.5s par exemple (dépend du modèle)
- **UMAP 3D** : ~2-5s pour 10-30 exemples
- **Navigation** : Instantanée (cache Streamlit)

---

**Built for FFGen**
