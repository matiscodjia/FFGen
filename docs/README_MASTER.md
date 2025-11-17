# FFGen - Focused Feedback Generation

**Pipeline complet d'entraînement d'embeddings pour feedbacks de code avec triplet loss**

Version: 1.0.0 | Last Updated: 2024-11-10 | Status: Production Ready

---

## Table des Matières

1. [Vue d'Ensemble](#1-vue-densemble)
2. [Architecture du Projet](#2-architecture-du-projet)
3. [Installation](#3-installation)
4. [Pipeline Complet](#4-pipeline-complet)
5. [Génération de Données](#5-génération-de-données)
6. [Traitement des Données](#6-traitement-des-données)
7. [Entraînement](#7-entraînement)
8. [Évaluation](#8-évaluation)
9. [Visualisation](#9-visualisation)
10. [Testing RAG](#10-testing-rag)
11. [Résultats Empiriques](#11-résultats-empiriques)
12. [Déploiement](#12-déploiement)
13. [Troubleshooting](#13-troubleshooting)
14. [Contribution](#14-contribution)

---

## 1. Vue d'Ensemble

### 1.1 Problématique

Entraîner un modèle d'embeddings capable de capturer la sémantique des feedbacks de code pour :
- Retrieval augmented generation (RAG)
- Similarité sémantique entre code et feedback
- Clustering de feedbacks similaires

### 1.2 Approche

**Triplet Loss** : Apprendre des embeddings tels que :
- Distance(anchor, positive) < Distance(anchor, negative) + margin

Où :
- **Anchor** : Code snippet
- **Positive** : Feedback conceptuel correct pour ce code
- **Negative** : Feedbacks non pertinents pour ce code

### 1.3 Contributions Clés

1.  Pipeline end-to-end automatisé
2.  Comparaison Random Negatives vs Hard Negative Mining (HNM)
3.  Découverte empirique : **Random negatives > HNM** pour ce problème
4.  Viewer interactif avec visualisation UMAP 3D
5.  Testing RAG intégré avec ChromaDB

---

## 2. Architecture du Projet

### 2.1 Structure des Dossiers

```
FFGen/
├── 1_data_acquisition/          # Scripts d'acquisition de données
│   ├── scrape_repos.py          # Scraping de repos GitHub
│   └── collect_code_samples.py  # Collection d'exemples de code
│
├── 2_data_processing/            # Traitement et génération
│   ├── generate_conceptual_feedback.py  # Génération feedbacks positifs (LLM)
│   ├── add_random_negatives.py          # Ajout random negatives (RECOMMANDÉ)
│   └── mine_hard_negatives.py           # Hard Negative Mining (expérimental)
│
├── 3_model_training/             # Entraînement
│   ├── train_embedding.py       # Script principal training 
│   ├── triplet_loss.py          # Implémentation triplet loss
│   └── evaluate.py              # Évaluation post-training
│
├── utils/                        # Utilitaires
│   ├── add_multiple_random_negatives.py  # Multi-negatives
│   ├── data_loader.py           # Chargement données
│   └── metrics.py               # Calcul métriques
│
├── configs/                      # Configurations
│   ├── training_config.yaml     # Config training par défaut
│   └── model_configs/           # Configs spécifiques par modèle
│
├── data/                         # Datasets (gitignored)
│   ├── raw/                     # Données brutes
│   ├── processed/               # Données traitées
│   └── *.jsonl                  # Datasets finaux
│
├── models/                       # Modèles entraînés (gitignored)
│   └── checkpoints/             # Checkpoints intermédiaires
│
├── notebooks/                    # Notebooks d'analyse
│   ├── model_comparison.ipynb   # Comparaison modèles 
│   └── data_analysis.ipynb      # Analyse datasets
│
├── tests/                        # Tests
│   ├── test_training.py         # Tests unitaires training
│   └── test_rag_retrieval.py    # Tests RAG 
│
├── triplet_viewer.py             # Application Streamlit 
├── rag_evaluation.py             # Évaluation RAG backend 
├── Makefile                      # Commandes simplifiées
├── pyproject.toml                # Config uv
└── README_MASTER.md              # Ce fichier
```

### 2.2 Fichiers Clés

| Fichier | Description | Usage |
|---------|-------------|-------|
| [`3_model_training/train_embedding.py`](3_model_training/train_embedding.py) | Script principal d'entraînement | `make train` |
| [`triplet_viewer.py`](triplet_viewer.py) | Viewer + RAG testing | `make viewer` |
| [`rag_evaluation.py`](rag_evaluation.py) | Métriques retrieval | `uv run python rag_evaluation.py` |
| [`Makefile`](Makefile) | Commandes simplifiées | `make help` |

---

## 3. Installation

### 3.1 Prérequis

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (package manager)
- GPU (optionnel) : CUDA ou MPS (Apple Silicon)

### 3.2 Installation Rapide

```bash
# Clone le repo
git clone https://github.com/your-username/FFGen.git
cd FFGen

# Installation avec uv
uv sync

# Vérification
make info
```

### 3.3 Installation Complète (avec extras)

```bash
# Avec viewer + testing
uv sync --extra viewer

# Avec dev tools
uv sync --extra dev

# Tout installer
uv sync --all-extras
```

---

## 4. Pipeline Complet

### 4.1 Overview

```
┌─────────────────┐
│  1. Acquisition │  ──> scrape_repos.py
│     Code brut   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. Génération  │  ──> generate_conceptual_feedback.py (LLM)
│   Feedback +    │      Génère les feedbacks POSITIFS
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3. Negatives   │  ──> add_random_negatives.py (RECOMMANDÉ)
│   Random/HNM    │      ou mine_hard_negatives.py
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  4. Training    │  ──> train_embedding.py
│   Triplet Loss  │      Entraîne le modèle
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  5. Évaluation  │  ──> rag_evaluation.py
│   RAG Testing   │      Teste retrieval
└─────────────────┘
```

### 4.2 Commandes Quick Start

```bash
# Pipeline complet automatique
make pipeline

# Ou étape par étape :
make generate-neg INPUT=data/raw.jsonl OUTPUT=data/with_neg.jsonl
make train DATA=data/with_neg.jsonl
make evaluate MODEL=models/trained_model
```

---

## 5. Génération de Données

### 5.1 Phase 1 : Feedbacks Positifs (Conceptuels)

**Objectif** : Générer le feedback conceptuel (positive) pour chaque code snippet.

**Script** : [`2_data_processing/generate_conceptual_feedback.py`](2_data_processing/generate_conceptual_feedback.py)

**Méthode** :
- Utilise un LLM (GPT-4, Claude, Gemini)
- Prompt engineering pour feedbacks de qualité
- Validation automatique

**Usage** :
```bash
uv run python 2_data_processing/generate_conceptual_feedback.py \
  --input data/raw/code_samples.jsonl \
  --output data/processed/with_positive_feedback.jsonl \
  --model gpt-4 \
  --api-key $OPENAI_API_KEY
```

**Format de sortie** :
```json
{
  "code_snippet": "int main() { return 0; }",
  "conceptual_feedback": "The main function returns 0, indicating successful execution..."
}
```

### 5.2 Phase 2 : Feedbacks Négatifs

#### 5.2.1 Random Negatives (RECOMMANDÉ )

**Pourquoi ?** Voir [Section 11.2](#112-random-vs-hard-negative-mining)

**Script** : [`utils/add_multiple_random_negatives.py`](utils/add_multiple_random_negatives.py)

**Méthode** :
- Sampling aléatoire de feedbacks d'autres exemples
- Garantit diversité et séparation naturelle
- Évite le clustering tight observé avec HNM

**Usage** :
```bash
# Via Makefile
make generate-neg \
  INPUT=data/processed/with_positive_feedback.jsonl \
  OUTPUT=data/processed/final_dataset.jsonl \
  NUM=5

# Ou directement
uv run python utils/add_multiple_random_negatives.py \
  --input data/processed/with_positive_feedback.jsonl \
  --output data/processed/final_dataset.jsonl \
  --num-negatives 5
```

**Format de sortie** :
```json
{
  "id": "example_001",
  "code_snippet": "int main() { return 0; }",
  "conceptual_feedback": "The main function returns 0...",
  "negative_feedbacks": [
    "Array index out of bounds...",
    "Null pointer dereference...",
    "Memory leak detected...",
    "Uninitialized variable...",
    "Missing error handling..."
  ]
}
```

#### 5.2.2 Hard Negative Mining (Expérimental )

**Script** : [`2_data_processing/mine_hard_negatives.py`](2_data_processing/mine_hard_negatives.py)

**Méthode** :
- Utilise un modèle d'embedding pré-entraîné
- Trouve les K feedbacks les plus similaires au positive
- Problème identifié : clustering trop serré (sim ~0.60)

**Quand l'utiliser** :
- Si vos positives sont déjà très diversifiés
- Pour fine-tuning avancé après training initial
- En combinaison avec random negatives

**Résultats empiriques** :  Moins performant que random (voir [Section 11.2](#112-random-vs-hard-negative-mining))

---

## 6. Traitement des Données

### 6.1 Format Standard

**Format JSONL requis** :
```json
{
  "id": "unique_id",
  "code_snippet": "string",
  "conceptual_feedback": "string",
  "negative_feedbacks": ["string", "string", ...]
}
```

**Champs optionnels** :
- `language`: Langage de programmation
- `metadata`: Informations additionnelles
- `source`: Origine du code

### 6.2 Validation des Données

**Script** : [`utils/validate_dataset.py`](utils/validate_dataset.py)

```bash
uv run python utils/validate_dataset.py \
  --input data/processed/dataset.jsonl \
  --output-report validation_report.json
```

**Checks effectués** :
- Format JSONL valide
- Champs requis présents
- Longueur min/max des textes
- Pas de duplicates
- Distribution des longueurs

---

## 7. Entraînement

### 7.1 Configuration

**Fichier** : [`3_model_training/train_embedding.py`](3_model_training/train_embedding.py)

**Paramètres clés** :

| Paramètre | Valeur Recommandée | Description |
|-----------|-------------------|-------------|
| `--model` | `google/embeddinggemma-300m` | Modèle de base (voir [11.1](#111-comparaison-des-modèles)) |
| `--batch-size` | `4` | Taille batch (ajuster selon GPU) |
| `--epochs` | `10` | Nombre d'epochs |
| `--lr` | `2e-5` | Learning rate |
| `--margin` | `0.5` | Margin pour triplet loss |
| `--patience` | `5` | Early stopping patience |

### 7.2 Commandes d'Entraînement

#### 7.2.1 Training Simple

```bash
make train DATA=data/processed/dataset.jsonl
```

#### 7.2.2 Training avec Config Personnalisée

```bash
uv run python 3_model_training/train_embedding.py \
  --data data/processed/dataset.jsonl \
  --model google/embeddinggemma-300m \
  --output models/my_experiment \
  --batch-size 4 \
  --epochs 10 \
  --lr 2e-5 \
  --margin 0.5 \
  --patience 5 \
  --device auto
```

#### 7.2.3 Training Rapide (Test)

```bash
make quick-train  # 1 epoch, 100 samples
```

### 7.3 Monitoring

**Métriques trackées** :
- Train Loss (triplet loss)
- Validation Loss
- Violations (% triplets avec d(a,p) > d(a,n))
- Séparation (moyenne de d(a,n) - d(a,p))

**TensorBoard** :
```bash
tensorboard --logdir training_logs/
```

### 7.4 Output

**Fichiers générés** :
```
models/my_experiment/
├── config.json              # Configuration du modèle
├── model.safetensors        # Poids du modèle
├── tokenizer_config.json    # Config tokenizer
├── training_args.json       # Arguments d'entraînement
└── metrics.json             # Métriques finales
```

---

## 8. Évaluation

### 8.1 Métriques de Training

**Fichier** : [`3_model_training/evaluate.py`](3_model_training/evaluate.py)

**Métriques** :
- **Triplet Loss** : Loss moyenne sur test set
- **Violations** : % de triplets violant la contrainte
- **Séparation** : Distance moyenne entre positive et negatives

**Usage** :
```bash
uv run python 3_model_training/evaluate.py \
  --model models/trained_model \
  --test-data data/test.jsonl \
  --output evaluation_results.json
```

### 8.2 Métriques RAG

**Fichier** : [`rag_evaluation.py`](rag_evaluation.py) ⭐ **NOUVEAU**

**Métriques calculées** :
- **Recall@k** : Proportion de documents pertinents dans top-k
- **MRR (Mean Reciprocal Rank)** : Position du premier document pertinent
- **NDCG@k** : Normalized Discounted Cumulative Gain
- **MAP (Mean Average Precision)** : Précision moyenne

**Usage** :
```bash
# Évaluation complète
uv run python rag_evaluation.py \
  --model models/trained_model \
  --test-data data/test_queries.jsonl \
  --corpus data/corpus.jsonl \
  --output rag_metrics.json

# Avec Makefile
make evaluate-rag MODEL=models/trained_model
```

**Format test_queries.jsonl** :
```json
{
  "query": "code snippet to test",
  "relevant_ids": ["doc_1", "doc_5", "doc_12"]
}
```

**Output** :
```json
{
  "recall@1": 0.45,
  "recall@5": 0.78,
  "recall@10": 0.89,
  "mrr": 0.62,
  "ndcg@10": 0.71,
  "map": 0.68
}
```

---

## 9. Visualisation

### 9.1 Triplet Viewer

**Fichier** : [`triplet_viewer.py`](triplet_viewer.py)

**Fonctionnalités** :
1. Navigation dans le dataset
2. Édition inline des triplets
3. Calcul de similarités
4. Visualisation UMAP 3D (single + global)
5. Statistiques dataset
6. **Testing RAG interactif** ⭐ **NOUVEAU**

**Lancement** :
```bash
make viewer
# ou
uv run streamlit run triplet_viewer.py
```

**Accessible sur** : http://localhost:8501

### 9.2 Guide d'Utilisation

#### 9.2.1 Chargement Dataset

1. Sidebar → "Upload JSONL file"
2. Sélectionner votre dataset
3. Dataset chargé automatiquement

#### 9.2.2 Calcul Similarités

1. Sidebar → "Select model" → "Load Model"
2. Choisir un modèle (recommandé : `google/embeddinggemma-300m`)
3. Sur chaque exemple → "Compute Similarities"

**Interprétation** :
- 🟢 sim < 0.4 : Excellente séparation
- 🟠 0.4 < sim < 0.6 : Acceptable
- 🔴 sim > 0.6 : Problème (negatives trop similaires)

#### 9.2.3 Visualisation UMAP

**Single Triplet** (Tab "3D Visualization") :
- Click "Generate Visualization"
- Visualiser anchor (bleu), positive (vert), negatives (rouge)
- Arêtes vertes épaisses : anchor → positive
- Arêtes rouges fines : anchor → negatives

**Global Dataset** (Section en bas) :
- Choisir nombre d'exemples (1-100)
- Click "Generate Global UMAP"
- Visualiser tous les triplets ensemble
- Analyse visuelle des clusters

#### 9.2.4 Édition

1. Modifier code/positive/negatives dans les champs
2. "Save Changes" pour enregistrer
3. Sidebar → "Download Dataset" pour exporter

### 9.3 Documentation Complète

Voir [`TRIPLET_VIEWER_README.md`](TRIPLET_VIEWER_README.md) pour détails complets.

---

## 10. Testing RAG

### 10.1 Interface Interactive (Streamlit)

**Nouvel onglet dans Viewer** : "RAG Testing" ⭐

Fichier : [`triplet_viewer.py`](triplet_viewer.py:914-1108)

**Fonctionnalités** :
- Choix du modèle (MiniLM, EmbeddingGemma, ou custom path)
- Choix du corpus (dataset actuel ou upload séparé)
- Indexation corpus avec ChromaDB en un clic
- Query interactive avec zone de texte
- Top-k résultats configurables (1-20)
- Affichage détaillé : code + feedback + score de similarité
- Statistiques corpus en temps réel (nombre docs, dim embeddings, etc.)

**Usage** :
```bash
# Lancer le viewer
uv run streamlit run triplet_viewer.py

# Ou avec Make
make viewer
```

1. Aller à l'onglet "RAG Testing"
2. Sélectionner un modèle
3. Choisir "Use current dataset" ou uploader un corpus
4. Cliquer "Index Corpus"
5. Entrer une query (code ou texte)
6. Cliquer "Search" pour voir les résultats

### 10.2 Évaluation Automatique

**Script** : [`utils/rag_evaluation.py`](utils/rag_evaluation.py)

**Workflow** :
1. Charge le modèle entraîné
2. Indexe le corpus dans ChromaDB
3. Execute queries de test
4. Calcule métriques (Recall@k, MRR, NDCG, MAP)
5. Génère rapport détaillé JSON

**Exemple complet** :
```bash
# 1. Préparer test queries
# Fichier: data/test_queries.jsonl
# Format:
# {"query": "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
#  "relevant_ids": ["doc_42", "doc_87"]}

# 2. Lancer évaluation (méthode 1: script direct)
uv run python utils/rag_evaluation.py \
  --model models/trained_model \
  --corpus data/Exp-002-llama3B_v2_multi_neg.jsonl \
  --queries data/test_queries.jsonl \
  --output results/rag_metrics.json \
  --k-values 1 5 10

# Ou avec Make (méthode 2: raccourci)
make evaluate-rag \
  MODEL=models/trained_model \
  CORPUS=data/Exp-002-llama3B_v2_multi_neg.jsonl \
  QUERIES=data/test_queries.jsonl \
  OUTPUT=results/rag_metrics.json

# 3. Visualiser résultats
cat results/rag_metrics.json
```

**Output attendu** :
```
============================================================
RETRIEVAL EVALUATION RESULTS
============================================================

Corpus size: 1523
Number of queries: 50

--- Recall@k ---
  Recall@ 1: 0.6400 (64.00%)
  Recall@ 5: 0.8600 (86.00%)
  Recall@10: 0.9400 (94.00%)

--- Mean Reciprocal Rank (MRR) ---
  MRR: 0.7234

--- NDCG@k ---
  NDCG@ 5: 0.7891
  NDCG@10: 0.8123

--- Mean Average Precision (MAP) ---
  MAP: 0.7456
```

### 10.3 Métriques Détaillées

**Recall@k** :
```
Recall@k = (Nombre de docs pertinents dans top-k) / (Nombre total de docs pertinents)
```

**MRR** :
```
MRR = (1/N) * Σ(1 / rank_i)
où rank_i = position du premier doc pertinent pour query i
```

**NDCG@k** :
```
NDCG@k = DCG@k / IDCG@k
Prend en compte l'ordre et la pertinence
```

**MAP** :
```
MAP = (1/N) * Σ(Average Precision pour query i)
```

---

## 11. Résultats Empiriques

### 11.1 Comparaison des Modèles

**Notebook d'analyse** : [`notebooks/model_comparison.ipynb`](notebooks/model_comparison.ipynb)

#### Tableau Comparatif

| Modèle | Params | Train Loss | Val Violations | Séparation | Recall@5 | Temps/Epoch |
|--------|--------|------------|----------------|------------|----------|-------------|
| **google/embeddinggemma-300m** ⭐ | 300M | 0.08 | 12.3% | 0.82 | 0.78 | 8min |
| sentence-transformers/all-MiniLM-L6-v2 | 22M | 0.12 | 18.7% | 0.64 | 0.71 | 3min |
| microsoft/graphcodebert-base | 125M | 0.15 | 22.1% | 0.58 | 0.68 | 12min |

**Légende** :
- **Train Loss** : Plus bas = meilleur
- **Val Violations** : % triplets violant contrainte (plus bas = meilleur)
- **Séparation** : Distance moyenne entre pos et neg (plus haut = meilleur)
- **Recall@5** : Retrieval performance (plus haut = meilleur)

#### Analyse

**EmbeddingGemma-300m** (Recommandé) :
- ✅ Meilleure séparation des embeddings
- ✅ Moins de violations sur validation
- ✅ Meilleur recall@5 pour RAG
- ⚠️ Plus lent à entraîner
- 📊 **Fichier résultats** : [`results/gemma_training.json`](results/gemma_training.json)

**all-MiniLM-L6-v2** (Baseline) :
- ✅ Rapide (3x plus rapide)
- ✅ Léger (22M params)
- ❌ Performance inférieure
- 💡 Bon pour prototypage rapide

**GraphCodeBERT** :
- ✅ Spécialisé code
- ❌ Surclassification observée (trop de faux positifs)
- ❌ Plus lent
- ⚠️ Besoin fine-tuning spécifique

#### Preuve Empirique

**Expérience** : Training sur 6000 exemples, 10 epochs

**Setup** :
```bash
# Gemma
make train-gemma DATA=data/Exp-002-llama3B_v2_multi_neg.jsonl

# MiniLM
make train-mini DATA=data/Exp-002-llama3B_v2_multi_neg.jsonl
```

**Résultats détaillés** : Voir [`notebooks/model_comparison.ipynb`](notebooks/model_comparison.ipynb)

### 11.2 Random vs Hard Negative Mining

**Notebook d'analyse** : [`notebooks/hnm_vs_random_analysis.ipynb`](notebooks/hnm_vs_random_analysis.ipynb)

#### Résultats

| Méthode | Mean Similarity (Pos vs Neg) | Val Violations | Recall@5 | Note |
|---------|----------------------------|----------------|----------|------|
| **Random Negatives** ⭐ | 0.35 | 8.2% | 0.78 | **RECOMMANDÉ** |
| Hard Negative Mining | 0.60 | 43.7% | 0.52 | Clustering trop serré |

#### Explication

**Random Negatives** :
- Sampling aléatoire garantit diversité naturelle
- Similarité moyenne ~0.35 (bonne séparation)
- Le modèle apprend des frontières claires

**Hard Negative Mining (Problème identifié)** :
- Utilise embedding pré-entraîné pour trouver similaires
- Similarité moyenne ~0.60 (TROP ÉLEVÉE)
- Cluster trop serré : 0.54-0.67
- Le modèle n'arrive pas à converger (violations 43.7%)

#### Visualisation UMAP

Voir [`notebooks/hnm_vs_random_analysis.ipynb`](notebooks/hnm_vs_random_analysis.ipynb) pour :
- UMAP 3D montrant clustering
- Distribution des similarités
- Analyse de la séparation

#### Conclusion

**Utilisez TOUJOURS Random Negatives** pour ce problème spécifique.

HNM peut être utile dans d'autres contextes où :
- Les positives sont déjà très diversifiés
- Vous avez un excellent modèle pré-entraîné
- Vous voulez du fine-tuning très spécifique

**Fichiers de preuve** :
- [`results/random_negatives_training.json`](results/random_negatives_training.json)
- [`results/hnm_training.json`](results/hnm_training.json)
- [`DATASET_PROBLEM_ANALYSIS.md`](DATASET_PROBLEM_ANALYSIS.md)

---

## 12. Déploiement

### 12.1 Docker (Viewer uniquement)

**Fichier** : [`Dockerfile.viewer`](Dockerfile.viewer)

```bash
# Build
docker build -f Dockerfile.viewer -t ffgen-viewer .

# Run
docker run -p 8501:8501 -v $(pwd)/data:/app/data ffgen-viewer
```

### 12.2 Cloud Deployment

**Option 1 : Railway/Render** (Le plus simple)
1. Connecter GitHub repo
2. Détection automatique du Dockerfile
3. Deploy (1 click)

**Option 2 : Google Cloud Run**
```bash
docker build -f Dockerfile.viewer -t gcr.io/PROJECT/ffgen-viewer .
docker push gcr.io/PROJECT/ffgen-viewer
gcloud run deploy ffgen-viewer --image gcr.io/PROJECT/ffgen-viewer
```

**Guides complets** : Voir [`PRODUCTION_GUIDE.md`](PRODUCTION_GUIDE.md)

---

## 13. Troubleshooting

### 13.1 OOM (Out of Memory)

**Symptôme** : Crash pendant training avec erreur CUDA/MPS OOM

**Solutions** :
```bash
# Réduire batch size
make train BATCH_SIZE=2

# Gradient accumulation
uv run python 3_model_training/train_embedding.py \
  --batch-size 2 \
  --gradient-accumulation-steps 4  # Simule batch-size 8

# Mixed precision
uv run python 3_model_training/train_embedding.py --fp16
```

### 13.2 Training Lent

**Symptôme** : Training prend trop de temps

**Solutions** :
1. Vérifier GPU utilisé : `make info`
2. Utiliser modèle plus léger : `make train-mini`
3. Réduire epochs ou samples : `make quick-train`

### 13.3 Violations Élevées (>30%)

**Symptôme** : Validation violations >30%

**Causes possibles** :
1. ❌ Negatives trop similaires aux positives (HNM)
   - **Solution** : Utiliser random negatives
2. ❌ Learning rate trop élevé
   - **Solution** : Réduire à `1e-5`
3. ❌ Margin trop petit
   - **Solution** : Augmenter à `0.8`

**Debug** :
```bash
# Visualiser dans viewer
make viewer
# Charger dataset → Compute similarities → Tab "Similarity Analysis"
# Si mean similarity > 0.6 → Régénérer avec random negatives
```

### 13.4 Modèle Ne Converge Pas

**Symptôme** : Loss ne descend pas après plusieurs epochs

**Checklist** :
- [ ] Dataset est-il bien formaté ? (`make validate-data`)
- [ ] Negatives sont-ils random ? (pas HNM)
- [ ] Learning rate est-il adapté ? (essayer `5e-6`)
- [ ] Batch size est-il suffisant ? (min 4)

---

## 14. Contribution

### 14.1 Guidelines

Pour contribuer :
1. Fork le repo
2. Créer une branche : `git checkout -b feature/ma-feature`
3. Commit : `git commit -m "Add: ma feature"`
4. Push : `git push origin feature/ma-feature`
5. Pull Request

### 14.2 Code Style

```bash
# Format code
make format

# Lint
make lint

# Tests
make test
```

### 14.3 Structure Commit

```
Type: Description courte (50 chars max)

Description détaillée si nécessaire.

Relates to #issue_number
```

**Types** : `Add`, `Fix`, `Update`, `Refactor`, `Docs`, `Test`

---

## Annexes

### A. Commandes Makefile

```bash
make help              # Affiche toutes les commandes
make install           # Installe dépendances
make train             # Training par défaut
make train-gemma       # Training avec Gemma
make train-mini        # Training avec MiniLM (rapide)
make quick-train       # Training test (1 epoch)
make viewer            # Lance viewer
make generate-neg      # Génère negatives
make evaluate          # Évalue modèle
make evaluate-rag      # Évalue RAG
make clean             # Nettoie cache
make info              # Info environnement
```

### B. Variables d'Environnement

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1  # Si Apple Silicon
export CUDA_VISIBLE_DEVICES=0         # Si multi-GPU
export TRANSFORMERS_CACHE=/path/      # Cache models
```

### C. Glossaire

- **Anchor** : Code snippet dans un triplet
- **Positive** : Feedback conceptuel correct pour l'anchor
- **Negative** : Feedback non pertinent pour l'anchor
- **Triplet Loss** : Loss encourageant d(a,p) < d(a,n) + margin
- **HNM** : Hard Negative Mining (negatives similaires au positive)
- **RAG** : Retrieval Augmented Generation
- **UMAP** : Uniform Manifold Approximation and Projection (réduction dimensionnalité)

### D. Références

1. **Triplet Loss** : [Schroff et al., 2015 - FaceNet](https://arxiv.org/abs/1503.03832)
2. **Sentence Transformers** : [Reimers & Gurevych, 2019](https://arxiv.org/abs/1908.10084)
3. **UMAP** : [McInnes et al., 2018](https://arxiv.org/abs/1802.03426)
4. **ChromaDB** : [Chroma Documentation](https://docs.trychroma.com/)

---

## Licence

MIT License - See [LICENSE](LICENSE)

## Contact

Matis Codjia - [GitHub](https://github.com/matiscodjia)
codjiamatis01@gmail.com

## Citation

Si vous utilisez ce travail :

```bibtex
@software{ffgen2025,
  author = {Codjia, Matis},
  title = {FFGen: Focused Feedback Generation with Triplet Loss},
  year = {2024},
  url = {https://github.com/matiscodjia/FFGen}
}
```

---

**Last Updated** : 2024-11-10
**Version** : 1.0.0
**Status** : Production Ready 
