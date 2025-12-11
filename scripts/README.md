# Industrial Training Pipeline for LoRA Adapters

Pipeline massif d'entraînement pour l'étude comparative des adapters LoRA sur différentes configurations.

## 📋 Vue d'ensemble

Ce pipeline entraîne **16 configurations d'adapters** :
- **2 modèles de base** :
  - `jinaai/jina-embeddings-v3` (Jina Code Embed)
  - `google/gemma-2-2b-it` (Gemma 300M Embedding)
- **2 datasets** :
  - RAFT original (`data/raft_dataset.jsonl`)
  - RAFT Ultra-Clean (`data/ultra_clean_final_dataset/`)
- **4 batch sizes** : 32, 64, 128, 256

## 🏗️ Architecture

```
scripts/
├── experiment_pipeline.py      # Orchestrateur principal
├── train_single_experiment.py  # Entraînement d'une config
└── analyze_experiments.py      # Analyse et visualisation

experiments/
├── runs/                       # Modèles entraînés
│   ├── jina-code-embed_raft_bs32/
│   ├── jina-code-embed_raft_bs64/
│   └── ...
├── logs/                       # Logs d'entraînement
├── results/                    # Résultats JSON
└── analysis/                   # Graphiques et rapports
```

## 🚀 Utilisation

### 1. Installation des dépendances

```bash
# Avec uv (recommandé)
uv sync

# Ou avec pip
pip install -e .
```

### 2. Vérification avant lancement (Dry Run)

```bash
python scripts/experiment_pipeline.py --dry-run
```

Cela affiche :
- Les 16 configurations qui seront entraînées
- Le plan d'expérimentation
- Sans lancer l'entraînement

### 3. Lancement du pipeline complet

```bash
python scripts/experiment_pipeline.py --workspace experiments
```

**Options disponibles** :
- `--workspace DIR` : Dossier de travail (défaut: `experiments`)
- `--dry-run` : Simulation sans entraînement
- `--resume-from FILE` : Reprendre depuis un point
- `--stop-on-failure` : Arrêter si une expérience échoue

**Comportement** :
- Demande confirmation avant de démarrer
- Continue même si une expérience échoue (sauf avec `--stop-on-failure`)
- Sauvegarde automatiquement les métriques
- Push automatique sur HuggingFace Hub

### 4. Analyse des résultats

Une fois les entraînements terminés :

```bash
python scripts/analyze_experiments.py --workspace experiments
```

**Génère** :
- 📊 `batch_size_effect.png` - Impact de la taille du batch
- 📊 `dataset_quality_effect.png` - Impact de la qualité des données
- 📊 `model_comparison.png` - Comparaison des modèles
- 📊 `training_curves.png` - Dynamiques d'entraînement
- 📝 `summary_report.txt` - Rapport textuel complet
- 📄 `all_experiments.csv` - Toutes les métriques en CSV

## 📊 Métriques trackées

Pour chaque expérience :

### Métriques d'entraînement (par step)
- **Loss** : NCE loss symétrique
- **Accuracy** : Précision de matching

### Métriques d'évaluation (si validation disponible)
- **MRR** (Mean Reciprocal Rank) : Qualité du ranking
- **Recall@1** : Précision top-1
- **Recall@5** : Précision top-5
- **Recall@10** : Précision top-10

### Métriques globales
- Best/Final loss et accuracy
- Durée d'entraînement
- Paramètres entraînables

## 🔧 Configuration des expériences

### Hyperparamètres par défaut

```python
learning_rate = 2e-4
num_epochs = 3
warmup_steps = 100
temperature = 0.07

# LoRA
lora_r = 16
lora_alpha = 32
lora_dropout = 0.1

# Tokenization
code_max_length = 512
feedback_max_length = 256
```

### Modifier une configuration

Éditez `scripts/experiment_pipeline.py` :

```python
class ExperimentConfig:
    learning_rate: float = 2e-4  # ← Modifier ici
    num_epochs: int = 3           # ← Ou ici
    # ...
```

## 📁 Structure des résultats

### Par expérience

Chaque expérience produit :

```
experiments/runs/jina-code-embed_raft_bs32/
├── config.json              # Configuration complète
├── metrics.jsonl            # Métriques step-by-step
├── summary.json             # Résumé des métriques
├── training_state.json      # État de l'entraînement
├── adapter_model.bin        # Adapter LoRA
├── adapter_config.json      # Config de l'adapter
└── logs/                    # TensorBoard logs
```

### Logs globaux

```
experiments/
├── experiment_plan.json           # Plan complet
├── pipeline_report_*.json         # Rapport du pipeline
└── analysis/                      # Visualisations
```

## 🎯 Objectif de l'étude

Démontrer que :
1. **Batch size croissant** → Meilleures performances
2. **Données plus propres** → Meilleur modèle
3. **Comparaison Jina vs Gemma** → Quel modèle excelle ?

## 🔍 Analyse approfondie

### Lire les métriques d'une expérience

```python
import json

# Charger le résumé
with open("experiments/runs/jina-code-embed_raft_bs32/summary.json") as f:
    summary = json.load(f)

print(f"Best MRR: {summary['best_mrr']}")
print(f"Best Loss: {summary['best_loss']}")
```

### Comparer deux expériences

```python
import pandas as pd

df = pd.read_csv("experiments/analysis/all_experiments.csv")

# Comparer Jina vs Gemma
jina = df[df['model'] == 'jina-code-embed']
gemma = df[df['model'] == 'gemma-embedding-300m']

print("Jina mean accuracy:", jina['best_accuracy'].mean())
print("Gemma mean accuracy:", gemma['best_accuracy'].mean())
```

## 🐛 Dépannage

### Erreur CUDA Out of Memory

Réduire le batch size ou activer gradient checkpointing :

```python
# Dans train_single_experiment.py
training_args = TrainingArguments(
    gradient_checkpointing=True,  # ← Déjà activé
    per_device_train_batch_size=32,  # ← Réduire si nécessaire
)
```

### Échec de push vers Hub

Vérifier l'authentification :

```bash
huggingface-cli login
```

Le pipeline continue même si le push échoue.

### Reprendre après une interruption

```bash
python scripts/experiment_pipeline.py --resume-from experiments/experiment_plan.json
```

## 📈 Interprétation des résultats

### Graphiques générés

1. **batch_size_effect.png**
   - 4 courbes montrant l'évolution des métriques avec la taille du batch
   - Devrait montrer une amélioration avec batch size croissant

2. **dataset_quality_effect.png**
   - Bars comparant original vs ultra-clean
   - Devrait montrer l'avantage des données nettoyées

3. **model_comparison.png**
   - Heatmaps et comparaisons directes Jina vs Gemma
   - Score composite normalisé

4. **training_curves.png**
   - Dynamiques d'entraînement
   - Convergence et stabilité

### Rapport textuel

Le fichier `summary_report.txt` contient :
- Statistiques globales
- Meilleures configurations
- Analyses par dimension (batch, dataset, modèle)
- Table détaillée complète

## 🚀 Exemple complet

```bash
# 1. Vérifier la config
python scripts/experiment_pipeline.py --dry-run

# 2. Lancer (avec confirmation)
python scripts/experiment_pipeline.py

# 3. Analyser
python scripts/analyze_experiments.py

# 4. Consulter les résultats
ls experiments/analysis/
cat experiments/analysis/summary_report.txt
```

## 🔗 Hub Models

Les adapters sont pushés sur HuggingFace Hub :
- `matiscodjia/ffgen-jina-code-raft-bs32`
- `matiscodjia/ffgen-jina-code-raft-bs64`
- `matiscodjia/ffgen-jina-code-raft-bs128`
- `matiscodjia/ffgen-jina-code-raft-bs256`
- `matiscodjia/ffgen-jina-code-raft-ultra-clean-bs32`
- ... (16 au total)

## 📝 Notes importantes

1. **Durée estimée** : ~2-4h par expérience selon le GPU (total ~32-64h pour 16)
2. **Stockage** : ~500MB-1GB par expérience (total ~8-16GB)
3. **GPU recommandé** : NVIDIA avec ≥16GB VRAM (A100, V100, RTX 4090)
4. **Pause entre expériences** : 10s pour éviter la surchauffe

## 🤝 Support

Pour les questions :
- Consulter les logs : `experiments/logs/`
- Vérifier les erreurs dans le rapport final
- Examiner les métriques individuelles dans `experiments/runs/`
