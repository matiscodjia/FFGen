# Multi-Configuration LoRA Training

## Vue d'ensemble

Le script `train_lora_multiconfig.py` permet de lancer plusieurs entraînements LoRA successifs avec des configurations différentes. Il suffit d'éditer un dictionnaire Python pour définir toutes les configurations.

## Utilisation rapide

### 1. Éditer les configurations

Ouvrir `train_lora_multiconfig.py` et modifier le dictionnaire `TRAINING_CONFIGS` :

```python
TRAINING_CONFIGS = [
    {
        "name": "gemma_300m_r8_noquant",
        "model": "google/embeddinggemma-300m",
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": None,  # Auto-detect
        "use_8bit": False,
        "gradient_checkpointing": False,
        "batch_size": 4,
        "epochs": 10,
        "lr": 2e-5,
        "margin": 0.5,
        "max_length": 512,
        "patience": 5,
    },
    # Ajouter d'autres configurations ici...
]
```

### 2. Lancer l'entraînement

```bash
# Avec le dataset par défaut (multi_neg_10)
python 4_model_training/train_lora_multiconfig.py

# Avec un dataset spécifique
python 4_model_training/train_lora_multiconfig.py --dataset data/mon_dataset.jsonl

# Limiter le nombre d'exemples (pour tester)
python 4_model_training/train_lora_multiconfig.py --max-samples 1000

# Afficher les configurations sans lancer l'entraînement
python 4_model_training/train_lora_multiconfig.py --configs-only
```

## Paramètres de configuration

### Paramètres du modèle
- `name`: Nom de l'expérience (utilisé pour les dossiers de sortie)
- `model`: Modèle de base HuggingFace (ex: `"google/embeddinggemma-300m"`)

### Paramètres LoRA
- `lora_r`: Rang LoRA (8, 16, 32...) - Plus élevé = plus de paramètres
- `lora_alpha`: Facteur d'échelle LoRA (généralement 2× le rang)
- `lora_dropout`: Dropout pour LoRA (0.05-0.1)
- `target_modules`: Modules à adapter (None = auto-détection)

### Optimisation mémoire
- `use_8bit`: Quantisation 8-bit (économise ~50% de mémoire)
- `gradient_checkpointing`: Économise de la mémoire (ralentit l'entraînement)

### Hyperparamètres d'entraînement
- `batch_size`: Taille du batch (2-8 selon mémoire)
- `epochs`: Nombre d'époques (10-20)
- `lr`: Learning rate (1e-5 à 3e-5)
- `margin`: Marge de triplet loss (0.5)
- `max_length`: Longueur max des séquences (512)
- `patience`: Patience pour early stopping (5-7)

## Structure des sorties

```
lora_output/
├── gemma_300m_r8_noquant_20251119_103045/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── config.json
│   └── tokenizer files...
├── gemma_300m_r16_8bit_20251119_113045/
│   └── ...
└── ...

logs/
├── gemma_300m_r8_noquant_20251119_103045/
│   ├── final_report.png
│   └── config.json
├── gemma_300m_r16_8bit_20251119_113045/
│   └── ...
├── training_summary.json
└── ...
```

### Fichiers de sortie

- **lora_output/**: Contient uniquement les meilleurs modèles finaux
  - Chaque sous-dossier contient un modèle LoRA complet
  - Prêt à être chargé avec PEFT

- **logs/**: Contient les logs et graphiques
  - `final_report.png`: Graphiques d'entraînement
  - `config.json`: Configuration utilisée
  - `training_summary.json`: Résumé global de tous les entraînements

## Exemples de configurations

### Configuration économique (peu de mémoire)
```python
{
    "name": "eco_r8_8bit",
    "model": "google/embeddinggemma-300m",
    "lora_r": 8,
    "lora_alpha": 16,
    "use_8bit": True,
    "gradient_checkpointing": True,
    "batch_size": 2,
    "epochs": 10,
    "lr": 2e-5,
}
```

### Configuration équilibrée
```python
{
    "name": "balanced_r16",
    "model": "google/embeddinggemma-300m",
    "lora_r": 16,
    "lora_alpha": 32,
    "use_8bit": False,
    "gradient_checkpointing": False,
    "batch_size": 4,
    "epochs": 15,
    "lr": 2e-5,
}
```

### Configuration haute performance
```python
{
    "name": "perf_r32",
    "model": "google/embeddinggemma-300m",
    "lora_r": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.1,
    "use_8bit": False,
    "gradient_checkpointing": False,
    "batch_size": 8,
    "epochs": 20,
    "lr": 1e-5,
    "patience": 7,
}
```

## Fonctionnalités

### Gestion automatique des checkpoints
- Sauvegarde automatique du meilleur modèle pendant l'entraînement
- Nettoyage automatique des checkpoints intermédiaires
- Seul le meilleur modèle est conservé dans `lora_output/`

### Logs et visualisations
- Graphiques d'entraînement automatiques (.png dans `logs/`)
- Résumé JSON de chaque expérience
- Résumé global de tous les entraînements

### Gestion des erreurs
- Continue avec la configuration suivante en cas d'erreur
- Log détaillé des erreurs dans le résumé final
- Nettoyage automatique des fichiers temporaires

## Monitoring pendant l'entraînement

Le script affiche en temps réel :
- Configuration en cours
- Progress bar avec loss
- Métriques de validation (loss, violations, séparation)
- Best model save notifications

## Résumé final

À la fin, le script affiche :
- Nombre total d'expériences
- Succès / Échecs
- **Meilleur modèle** avec sa configuration et son score

## Tips et astuces

### Optimisation GPU/MPS
- Commencer avec `batch_size=2` et augmenter progressivement
- Utiliser `use_8bit=True` si mémoire limitée
- `gradient_checkpointing=True` réduit la mémoire mais ralentit

### Choix du rang LoRA
- `r=8` : Rapide, peu de paramètres, bon pour fine-tuning léger
- `r=16` : Équilibré, bon compromis général
- `r=32` : Plus de capacité, nécessite plus de données

### Learning rate
- Modèles petits (300M) : 2e-5 à 3e-5
- Modèles moyens (1B+) : 1e-5 à 2e-5
- Avec quantisation 8-bit : augmenter légèrement (×1.5)

### Early stopping
- Dataset petit (<10k) : `patience=3`
- Dataset moyen (10k-100k) : `patience=5`
- Dataset large (>100k) : `patience=7`

## Dépannage

### Out of Memory
1. Réduire `batch_size`
2. Activer `use_8bit=True`
3. Activer `gradient_checkpointing=True`
4. Réduire `max_length`
5. Réduire `lora_r`

### Entraînement trop lent
1. Augmenter `batch_size` (si mémoire disponible)
2. Désactiver `gradient_checkpointing`
3. Réduire `max_length` si séquences courtes
4. Utiliser un rang LoRA plus petit

### Loss ne diminue pas
1. Augmenter `lr` (×2 ou ×3)
2. Augmenter `lora_r`
3. Vérifier les données (négatifs suffisamment différents)
4. Augmenter `margin`

## Charger un modèle entraîné

```python
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

# Charger le modèle de base
base_model = AutoModel.from_pretrained("google/embeddinggemma-300m")
tokenizer = AutoTokenizer.from_pretrained("google/embeddinggemma-300m")

# Charger l'adapter LoRA
model = PeftModel.from_pretrained(
    base_model,
    "lora_output/gemma_300m_r8_noquant_20251119_103045"
)

# Utiliser le modèle
inputs = tokenizer("code snippet", return_tensors="pt")
outputs = model(**inputs)
```

## Notes importantes

- Les checkpoints intermédiaires sont **automatiquement supprimés**
- Seuls les **meilleurs modèles** sont conservés
- Les graphiques sont dans `logs/`, pas dans `lora_output/`
- Le dataset par défaut est `multi_neg_10.jsonl`
- Chaque expérience a un timestamp unique pour éviter les conflits
