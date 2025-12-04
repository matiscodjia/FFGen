# Analyse du problème et solutions

## 🔴 Problème identifié

**Le modèle overfitte aux exemples spécifiques, pas aux patterns**

### Preuves

1. **Val fonctionne, Test collapse** :
   - Val Loss: 0.73, Acc: 0.72
   - Test Loss: 2.15, Acc: 0.34
   - Gap : +195% loss, -53% accuracy

2. **Variance collapse sur test** :
   - Val variance: 0.0008
   - Test variance: 0.0003 (chute de 62%)

3. **Distribution identique** :
   - Tous les clusters présents dans train/val/test
   - test_train_ratio ≈ 1.0 pour tous les clusters
   - Code length diff = 1.6% seulement

### Conclusion

Le modèle **mémorise les exemples** au lieu d'apprendre les **patterns génériques**.

## 🎯 Hypothèses sur la cause

### Hypothèse #1 : Dataset trop petit (11,806 samples)

Avec seulement ~9,500 exemples de train, le modèle peut les mémoriser :
- 10 epochs × 591 batches × 16 samples = voir chaque exemple ~10 fois
- Le modèle apprend "ce code spécifique va avec ce feedback spécifique"
- Ne généralise pas aux nouveaux exemples

### Hypothèse #2 : Embeddings pré-entraînés trop spécifiques

Le modèle base `Salesforce/SFR-Embedding-Code-400M_R` :
- Pré-entraîné sur du code
- Peut avoir des embeddings très "rigides"
- LoRA (r=16) n'a pas assez de capacité pour adapter

### Hypothèse #3 : InfoNCE avec in-batch negatives insuffisant

Avec batch_size=16 :
- Seulement 15 negatives par positive
- Le modèle apprend à discriminer dans un batch
- Mais pas à généraliser hors du batch

## 💡 Solutions proposées

### Solution #1 : Augmentation de données

**Principe** : Créer des variations des exemples pour forcer la généralisation

```python
def augment_code(code):
    # Variations syntaxiques qui ne changent pas la sémantique
    variations = [
        rename_variables(code),
        reorder_independent_statements(code),
        change_whitespace(code),
        add_comments(code)
    ]
    return random.choice(variations)
```

**Avantages** :
- ✅ Force le modèle à apprendre les patterns, pas les exemples
- ✅ Facile à implémenter
- ✅ Pas besoin de nouvelles données

**Inconvénients** :
- ⚠️ Nécessite des outils de parsing/refactoring
- ⚠️ Risque de changer la sémantique

### Solution #2 : Curriculum Learning

**Principe** : Entraîner d'abord sur des exemples faciles, puis difficiles

```python
# Epoch 1-3: Exemples avec feedback simple
# Epoch 4-6: Exemples avec feedback moyen
# Epoch 7-10: Tous les exemples
```

**Avantages** :
- ✅ Force une généralisation progressive
- ✅ Peut améliorer la convergence

**Inconvénients** :
- ⚠️ Nécessite de définir "facile" vs "difficile"
- ⚠️ Peut ralentir l'entraînement

### Solution #3 : Augmenter la capacité LoRA

**Principe** : Donner plus de flexibilité au modèle pour adapter

```python
# AVANT
lora_r = 16
lora_alpha = 32

# APRÈS
lora_r = 64  # ou 128
lora_alpha = 128
```

**Avantages** :
- ✅ Très simple à implémenter
- ✅ Peut aider si le problème est la capacité

**Inconvénients** :
- ⚠️ Plus de paramètres = plus de risque d'overfitting
- ⚠️ Plus lent à entraîner

### Solution #4 : Hard Negative Mining

**Principe** : Utiliser des negatives plus difficiles (pas juste in-batch)

```python
# Construire une memory bank de tous les embeddings
# Pour chaque positive, chercher les K negatives les plus similaires
# Loss sur ces hard negatives
```

**Avantages** :
- ✅ Force le modèle à mieux discriminer
- ✅ Approche standard en contrastive learning

**Inconvénients** :
- ⚠️ Plus complexe à implémenter
- ⚠️ Plus coûteux en mémoire/compute

### Solution #5 : Dropout plus agressif

**Principe** : Forcer plus de robustesse pendant l'entraînement

```python
# AVANT
lora_dropout = 0.1

# APRÈS
lora_dropout = 0.3  # ou même 0.5
```

**Avantages** :
- ✅ Très simple
- ✅ Réduit l'overfitting

**Inconvénients** :
- ⚠️ Peut ralentir la convergence
- ⚠️ Peut dégrader les perfs sur val aussi

### Solution #6 : Mixup / Cutmix pour embeddings

**Principe** : Mélanger des embeddings pour créer des exemples hybrides

```python
lambda = Beta(alpha=0.2, beta=0.2)
mixed_embedding = lambda * emb1 + (1-lambda) * emb2
```

**Avantages** :
- ✅ Prouvé efficace en vision
- ✅ Régularisation forte

**Inconvénients** :
- ⚠️ Pas évident si ça marche pour du code
- ⚠️ Change la sémantique des exemples

## 🏆 Recommandation

**Commencer par la combinaison** :

1. **Augmenter LoRA rank** (r=64)
   - Simple, rapide à tester
   - Coût : quelques heures de training

2. **Augmenter dropout** (0.3)
   - Simple, gratuit
   - Peut aider immédiatement

3. **Hard negative mining**
   - Si les deux premiers ne suffisent pas
   - Plus complexe mais efficace

## 📊 Métriques à surveiller

Pour valider qu'une solution marche :

| Métrique | Avant | Objectif après fix |
|----------|-------|-------------------|
| Val Loss | 0.73 | 0.60-0.70 |
| Test Loss | 2.15 | **0.70-0.90** |
| Gap Val→Test | +195% | **<30%** |
| Test Variance | 0.0003 | **>0.0005** |
| Test Accuracy | 0.34 | **>0.55** |

Le signal clé : **variance sur test doit rester > 0.0005**

## 🔬 Expériences à mener

### Expérience #1 : LoRA rank + dropout

```python
Config:
  lora_r = 64
  lora_alpha = 128
  lora_dropout = 0.3
  batch_size = 16
  temperature = 0.07
```

Temps estimé : 3-4h de training

### Expérience #2 : Hard negatives

```python
Config:
  Garder LoRA r=16
  Ajouter memory bank avec top-32 hard negatives
  batch_size = 16 (in-batch) + 32 (hard negatives)
```

Temps estimé : 5-6h (plus complexe)

### Expérience #3 : Augmentation de batch size

```python
Config:
  batch_size = 64 (au lieu de 16)
  gradient_accumulation_steps = 4 si pas assez de mémoire
```

Temps estimé : 3-4h

Plus de negatives = meilleure généralisation potentielle
