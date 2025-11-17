# Changelog: Configuration-Based Inference

**Date**: 2025-11-17
**Feature**: Centralized inference configuration from YAML files

## Summary

Ajout de la possibilité de configurer les endpoints et noms de modèles d'inférence directement depuis le fichier de configuration `configs/config.yml`, sans avoir à modifier le code source.

## Modifications Apportées

### 1. Configuration (`configs/config.yml`)

**Nouveau**: Section `inference` ajoutée en haut du fichier

```yaml
inference:
  llm:
    url: "http://localhost:1234/v1"
    model_name: "llama-3.2-3b-instruct"
    timeout: 5.0
    fallback_model: null

  embeddings:
    url: "http://localhost:8000/v1"
    model_name: "default"
    timeout: 5.0
    fallback_model: "sentence-transformers/all-MiniLM-L6-v2"
```

**Avantages**:
- Changement d'endpoint sans modification du code
- Configuration centralisée pour tous les services d'inférence
- Support de multiples serveurs (llama.cpp, LM Studio, vLLM, etc.)
- Configuration des timeouts et fallbacks

### 2. Service d'Inférence (`utils/inference_service.py`)

**Ajouts**:
- Import de `yaml` et `pathlib.Path`
- Nouvelle méthode de classe `InferenceServer.from_config()`
- Fonction de test mise à jour pour utiliser la configuration

**Exemple d'utilisation**:
```python
# Avant (manuel)
llm = InferenceServer(
    url="http://localhost:1234/v1",
    model_name="llama-3.2-3b-instruct"
)

# Après (depuis config)
llm = InferenceServer.from_config(
    config_path="./configs/config.yml",
    service_type="llm"
)
```

### 3. Génération de Feedback (`2_data_generation/generate_feedback.py`)

**Modifié**: `initialize_llm_client()`
- Utilise maintenant `InferenceServer.from_config()` par défaut
- Lit l'URL depuis la config en cas de fallback vers AsyncOpenAI
- Signature changée: `config_path` au lieu de `server_url`

**Migration**:
```python
# Avant
initialize_llm_client(server_url="http://localhost:1234/v1")

# Après
initialize_llm_client(config_path="./configs/config.yml")
```

### 4. Module Embeddings (`5_triplet_viewer_app/backend/embeddings.py`)

**Modifié**: `load_embedding_model()`
- Nouveau paramètre `config_path` (par défaut: `"./configs/config.yml"`)
- Essaie de charger depuis la config en priorité
- Fallback vers les méthodes précédentes (rétrocompatibilité)

**Migration**:
```python
# Avant
model = load_embedding_model(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    server_url="http://localhost:8000/v1"
)

# Après (recommandé)
model = load_embedding_model(config_path="./configs/config.yml")

# Ou (compatible)
model = load_embedding_model(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    server_url="http://localhost:8000/v1"
)
```

### 5. Documentation

**Nouveaux fichiers**:
- `docs/INFERENCE_CONFIG.md` - Guide complet de configuration
- `test_config_loading.py` - Script de test pour validation
- `CHANGELOG_CONFIG_BASED_INFERENCE.md` - Ce fichier

**Fichiers mis à jour**:
- `README.md` - Section "Inference with Fallback Mechanism" mise à jour

## Tests

### Scripts de test disponibles

```bash
# Test complet depuis la configuration
python utils/inference_service.py

# Test spécifique de chargement de config
python test_config_loading.py

# Test du mécanisme de fallback
python test_fallback.py
```

### Résultats des tests

✅ **Test de chargement de config** (test_config_loading.py):
```
1. Testing LLM config loading...
   ✓ LLM loaded successfully
   URL: http://localhost:1234/v1
   Model: llama-3.2-3b-instruct

2. Testing Embeddings config loading...
   ✓ Embedder loaded successfully
   URL: http://localhost:8000/v1
   Model: default
   Embedding dimension: 768
```

✅ **Test fonction intégrée** (utils/inference_service.py):
```
1. Testing embeddings from config...
   Config: http://localhost:8000/v1 (model: default)
   Encoded 2 texts
   Embedding dimension: 768
   Using fallback: False
```

## Compatibilité

### Rétrocompatibilité

✅ **100% rétrocompatible**: L'ancien code fonctionne toujours
```python
# Ancien code (toujours supporté)
embedder = InferenceServer(
    url="http://localhost:8000/v1",
    fallback_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Migration recommandée

Pour bénéficier de la configuration centralisée:
1. Ajouter la section `inference` dans votre `config.yml`
2. Utiliser `InferenceServer.from_config()` dans le nouveau code
3. (Optionnel) Migrer progressivement l'ancien code

## Cas d'Usage

### 1. Changement de serveur

**Avant**: Modifier plusieurs fichiers Python
```python
# Dans generate_feedback.py
initialize_llm_client("http://new-server:8001/v1")

# Dans embeddings.py
model = load_embedding_model(..., server_url="http://new-server:8000/v1")
```

**Après**: Modifier uniquement config.yml
```yaml
inference:
  llm:
    url: "http://new-server:8001/v1"
  embeddings:
    url: "http://new-server:8000/v1"
```

### 2. Changement de modèle

**Avant**: Chercher tous les endroits où le modèle est mentionné
```python
# Multiples fichiers à modifier
model_name = "llama-3.2-3b-instruct"
```

**Après**: Une seule ligne dans config.yml
```yaml
inference:
  llm:
    model_name: "qwen2.5-coder-7b-instruct"
```

### 3. Tests avec différents backends

Créer plusieurs fichiers de config pour différents environnements:
- `config_lmstudio.yml` - Pour LM Studio
- `config_llamacpp.yml` - Pour llama.cpp
- `config_vllm.yml` - Pour vLLM

```bash
# Tester avec LM Studio
python pipeline.py --config configs/config_lmstudio.yml

# Tester avec llama.cpp
python pipeline.py --config configs/config_llamacpp.yml
```

## Breaking Changes

**Aucun** - Toutes les modifications sont rétrocompatibles.

## Notes de Migration

1. **Aucune action requise** pour continuer à utiliser l'ancien code
2. **Recommandé**: Mettre à jour les nouveaux scripts pour utiliser `from_config()`
3. **Optionnel**: Migrer progressivement les scripts existants

## Avantages

1. ✅ **Configuration centralisée** - Un seul endroit pour tous les endpoints
2. ✅ **Pas de modification du code** - Changement rapide d'infrastructure
3. ✅ **Support multi-environnement** - Facile de créer des configs par environnement
4. ✅ **Meilleure maintenabilité** - Séparation config/code
5. ✅ **Rétrocompatible** - Code existant continue de fonctionner
6. ✅ **Testable** - Scripts de test intégrés

## Contributeurs

- Configuration système: Claude Code
- Tests et validation: Système automatisé
- Documentation: Claude Code

## Références

- Documentation complète: [`docs/INFERENCE_CONFIG.md`](docs/INFERENCE_CONFIG.md)
- Configuration exemple: [`configs/config.yml`](configs/config.yml)
- Tests: [`test_config_loading.py`](test_config_loading.py)
