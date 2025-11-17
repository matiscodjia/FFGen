# Configuration des Services d'Inférence

Ce document explique comment configurer les endpoints et modèles d'inférence depuis le fichier de configuration.

## Vue d'ensemble

Le système supporte maintenant la configuration centralisée des services d'inférence (LLM et embeddings) via le fichier `configs/config.yml`. Cela permet de:
- Modifier les URLs des serveurs d'inférence sans toucher au code
- Changer les noms de modèles facilement
- Configurer les timeouts et fallbacks
- Basculer entre différents serveurs (llama.cpp, LM Studio, etc.)

## Configuration

### Structure dans `configs/config.yml`

```yaml
# --- Services d'Inférence ---
inference:
  # LLM pour la génération de texte
  llm:
    url: "http://localhost:1234/v1"
    model_name: "llama-3.2-3b-instruct"
    timeout: 5.0
    fallback_model: null  # Optional local model fallback

  # Embeddings pour le negative mining
  embeddings:
    url: "http://localhost:8000/v1"
    model_name: "default"
    timeout: 5.0
    fallback_model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Paramètres

#### Section `llm`
- **`url`**: Endpoint du serveur OpenAI-compatible (llama.cpp, LM Studio, vLLM, etc.)
- **`model_name`**: Nom du modèle sur le serveur
- **`timeout`**: Timeout en secondes pour la vérification de connexion
- **`fallback_model`**: (Optionnel) Modèle local à utiliser si le serveur est indisponible

#### Section `embeddings`
- **`url`**: Endpoint du serveur d'embeddings
- **`model_name`**: Nom du modèle d'embeddings sur le serveur
- **`timeout`**: Timeout en secondes pour la vérification de connexion
- **`fallback_model`**: Modèle SentenceTransformers local à utiliser en fallback

## Utilisation dans le Code

### Chargement depuis la configuration

```python
from utils.inference_service import InferenceServer

# Charger le service LLM
llm = InferenceServer.from_config(
    config_path="./configs/config.yml",
    service_type="llm"
)

# Charger le service d'embeddings
embedder = InferenceServer.from_config(
    config_path="./configs/config.yml",
    service_type="embeddings"
)

# Utilisation
messages = [{"role": "user", "content": "Hello"}]
response = await llm.chat(messages)

texts = ["text1", "text2"]
embeddings = await embedder.encode(texts)
```

### Chargement manuel (ancien style, toujours supporté)

```python
# Si vous préférez configurer manuellement
llm = InferenceServer(
    url="http://localhost:1234/v1",
    model_name="llama-3.2-3b-instruct"
)
```

## Exemples de Configuration

### Configuration pour llama.cpp

```yaml
inference:
  llm:
    url: "http://localhost:8080/v1"
    model_name: "llama-3.2-3b-instruct"
    timeout: 10.0
    fallback_model: null

  embeddings:
    url: "http://localhost:8080/v1"
    model_name: "nomic-embed-text"
    timeout: 10.0
    fallback_model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Configuration pour LM Studio

```yaml
inference:
  llm:
    url: "http://localhost:1234/v1"
    model_name: "lmstudio-community/qwen2.5-coder-7b-instruct"
    timeout: 5.0
    fallback_model: null

  embeddings:
    url: "http://localhost:1235/v1"
    model_name: "nomic-ai/nomic-embed-text-v1.5-GGUF"
    timeout: 5.0
    fallback_model: "sentence-transformers/all-MiniLM-L6-v2"
```

### Configuration pour serveurs distants

```yaml
inference:
  llm:
    url: "https://api.example.com/v1"
    model_name: "llama-3.2-3b-instruct"
    timeout: 30.0
    fallback_model: null

  embeddings:
    url: "https://api.example.com/v1"
    model_name: "text-embedding-3-small"
    timeout: 30.0
    fallback_model: "sentence-transformers/all-MiniLM-L6-v2"
```

## Mécanisme de Fallback

Le système intègre un mécanisme de fallback automatique:

1. **Tentative de connexion au serveur**: Le système essaie d'abord de se connecter à l'URL configurée
2. **Fallback automatique**: Si le serveur est indisponible et qu'un `fallback_model` est configuré, le système bascule automatiquement sur le modèle local
3. **Messages informatifs**: Le système affiche des messages clairs indiquant quel backend est utilisé

Exemple de sortie:
```
[InferenceServer] ✓ Connected to server at http://localhost:8000/v1
[InferenceServer] !  Server unavailable at http://localhost:8001/v1
[InferenceServer]  Falling back to local model: sentence-transformers/all-MiniLM-L6-v2
[InferenceServer] ✓ Loaded local model: sentence-transformers/all-MiniLM-L6-v2
```

## Tests

### Tester la configuration

```bash
# Test complet avec la configuration
python utils/inference_service.py

# Test spécifique
python test_config_loading.py

# Test du fallback
python test_fallback.py
```

## Composants Modifiés

Les composants suivants utilisent maintenant la configuration:
- ✅ `utils/inference_service.py` - Service principal
- ✅ `2_data_generation/generate_feedback.py` - Génération de feedback LLM
- ✅ `5_triplet_viewer_app/backend/embeddings.py` - Module d'embeddings

## Migration

Si vous utilisez l'ancien code, voici comment migrer:

### Avant
```python
llm = InferenceServer(
    url="http://localhost:1234/v1",
    model_name="llama-3.2-3b-instruct"
)
```

### Après
```python
# Option 1: Depuis la config (recommandé)
llm = InferenceServer.from_config(
    config_path="./configs/config.yml",
    service_type="llm"
)

# Option 2: Garder l'ancien code (toujours compatible)
llm = InferenceServer(
    url="http://localhost:1234/v1",
    model_name="llama-3.2-3b-instruct"
)
```

## Dépannage

### Le serveur n'est pas détecté
- Vérifiez que le serveur est bien démarré
- Vérifiez l'URL et le port dans la configuration
- Augmentez le `timeout` si le serveur est lent à démarrer

### Le fallback ne fonctionne pas
- Vérifiez que `fallback_model` est configuré
- Vérifiez que le modèle local est bien installé
- Pour les embeddings: installez `sentence-transformers`

### Erreur "Config file missing 'inference' section"
- Assurez-vous que votre fichier `config.yml` contient bien la section `inference`
- Vérifiez l'indentation YAML (2 espaces par niveau)
