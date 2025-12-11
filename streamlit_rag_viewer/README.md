# Streamlit RAG Viewer

Simple, colorful, and professional interface for semantic code feedback search.

## Features

- ✅ **Pure Streamlit** - No separate backend
- 🎨 **Colorful & Adaptive** - Modern gradient UI
- 🤖 **Base Model + PEFT** - Load 400M model + your adapter
- 🔍 **Semantic Search** - ChromaDB vector search
- 📦 **Flexible Data** - HuggingFace Hub or local JSONL

## Quick Start

```bash
cd streamlit_rag_viewer

# Install
pip install -r requirements.txt

# Run
streamlit run app.py
```

Opens at **http://localhost:8501**

## Usage

### 1. Configure (Sidebar)

- **Dataset**: Choose HuggingFace Hub or local JSONL
- **Base Model**: `Salesforce/SFR-Embedding-Code-400M_R`
- **PEFT Adapter**: Your adapter from Hub (e.g., `matis35/my-adapter`)

### 2. Load & Index

Click **🚀 Load & Index** - it will:
1. Load base model (400M)
2. Load your PEFT adapter
3. Combine them
4. Load dataset
5. Index in ChromaDB

### 3. Search

- Paste code
- Choose k results
- Click **🔍 Search**

## Architecture

**Single Streamlit app** with:
- Model: Base (400M) + PEFT adapter loaded together
- Storage: ChromaDB (persistent in `.chroma_cache/`)
- UI: Gradient colors, adaptive layout

## Why No Backend?

Everything runs in Streamlit:
- Model loading
- Encoding
- ChromaDB
- Search

**No FastAPI needed** - Streamlit handles it all!

## Example Adapter

Your PEFT adapter structure on Hub:

```
matis35/my-code-adapter/
├── adapter_config.json
├── adapter_model.bin
└── README.md
```

The app will:
1. Load base: `AutoModel.from_pretrained("Salesforce/SFR-Embedding-Code-400M_R")`
2. Load adapter: `PeftModel.from_pretrained(base_model, "matis35/my-code-adapter")`
3. Use combined model for encoding

## Enjoy! 🚀
