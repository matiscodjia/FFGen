# FFGen Triplet Viewer

A beautiful, minimal web interface for visualizing, editing, and analyzing training triplets for the FFGen embedding model training pipeline.

## Directory Structure

```
5_triplet_viewer_app/
├── backend/                    # Backend logic and data processing
│   ├── __init__.py
│   ├── data_loader.py         # Dataset loading and saving
│   ├── embeddings.py          # Embedding model loading and computation
│   ├── similarity.py          # Similarity calculations
│   └── rag.py                 # RAG testing functionality
├── frontend/                   # Frontend UI and visualizations
│   ├── __init__.py
│   ├── styles.py              # Custom CSS styling
│   ├── components.py          # UI component rendering functions
│   └── visualizations.py      # Plotly visualization functions
├── app.py                      # Main Streamlit application
└── README.md                   # This file
```

## Architecture

### Backend (`backend/`)

The backend handles all data processing, embedding computation, and RAG functionality:

- **data_loader.py**: Functions for loading JSONL datasets and saving edited versions
- **embeddings.py**: Model loading and embedding computation for single and multiple triplets
- **similarity.py**: Cosine similarity computation between positive and negative feedbacks
- **rag.py**: RAG testing utilities with optional ChromaDB integration

### Frontend (`frontend/`)

The frontend provides the user interface and visualizations:

- **styles.py**: Custom CSS styling with a clean, minimal design
- **components.py**: Reusable UI components (navigation, statistics, tabs, etc.)
- **visualizations.py**: 3D and 2D visualizations using Plotly and UMAP

### Main Application (`app.py`)

The main entry point that:
- Initializes Streamlit configuration
- Manages session state
- Orchestrates backend and frontend components
- Handles user interactions

## Features

### 1. Dataset Visualization
- Load JSONL datasets
- Browse triplets (anchor code, positive feedback, negative feedbacks)
- Navigate with Previous/Next buttons or jump to specific examples

### 2. Interactive Editing
- Edit code snippets
- Modify positive and negative feedbacks
- Add new negative feedbacks
- Export edited datasets

### 3. Similarity Analysis
- Compute cosine similarities between positive and negative feedbacks
- Visualize similarity distributions
- Quality assessment (identifies negatives that are too similar to positives)

### 4. 3D Embedding Visualization
- Visualize single triplet embeddings in 3D space using UMAP
- Interactive plots with connections from anchor to positive/negatives
- Color-coded by type (blue=anchor, green=positive, red=negative)

### 5. Global Dataset Visualization
- Visualize multiple triplets together in 3D UMAP space
- Customizable number of examples to sample
- Shows relationships across the entire dataset

### 6. RAG Testing
- Index corpus with embedding models
- Interactive search functionality
- Test different models (MiniLM, EmbeddingGemma, GraphCodeBERT, or custom)
- View top-k retrieval results with similarity scores

## Usage

### Running the Application

From the project root directory:

```bash
cd 5_triplet_viewer_app
streamlit run app.py
```

Or use the headless mode for server deployments:

```bash
STREAMLIT_SERVER_HEADLESS=true streamlit run app.py
```

### Workflow

1. **Upload Dataset**: Use the sidebar to upload a JSONL file containing triplets
2. **Load Model**: Select and load an embedding model for similarity and visualization
3. **Browse Triplets**: Navigate through examples using the controls
4. **Compute Similarities**: Click "Compute Similarities" to analyze negative quality
5. **Visualize**: Use the tabs to view statistics, visualizations, and RAG testing
6. **Edit**: Modify triplets directly in the text areas
7. **Export**: Download the edited dataset from the sidebar

## Dataset Format

The expected JSONL format for datasets:

```json
{
  "code_snippet": "def example():\n    return True",
  "conceptual_feedback": "This function implements...",
  "negative_feedbacks": [
    "This is an incorrect description...",
    "Another wrong feedback..."
  ],
  "code_id": "unique_id_123",
  "id": "doc_456"
}
```

## Dependencies

Core dependencies:
- `streamlit`: Web interface framework
- `sentence-transformers`: Embedding models
- `plotly`: Interactive visualizations
- `umap-learn`: Dimensionality reduction
- `scikit-learn`: Similarity computation
- `pandas`, `numpy`: Data processing

Optional dependencies:
- `chromadb`: For RAG testing (install with `uv pip install chromadb`)

## Configuration

### Model Options

Available embedding models (configurable in app.py):
- `sentence-transformers/all-MiniLM-L6-v2` (default, fast)
- `google/embeddinggemma-300m` (FFGen base model)
- `microsoft/graphcodebert-base` (code-specific)

### Customization

To add new models, edit `app.py`:

```python
model_options = [
    "your-model-name-here",
    "another/model",
]
```

To modify styling, edit `frontend/styles.py`.

## Development

### Adding New Features

1. **Backend functionality**: Add to appropriate module in `backend/`
2. **UI components**: Add rendering functions to `frontend/components.py`
3. **Visualizations**: Add plotting functions to `frontend/visualizations.py`
4. **Styling**: Modify `frontend/styles.py`

### Code Organization Principles

- **Separation of concerns**: Backend handles logic, frontend handles presentation
- **Reusable components**: Functions are modular and can be used independently
- **Session state management**: All state is managed through `st.session_state`
- **Caching**: Model loading is cached with `@st.cache_resource`

## Troubleshooting

### ChromaDB Errors

If you see ChromaDB-related errors, either:
1. Install ChromaDB: `uv pip install chromadb`
2. Don't use the RAG Testing tab (other features work without it)

### Memory Issues

For large datasets:
- Reduce the number of examples in Global Visualization
- Close unused browser tabs
- Restart the Streamlit server

### Slow Performance

- Use smaller embedding models (MiniLM is fastest)
- Reduce batch sizes in backend code if needed
- Compute similarities only when necessary

## License

Part of the FFGen project. See main project README for license information.
