# MIPS-based RAG Implementation Summary

## What Was Done

### 1. Data Extraction and Merging
- Created `extract_and_merge_codes.py` to extract all C code from the `codes/` directory
- Merged with existing `processed_collection.parquet`
- **Result**: 14,578 unique code snippets in `data/collections.parquet`

### 2. Parquet to JSONL Conversion
- Created simple utility `utils/parquet_to_jsonl.py`
- Converted `collections.parquet` → `collections.jsonl`
- **Result**: 14,578 lines in `data/collections.jsonl` ready for pipeline stages

### 3. MIPS-based RAG System
Rebuilt the RAG (Retrieval Augmented Generation) system with:

#### Backend (`5_triplet_viewer_app/backend/rag.py`)
- **MIPS Implementation**: Maximum Inner Product Search
- **Query**: Code snippets
- **Response**: Relevant feedback texts
- **Caching**: Optional ChromaDB support for embedding caching
- **Key Features**:
  - Indexes feedback corpus from dataset
  - Computes normalized embeddings
  - Performs efficient similarity search using numpy dot product
  - Returns ranked results with similarity scores

#### Frontend (`5_triplet_viewer_app/frontend/components.py`)
- **Updated RAG Testing Tab** with:
  - Clear MIPS interface: "Code → Feedback"
  - Top-k slider (1-20 results)
  - Similarity score visualization with:
    - Progress bars
    - Color-coded badges (🟢 high, 🟡 medium, 🔴 low similarity)
  - Display of retrieved feedbacks with associated code
  - ChromaDB status indicator

## How It Works

1. **Indexing**:
   ```python
   rag_tester.index_corpus(dataset, model, use_chromadb=True)
   ```
   - Extracts all feedbacks from dataset
   - Computes embeddings for each feedback
   - Optionally caches in ChromaDB

2. **Searching**:
   ```python
   results = rag_tester.search(code_query, top_k=5)
   # Returns: [(rank, similarity, feedback_doc), ...]
   ```
   - Encodes code query into embedding
   - Computes dot product with all feedback embeddings
   - Returns top-k most similar feedbacks with scores

## Files Modified

1. `extract_and_merge_codes.py` - NEW
2. `utils/parquet_to_jsonl.py` - NEW
3. `5_triplet_viewer_app/backend/rag.py` - UPDATED
4. `5_triplet_viewer_app/frontend/components.py` - UPDATED

## Data Files Created

1. `data/collections.parquet` - 14,578 records (merged dataset)
2. `data/collections.jsonl` - 14,578 lines (JSONL format)

## Usage

### Run Streamlit App:
```bash
STREAMLIT_SERVER_HEADLESS=true .venv/bin/streamlit run 5_triplet_viewer_app/app.py
```

### In the app:
1. Upload a JSONL dataset with `code_snippet` and `conceptual_feedback` fields
2. Go to "RAG Testing" tab
3. Select embedding model
4. Click "Index Feedbacks"
5. Enter code snippet in the query box
6. Adjust top-k slider
7. Click "Search Feedbacks"
8. View ranked results with similarity scores

## Key Improvements

- ✅ MIPS-based retrieval (code → feedback)
- ✅ Similarity scores displayed for each result
- ✅ Top-k adjustable cursor
- ✅ ChromaDB optional caching
- ✅ Clean, intuitive interface
- ✅ Color-coded similarity indicators
- ✅ Progress bars for visual feedback

## Bug Fix: Batch Size Error

### Issue
Error during indexing: "Batch size of 6102 is greater than max batch size of 5461"

### Solution
1. **Text Truncation**: Truncate feedbacks and queries to 2000 characters (~500 tokens)
2. **Batch Processing**: Process embeddings in batches of 64 items
3. **Progress Bar**: Added progress indicator for long indexing operations

### Changes in `backend/rag.py`:
- Truncate feedback texts before encoding
- Set `batch_size=64` for encoding
- Truncate query code before searching
- Added `show_progress_bar=True` for user feedback

This ensures the system works with large datasets (6000+ documents) without hitting model token limits.
