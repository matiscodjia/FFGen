"""
Streamlit RAG Viewer - Simple & Professional
Load base model + PEFT adapter for code feedback semantic search
"""

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from datasets import load_dataset
import chromadb
from pathlib import Path
import json

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="RAG Feedback Viewer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS - COLORFUL & ADAPTIVE
# ==========================================
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary: #3b82f6;
        --secondary: #8b5cf6;
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
    }

    /* Better spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Colorful headers */
    h1 {
        color: #3b82f6 !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h2 {
        color: #1e40af !important;
        border-bottom: 3px solid #3b82f6;
        padding-bottom: 0.5rem;
        margin-top: 2rem !important;
    }

    h3 {
        color: #6366f1 !important;
    }

    /* Status indicators */
    .status-success {
        background: #d1fae5;
        color: #065f46;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #10b981;
        margin: 1rem 0;
    }

    .status-error {
        background: #fee2e2;
        color: #991b1b;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ef4444;
        margin: 1rem 0;
    }

    .status-info {
        background: #dbeafe;
        color: #1e40af;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }

    /* Result cards */
    .result-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
    }

    .result-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .similarity-badge {
        background: rgba(255, 255, 255, 0.2);
        padding: 0.5rem 1rem;
        border-radius: 2rem;
        font-size: 0.9rem;
    }

    .code-block {
        background: rgba(0, 0, 0, 0.3);
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: 'Monaco', 'Menlo', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
        margin-top: 0.5rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        transition: transform 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
    }

    /* Metrics */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# STATE MANAGEMENT
# ==========================================
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False

# ==========================================
# HELPER FUNCTIONS
# ==========================================

@st.cache_resource
def load_model_with_adapter(base_model_name: str, adapter_name: str):
    """Load base model + PEFT adapter."""
    st.info(f"🔄 Loading base model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    base_model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True)

    st.info(f"🔄 Loading PEFT adapter: {adapter_name}")
    model = PeftModel.from_pretrained(base_model, adapter_name)
    model.eval()

    return model, tokenizer

def encode_text(text: str, model, tokenizer):
    """Encode text to embedding."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0].cpu().numpy().tolist()

@st.cache_data
def load_dataset_from_source(source: str, path: str):
    """Load dataset from HuggingFace or JSONL."""
    if source == "HuggingFace Hub":
        dataset = load_dataset(path)
        data = []
        for split in dataset.keys():
            data.extend(dataset[split].to_list())
        return data
    else:
        data = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return data

def initialize_chromadb(force_reindex=False):
    """Initialize ChromaDB - reuses existing collection if available."""
    db_path = Path("streamlit_rag_viewer/.chroma_cache")
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))

    # Try to get existing collection
    try:
        if force_reindex:
            client.delete_collection("feedbacks")
            collection = client.create_collection(name="feedbacks")
        else:
            # Try to get existing collection
            collection = client.get_collection(name="feedbacks")
    except:
        # Collection doesn't exist, create it
        collection = client.create_collection(name="feedbacks")

    return client, collection

# ==========================================
# MAIN APP
# ==========================================

st.title("🔍 RAG Feedback Viewer")
st.markdown("### Semantic search for code feedback using fine-tuned embeddings")

# ==========================================
# SIDEBAR - CONFIGURATION
# ==========================================

with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("📦 Dataset")
    data_source = st.selectbox(
        "Source",
        ["HuggingFace Hub", "Local JSONL"]
    )

    dataset_path = st.text_input(
        "Dataset Path",
        value="matis35/RAFT_CLEAN_V1",
        placeholder="matis35/RAFT_CLEAN_V1"
    )

    st.subheader("🤖 Model")
    base_model = st.text_input(
        "Base Model",
        value="Salesforce/SFR-Embedding-Code-400M_R",
        help="Base embedding model (400M)"
    )

    adapter_model = st.text_input(
        "PEFT Adapter",
        value="",
        placeholder="matis35/my-code-adapter",
        help="Your PEFT adapter from HuggingFace Hub"
    )

    st.divider()

    force_reindex = st.checkbox(
        "🔄 Force Re-index",
        value=False,
        help="Delete existing index and recreate from scratch"
    )

    col1, col2 = st.columns(2)

    with col1:
        load_and_index = st.button("🚀 Load & Index", use_container_width=True)

    with col2:
        load_existing = st.button("📂 Use Existing", use_container_width=True, help="Use cached ChromaDB index")

    if load_existing:
        try:
            client, collection = initialize_chromadb(force_reindex=False)
            existing_count = collection.count()

            if existing_count > 0:
                st.session_state.client = client
                st.session_state.collection = collection
                st.session_state.db_initialized = True
                st.success(f"✅ Loaded existing index: {existing_count:,} entries")

                # Also load model if not loaded
                if not st.session_state.model_loaded and adapter_model:
                    with st.spinner("Loading model..."):
                        model, tokenizer = load_model_with_adapter(base_model, adapter_model)
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model_loaded = True
                        st.success("✅ Model loaded!")
            else:
                st.warning("⚠️ No existing index found. Use 'Load & Index' first.")
        except Exception as e:
            st.error(f"❌ Error: {e}")

    if load_and_index:
        if not adapter_model:
            st.error("⚠️ Please provide a PEFT adapter path!")
        else:
            with st.spinner("Loading model..."):
                try:
                    model, tokenizer = load_model_with_adapter(base_model, adapter_model)
                    st.session_state.model = model
                    st.session_state.tokenizer = tokenizer
                    st.session_state.model_loaded = True
                    st.success("✅ Model loaded!")
                except Exception as e:
                    st.error(f"❌ Error loading model: {e}")

            if st.session_state.model_loaded:
                with st.spinner("Loading dataset..."):
                    try:
                        data = load_dataset_from_source(data_source, dataset_path)
                        st.session_state.dataset = data
                        st.session_state.dataset_loaded = True
                        st.success(f"✅ Dataset loaded: {len(data):,} entries")
                    except Exception as e:
                        st.error(f"❌ Error loading dataset: {e}")

            if st.session_state.dataset_loaded:
                try:
                    client, collection = initialize_chromadb(force_reindex=force_reindex)
                    st.session_state.client = client
                    st.session_state.collection = collection

                    # Check if collection already has data
                    existing_count = collection.count()

                    if existing_count > 0 and not force_reindex:
                        st.info(f"📦 Found existing index with {existing_count:,} entries. Using cached data.")
                        st.info("💡 Check 'Force Re-index' to recreate the index.")
                        st.session_state.db_initialized = True
                        st.success("✅ Using cached index!")
                    else:
                        with st.spinner("Indexing in ChromaDB..."):
                            progress_bar = st.progress(0)

                            batch_size = 100
                            for i in range(0, len(data), batch_size):
                                batch = data[i:i+batch_size]

                                feedbacks = [item.get("feedback", item.get("generated_feedback", "")) for item in batch]
                                codes = [item.get("code", item.get("code_snippet", "")) for item in batch]

                                embeddings = [encode_text(fb, model, tokenizer) for fb in feedbacks]

                                collection.add(
                                    embeddings=embeddings,
                                    documents=feedbacks,
                                    metadatas=[{"code": code} for code in codes],
                                    ids=[f"doc_{i+j}" for j in range(len(batch))]
                                )

                                progress_bar.progress(min(i+batch_size, len(data)) / len(data))

                            st.session_state.db_initialized = True
                            st.success("✅ Indexing complete!")
                            st.balloons()
                except Exception as e:
                    st.error(f"❌ Error indexing: {e}")

    st.divider()

    # Status
    st.subheader("📊 Status")
    if st.session_state.model_loaded:
        st.markdown('<div class="status-success">✅ Model loaded</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">❌ Model not loaded</div>', unsafe_allow_html=True)

    if st.session_state.dataset_loaded:
        st.markdown(f'<div class="status-success">✅ Dataset loaded<br/>{len(st.session_state.dataset):,} entries</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">❌ Dataset not loaded</div>', unsafe_allow_html=True)

    if st.session_state.db_initialized:
        st.markdown('<div class="status-success">✅ DB indexed</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-error">❌ DB not indexed</div>', unsafe_allow_html=True)

# ==========================================
# MAIN AREA - SEARCH
# ==========================================

st.header("🔎 Search Feedbacks")

if not st.session_state.db_initialized:
    st.markdown('<div class="status-info">👈 Configure and load a dataset from the sidebar to start searching</div>', unsafe_allow_html=True)
else:
    code_input = st.text_area(
        "Code Snippet",
        height=200,
        value="""int my_compute_factorial_rec(int nb)
{
    if (nb == 0) {
        return (1);
    }
    if (nb < 0 || nb > 12) {
        return (0);
    } else {
        return nb * my_compute_factorial_rec(nb - 1);
    }
}""",
        placeholder="Paste your code here..."
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        k = st.slider("Number of results (k)", min_value=1, max_value=20, value=5)

    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        search_btn = st.button("🔍 Search", use_container_width=True)

    if search_btn and code_input.strip():
        with st.spinner("Searching..."):
            try:
                # Encode query
                query_embedding = encode_text(
                    code_input,
                    st.session_state.model,
                    st.session_state.tokenizer
                )

                # Search
                results = st.session_state.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=k
                )

                st.subheader(f"📋 Top {k} Results")

                for i in range(len(results['documents'][0])):
                    feedback = results['documents'][0][i]
                    code = results['metadatas'][0][i]['code']
                    distance = results['distances'][0][i]
                    similarity = 1 - distance

                    st.markdown(f"""
                    <div class="result-card">
                        <div class="result-header">
                            <span>🎯 Result #{i+1}</span>
                            <span class="similarity-badge">Similarity: {similarity:.3f}</span>
                        </div>
                        <div style="margin-bottom: 1rem;">
                            <strong>💬 Feedback:</strong><br/>
                            {feedback}
                        </div>
                        <div>
                            <strong>💻 Original Code:</strong>
                            <div class="code-block">{code}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Search error: {e}")
