"""
Streamlit RAG Viewer avec Cache Intelligent
"""

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
import chromadb
from pathlib import Path
import json
import time
import logging
import sys
# Import des modules custom
from cache_manager import CacheManager
from deepseek_caller import DeepSeekCaller
from stats_logger import StatsLogger
from config import SIMILARITY_THRESHOLD
from utils import load_css
from huggingface_hub import login
import os

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="RAG Feedback System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS
# ==========================================
load_css("assets/style.css")

# ==========================================
# STATE MANAGEMENT
# ==========================================
if 'model_loaded' not in st.session_state: st.session_state.model_loaded = False
if 'dataset_loaded' not in st.session_state: st.session_state.dataset_loaded = False
if 'db_initialized' not in st.session_state: st.session_state.db_initialized = False
if 'cache_manager' not in st.session_state: st.session_state.cache_manager = None
if 'deepseek_caller' not in st.session_state: st.session_state.deepseek_caller = None
if 'stats_logger' not in st.session_state: st.session_state.stats_logger = StatsLogger()

# ==========================================
# HELPER FUNCTIONS
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("FFGen_System")
hf_token = os.environ.get("HF_TOKEN")

if hf_token:
    # Se connecte explicitement
    login(token=hf_token)
    print("Successfully connected to huggingface")
else:
    try:
        if "HF_TOKEN" in st.secrets:
            login(token=st.secrets["HF_TOKEN"])
            print("Connected via st.secrets")
        else:
            print("No HF key found")
    except FileNotFoundError:
        print("Local execution without secrets")
@st.cache_resource
def load_full_model(model_path: str):
    """Load standard HuggingFace model."""
    st.info(f"Loading model from: {model_path}")
    logger.info(f" Loading from: {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto"
        )
        logger.info(f"Modèle chargé avec succès !")
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Erreur de chargement: {e}")
        logger.error("Echec du chargement du modèle !")
        return None, None

def encode_text(text: str, model, tokenizer):
    """Encode text to embedding."""
    device = next(model.parameters()).device

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings[0].cpu().numpy().tolist()

@st.cache_data
def load_dataset_from_source(source: str, path: str):
    logger.info(f"Source séléctionnée {source}")
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
    db_path = Path("streamlit_rag_viewer/chroma_db_storage")
    db_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(db_path))

    try:
        if force_reindex:
            try: client.delete_collection("feedbacks")
            except: pass
            collection = client.create_collection(name="feedbacks")
        else:
            collection = client.get_collection(name="feedbacks")
    except:
        collection = client.create_collection(name="feedbacks")

    return client, collection

# ==========================================
# MAIN APP
# ==========================================

st.title("FFGEN")
st.markdown("### Submit code and get instant feedback")

# ==========================================
# SIDEBAR - CONFIGURATION
# ==========================================

with st.sidebar:
    st.header(" Configuration")

    # --- MODEL SELECTION ---
    st.subheader("Embedding Model")
    model_path = st.text_input(
        "Model Path (Local or HF)",
        value="matis35/gemmaembedding-fgdor",
        help="Path to embedding model"
    )

    # --- DATASET SELECTION ---
    st.subheader("Dataset")
    data_source = st.selectbox("Source", ["HuggingFace Hub", "Local JSONL"])
    dataset_path = st.text_input("Dataset Path", value="matis35/SYNT_V4")

    st.divider()

    # --- CACHE SETTINGS ---
    st.subheader("Cache Settings")

    # Permettre de modifier le threshold dynamiquement
    if 'custom_threshold' not in st.session_state:
        st.session_state.custom_threshold = SIMILARITY_THRESHOLD

    custom_threshold = st.slider(
        "Similarity Threshold",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.custom_threshold,
        step=0.05,
        help="Distance < threshold = HIT. Modifier cette valeur change le comportement du cache sans réindexer."
    )

    if custom_threshold != st.session_state.custom_threshold:
        st.session_state.custom_threshold = custom_threshold
        # Mettre à jour le threshold du cache manager existant si disponible
        if st.session_state.get('cache_manager'):
            st.session_state.cache_manager.threshold = custom_threshold
        st.info(f"Threshold updated to {custom_threshold:.2f}")

    st.caption(f"Current: Distance < {st.session_state.custom_threshold:.2f} = HIT")

    st.divider()

    force_reindex = st.checkbox("Force Re-index", value=False)

    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("Load & Index", use_container_width=True)
    with col2:
        use_cached_btn = st.button(" Use Cached", use_container_width=True)

    # --- LOAD CACHED DB ---
    if use_cached_btn:
        try:
            client, collection = initialize_chromadb(force_reindex=False)
            count = collection.count()
            if count > 0:
                st.session_state.client = client
                st.session_state.collection = collection
                st.session_state.db_initialized = True
                st.success(f"DB Loaded: {count} docs")
                logger.info(f"Base de données démarrée avec succès: {count} instances")
                if not st.session_state.model_loaded:
                    model, tokenizer = load_full_model(model_path)
                    if model:
                        st.session_state.model = model
                        st.session_state.tokenizer = tokenizer
                        st.session_state.model_loaded = True

                        # Initialiser cache manager avec threshold dynamique
                        encoder_fn = lambda text: encode_text(text, model, tokenizer)
                        st.session_state.cache_manager = CacheManager(
                            collection,
                            encoder_fn,
                            threshold=st.session_state.custom_threshold
                        )

                        # Initialiser DeepSeek caller
                        try:
                            st.session_state.deepseek_caller = DeepSeekCaller()
                            st.success(" DeepSeek API Ready")
                            logger.info("API prête")
                        except Exception as e:
                            st.warning(f" DeepSeek API unavailable: {e}")
                            logger.error(f"API non disponible: {e}")
            else:
                st.warning(" Empty DB. Please Load & Index first.")
        except Exception as e:
            st.error(f"Error: {e}")
            logger.error(f"Problème avec la base de données: {e}")

    # --- LOAD AND INDEX ---
    if load_btn:
        with st.spinner("Loading Model..."):
            model, tokenizer = load_full_model(model_path)
            if model:
                st.session_state.model = model
                st.session_state.tokenizer = tokenizer
                st.session_state.model_loaded = True
            else:
                st.stop()

        with st.spinner("Loading Dataset..."):
            logger.info("Chargement du dataset")
            try:
                data = load_dataset_from_source(data_source, dataset_path)
                st.session_state.dataset = data
                st.session_state.dataset_loaded = True
            except Exception as e:
                st.error(f"Dataset Error: {e}")
                logger.error("Problème de chargement du dataset")
                st.stop()

        if st.session_state.dataset_loaded:
            with st.spinner(f"Indexing {len(data)} items..."):
                client, collection = initialize_chromadb(force_reindex=force_reindex)

                batch_size = 64
                progress_bar = st.progress(0)

                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size]

                    feedbacks = [item.get("feedback", item.get("generated_feedback", "")) for item in batch]
                    codes = [item.get("code") for item in batch]

                    # IMPORTANT: Encode FEEDBACK for bi-encoder retrieval (code→feedback)
                    embeddings = [encode_text(fb, model, tokenizer) for fb in feedbacks]

                    # Store code as metadata for later comparison
                    metadatas = [{"code": c if c else ""} for c in codes]
                    ids = [f"id_{i+j}" for j in range(len(batch))]

                    collection.add(
                        embeddings=embeddings,
                        documents=feedbacks,
                        metadatas=metadatas,
                        ids=ids
                    )
                    progress_bar.progress(min(1.0, (i + batch_size) / len(data)))

                st.session_state.client = client
                st.session_state.collection = collection
                st.session_state.db_initialized = True

                # Initialiser cache manager avec threshold dynamique
                encoder_fn = lambda text: encode_text(text, model, tokenizer)
                st.session_state.cache_manager = CacheManager(
                    collection,
                    encoder_fn,
                    threshold=st.session_state.custom_threshold
                )

                # Initialiser DeepSeek
                try:
                    st.session_state.deepseek_caller = DeepSeekCaller()
                except:
                    pass

                st.success(" Indexing Complete!")

# ==========================================
# MAIN INTERFACE - QUERY
# ==========================================

if st.session_state.db_initialized and st.session_state.cache_manager:

    st.header(" Submit Your Code")

    # Formulaire enrichi
    with st.form("code_submission"):
        col1, col2 = st.columns([2, 1])

        with col1:
            code_input = st.text_area(
                "C Code",
                height=300,
                placeholder="Paste your C code here...",
                help="The code you want feedback on"
            )

        with col2:
            theme = st.text_input(
                "Exercise Theme",
                placeholder="e.g., Binary Search",
                help="What is this exercise about?"
            )

            difficulty = st.selectbox(
                "Difficulty Level",
                ["beginner", "intermediate", "advanced"]
            )

            error_category = st.text_input(
                "Error Category (optional)",
                placeholder="e.g., Off-by-one Error",
                help="If you know the type of error"
            )

        instructions = st.text_area(
            "Exercise Instructions (optional)",
            placeholder="Describe what the function should do...",
            help="Helps generate better feedback on cache miss"
        )

        col1, col2 = st.columns(2)
        with col1:
            test_scope = st.text_input(
                "Test Cases Scope (optional)",
                placeholder="e.g., Test with n=0, n=5, n=10",
                help="What tests should pass"
            )

        with col2:
            failed_tests = st.text_input(
                "Failed Tests (optional)",
                placeholder="e.g., Test n=0 returns wrong value",
                help="Which tests are failing"
            )

        submit_btn = st.form_submit_button(" Search Feedback", use_container_width=True)

    # TRAITEMENT DE LA REQUÊTE
    if submit_btn and code_input:
        start_time = time.time()

        # Contexte complet
        context = {
            "code": code_input,
            "theme": theme or "N/A",
            "difficulty": difficulty,
            "error_category": error_category or "Unknown",
            "instructions": instructions or "No instructions provided",
            "test_cases_scope": [test_scope] if test_scope else [],
            "failed_tests": [failed_tests] if failed_tests else []
        }

        # Query cache
        with st.spinner(" Searching cache..."):
            cache_result = st.session_state.cache_manager.query_cache(code_input, context)

        response_time = (time.time() - start_time) * 1000  # ms

        #  CACHE HIT ou PERFECT MATCH
        if cache_result['status'] in ['hit', 'perfect_match']:
            is_perfect = cache_result['status'] == 'perfect_match'

            st.markdown('<div class="hit-card">', unsafe_allow_html=True)

            if is_perfect:
                st.markdown("### PERFECT CODE MATCH - Exact Feedback Found")
                st.success("The submitted code is identical (similarity > 95%) to a code in the database. This feedback is 100% accurate.")
            else:
                st.markdown("### Cache HIT - Feedback from Database")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{cache_result['confidence']:.2f}")
            with col2:
                st.metric("Best Match Distance (code→feedback)", f"{cache_result['similarity_scores'][0]:.4f}")
            with col3:
                st.metric("Response Time", f"{response_time:.0f} ms")

            # Afficher code similarity si disponible
            if cache_result.get('code_similarity') is not None:
                st.metric("Code Similarity", f"{cache_result['code_similarity']:.4f}",
                         help="Similarity between your code and reference code (1.0 = identical)")

            if cache_result['needs_warning'] and not is_perfect:
                st.warning(" **Note:** Confidence is moderate. Review carefully.")

            # Afficher les résultats
            for result in cache_result['results']:
                # Calculer distance code_soumis ↔ code_référence
                code_ref = result['code']
                if code_ref and code_ref != 'N/A':
                    code_ref_embedding = encode_text(code_ref, st.session_state.model, st.session_state.tokenizer)
                    code_submitted_embedding = encode_text(code_input, st.session_state.model, st.session_state.tokenizer)

                    # Cosine similarity
                    import numpy as np
                    similarity = np.dot(code_ref_embedding, code_submitted_embedding)
                    code_distance = 1 - similarity
                else:
                    code_distance = None

                with st.expander(f" Match #{result['rank']} (code→feedback distance: {result['distance']:.4f})"):
                    # Métriques côte à côte
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Code → Feedback", f"{result['distance']:.4f}", help="Distance entre votre code et ce feedback (apprentissage bi-encoder)")
                    with col2:
                        if code_distance is not None:
                            st.metric("Code → Code Ref", f"{code_distance:.4f}", help="Distance entre votre code et le code de référence pour ce feedback")

                    st.markdown("**Feedback:**")
                    st.write(result['feedback'])

                    st.markdown("**Reference Code (this feedback was given for):**")
                    st.code(result['code'], language='c')

            st.markdown('</div>', unsafe_allow_html=True)

            # Log stats
            st.session_state.stats_logger.log_query({
                "query_id": cache_result['query_id'],
                "status": "hit",
                "similarity_score": cache_result['similarity_scores'][0],
                "confidence": cache_result['confidence'],
                "response_time_ms": response_time,
                "theme": theme,
                "error_category": error_category,
                "difficulty": difficulty,
                "deepseek_tokens": 0,
                "cache_size": st.session_state.collection.count()
            })

        #  CACHE MISS
        elif cache_result['status'] == 'miss':
            st.markdown('<div class="miss-card">', unsafe_allow_html=True)
            st.markdown("###  Cache MISS - Generating New Feedback")

            st.info(f" Closest match distance: {cache_result.get('closest_distance', 1.0):.4f} (threshold: {st.session_state.custom_threshold:.2f})")

            # Afficher les codes les plus proches même en cas de miss
            if cache_result['results']:
                st.markdown("#### Closest matches found (but below threshold):")
                for result in cache_result['results']:
                    # Calculer distance code_soumis ↔ code_référence
                    code_ref = result['code']
                    if code_ref and code_ref != 'N/A':
                        code_ref_embedding = encode_text(code_ref, st.session_state.model, st.session_state.tokenizer)
                        code_submitted_embedding = encode_text(code_input, st.session_state.model, st.session_state.tokenizer)

                        import numpy as np
                        similarity = np.dot(code_ref_embedding, code_submitted_embedding)
                        code_distance = 1 - similarity
                    else:
                        code_distance = None

                    with st.expander(f"Match #{result['rank']} (code→feedback: {result['distance']:.4f})"):
                        # Métriques côte à côte
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Code → Feedback", f"{result['distance']:.4f}", help="Distance bi-encoder (apprentissage)")
                        with col2:
                            if code_distance is not None:
                                st.metric("Code → Code Ref", f"{code_distance:.4f}", help="Distance code soumis vs code de référence")

                        st.markdown("**Feedback (given for reference code):**")
                        st.write(result['feedback'])

                        st.markdown("**Reference Code:**")
                        st.code(result['code'], language='c')

                st.divider()

            # Appeler DeepSeek
            if st.session_state.deepseek_caller:
                with st.spinner(" Generating feedback with DeepSeek..."):
                    deepseek_result = st.session_state.deepseek_caller.generate_feedback(context)

                if deepseek_result.get('feedback'):
                    feedback = deepseek_result['feedback']
                    tokens_used = deepseek_result['tokens_total']

                    st.success(" Feedback Generated!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tokens Used", tokens_used)
                    with col2:
                        st.metric("Generation Time", f"{deepseek_result['generation_time_ms']:.0f} ms")
                    with col3:
                        st.metric("Total Time", f"{response_time + deepseek_result['generation_time_ms']:.0f} ms")

                    st.markdown("**Generated Feedback:**")
                    st.write(feedback)

                    # Distillation : Ajouter au cache
                    with st.spinner(" Adding to cache (distillation)..."):
                        # Encoder le feedback
                        feedback_embedding = encode_text(feedback, st.session_state.model, st.session_state.tokenizer)

                        success = st.session_state.cache_manager.add_to_cache(
                            code=code_input,
                            feedback=feedback,
                            metadata=context,
                            embedding=feedback_embedding
                        )

                        if success:
                            st.success(" Feedback added to cache for future queries!")

                    # Log cache miss (format dataset)
                    miss_data = {
                        **context,
                        "tags": [tag.strip() for tag in error_category.split(',') if tag.strip()] if error_category else [],
                        "feedback": feedback,
                        "query_id": cache_result['query_id'],
                        "tokens_used": tokens_used
                    }
                    st.session_state.stats_logger.log_cache_miss(miss_data)

                    # Log stats
                    st.session_state.stats_logger.log_query({
                        "query_id": cache_result['query_id'],
                        "status": "miss",
                        "similarity_score": cache_result.get('closest_distance', 1.0),
                        "confidence": 1.0,  # LLM généré = haute confiance
                        "response_time_ms": response_time + deepseek_result['generation_time_ms'],
                        "theme": theme,
                        "error_category": error_category,
                        "difficulty": difficulty,
                        "deepseek_tokens": tokens_used,
                        "cache_size": st.session_state.collection.count()
                    })
                else:
                    st.error(f" Error: {deepseek_result.get('error', 'Unknown error')}")
            else:
                st.error(" DeepSeek API not available. Cannot generate feedback.")

            st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info(" Please configure and load the model + dataset from the sidebar first.")

    st.markdown("""
    ### How to use:
    1. **Load Model & Dataset** (or use cached DB)
    2. **Fill in the form** with your code and its context
    3. **Submit** to get feedback
    4. **Check the Stats page** to see cache performance

    ### Cache System:
    -  **Hit**: Similar code found in database (instant response) Or Relevant feedabck code found in db with code feedback embedder 
    -  **Miss**: No match found, generates new feedback (slower, uses API tokens)
    -  **Distillation**: New feedbacks are automatically added to the cache
    """)
