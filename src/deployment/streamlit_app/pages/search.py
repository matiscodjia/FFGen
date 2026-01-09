import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from utils import load_css
# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Search & Logs Analysis", page_icon="🔍", layout="wide")

def load_css(file_name):
    try:
        css_file = Path(__file__).parent.parent / file_name
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError: st.error(f"CSS file not found: {file_name}")

load_css("assets/style_search.css")

# ==========================================
# UTILS
# ==========================================
def encode_text(text, model, tokenizer):
    if not text: return None
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy()

def calculate_cosine_distance(vec_a, vec_b):
    if vec_a is None or vec_b is None: return 1.0
    return 1.0 - min(max(np.dot(vec_a, vec_b), -1.0), 1.0)

# ==========================================
# UI
# ==========================================
if 'db_initialized' not in st.session_state or not st.session_state.db_initialized:
    st.warning("**System Not Initialized**")
    st.info("Please go to the **Home** page to load the model and dataset first.")
    st.stop()

st.title("Semantic Search & Debugger")
st.markdown("Debug vector search and verify semantic distance between query and reference code.")

col_main, col_sidebar = st.columns([3, 1])
with col_main:
    query_code = st.text_area("Input Code Snippet", height=200, placeholder="Paste code here...")

with col_sidebar:
    st.subheader("Search Params")
    k = st.slider("Retrieval Count (k)", 1, 20, 5)
    threshold_diff = st.slider("Divergence Threshold", 0.0, 1.0, 0.25, step=0.01)
    search_btn = st.button("🚀 Run Analysis", use_container_width=True)

st.markdown("### System Logs")
log_container = st.container()

if search_btn and query_code:
    start_time = time.time()
    with log_container:
        st.markdown('<div class="debug-terminal">', unsafe_allow_html=True)
        st.text(f"[*] Starting Analysis Pipeline...")
    
    try:
        with st.spinner("Encoding query..."):
            query_emb = encode_text(query_code, st.session_state.model, st.session_state.tokenizer)
        
        with log_container:
            st.text(f"[+] Query Encoded: Length {len(query_code)}, Norm {np.linalg.norm(query_emb):.4f}")

        with st.spinner(f"Querying ChromaDB (k={k})..."):
            results = st.session_state.collection.query(
                query_embeddings=[query_emb.tolist()], n_results=k, include=['documents', 'metadatas', 'distances']
            )

        st.divider()
        st.subheader("Analysis Results")
        valid_results = 0
        
        for i in range(len(results['documents'][0])):
            doc_id = results['ids'][0][i]
            feedback = results['documents'][0][i]
            ref_code = results['metadatas'][0][i].get('code', '')
            
            ref_code_emb = encode_text(ref_code, st.session_state.model, st.session_state.tokenizer)
            code_sem_dist = calculate_cosine_distance(query_emb, ref_code_emb)
            
            status_text = "REJECTED"
            status_class = "badge-error"
            if code_sem_dist < 0.05:
                status_class = "badge-success"; status_text = "EXACT MATCH"
            elif code_sem_dist < threshold_diff:
                status_class = "badge-warning"; status_text = "SEMANTIC MATCH"

            with log_container:
                st.text(f"    [Candidate #{i+1} - {doc_id}]")
                st.text(f"    > Code Dist : {code_sem_dist:.4f} | Status: {status_text}")

            if code_sem_dist < threshold_diff:
                valid_results += 1
                with st.expander(f"Result #{i+1} - {status_text} (Dist: {code_sem_dist:.3f})", expanded=True):
                    st.markdown(f"""
                    <div class="result-card">
                        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                            <span class="badge {status_class}">{status_text}</span>
                            <small style="color:#64748b">ID: {doc_id}</small>
                        </div>
                        <p style="font-size:1.05rem; line-height:1.6;">{feedback}</p>
                    </div>""", unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1: st.caption("Your Code"); st.code(query_code, language='c')
                    with c2: st.caption("DB Code"); st.code(ref_code, language='c')

        total_time = (time.time() - start_time) * 1000
        with log_container:
            st.text(f"[*] Finished in {total_time:.2f}ms. Valid: {valid_results}/{k}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        if valid_results == 0:
            st.warning("No results found within the semantic threshold.")

    except Exception as e:
        st.error(f"Analysis Failed: {str(e)}")