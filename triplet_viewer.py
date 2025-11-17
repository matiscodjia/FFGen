#!/usr/bin/env python3
"""
FFGen Triplet Viewer - Streamlit App
Beautiful, minimal interface for visualizing and editing training triplets
"""

import streamlit as st
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import umap
from typing import List, Dict

# Optional ChromaDB for RAG testing
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="FFGen Triplet Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Ultra minimal and clean
st.markdown("""
<style>
    /* Clean typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', sans-serif;
    }

    /* Main header */
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #1a1a1a;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }

    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Clean buttons */
    .stButton > button {
        background: white;
        color: #1a1a1a;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    .stButton > button:hover {
        border-color: #3b82f6;
        box-shadow: 0 4px 12px rgba(59,130,246,0.15);
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Primary button */
    button[kind="primary"] {
        background: #3b82f6 !important;
        color: white !important;
        border: none !important;
    }

    button[kind="primary"]:hover {
        background: #2563eb !important;
    }

    /* Code blocks */
    .code-block {
        background-color: #f9fafb;
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        font-family: 'Courier New', monospace;
        font-size: 0.875rem;
        line-height: 1.6;
    }

    /* Content cards */
    .positive-block {
        background: linear-gradient(to bottom right, #ecfdf5, #f0fdf4);
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #d1fae5;
        border-left: 3px solid #10b981;
    }

    .negative-block {
        background: linear-gradient(to bottom right, #fef2f2, #fef2f2);
        padding: 1.25rem;
        border-radius: 8px;
        border: 1px solid #fecaca;
        border-left: 3px solid #ef4444;
        margin-bottom: 0.75rem;
    }

    /* Text areas */
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
    }

    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.1);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #fafafa;
        border-right: 1px solid #e5e7eb;
    }

    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 8px 16px;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        border-color: #3b82f6;
    }

    /* Clean similarity bars */
    .similarity-bar {
        height: 6px;
        border-radius: 3px;
        transition: width 0.3s ease;
    }

    .sim-high { background: #ef4444; }
    .sim-medium { background: #f59e0b; }
    .sim-low { background: #10b981; }

    /* Info boxes */
    .stAlert {
        border-radius: 8px;
        border-left: 3px solid;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'similarities' not in st.session_state:
    st.session_state.similarities = {}
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}
if 'model' not in st.session_state:
    st.session_state.model = None
if 'edited_items' not in st.session_state:
    st.session_state.edited_items = set()
if 'global_embeddings' not in st.session_state:
    st.session_state.global_embeddings = None

@st.cache_resource
def load_embedding_model(model_name: str):
    """Load sentence transformer model (cached)"""
    with st.spinner(f'Loading model: {model_name}...'):
        return SentenceTransformer(model_name)

def compute_similarities(positive: str, negatives: List[str], model) -> List[float]:
    """Compute cosine similarities between positive and negatives"""
    if not negatives or not positive:
        return []
    try:
        positive_emb = model.encode([positive])
        negative_embs = model.encode(negatives)
        similarities = cosine_similarity(positive_emb, negative_embs)[0]
        return similarities.tolist()
    except Exception as e:
        st.error(f"Error computing similarities: {e}")
        return []

def compute_embeddings_for_visualization(item: Dict, model):
    """Compute embeddings for anchor, positive, and negatives"""
    code = item.get('code_snippet', '')
    positive = item.get('conceptual_feedback', item.get('refined_feedback', ''))
    negatives = item.get('negative_feedbacks', [])

    if not negatives and 'hard_negative_feedback' in item:
        negatives = [item['hard_negative_feedback']]

    texts = [code, positive] + negatives
    embeddings = model.encode(texts)

    return {
        'embeddings': embeddings,
        'labels': ['Anchor (Code)', 'Positive'] + [f'Negative {i+1}' for i in range(len(negatives))],
        'types': ['anchor', 'positive'] + ['negative'] * len(negatives)
    }

def create_3d_visualization(embeddings_data: Dict):
    """Create 3D interactive visualization with connections using UMAP"""
    embeddings = embeddings_data['embeddings']
    labels = embeddings_data['labels']
    types = embeddings_data['types']

    # Reduce to 3D using UMAP
    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=min(15, len(embeddings)-1))
    coords_3d = reducer.fit_transform(embeddings)

    # Color mapping
    color_map = {
        'anchor': '#3b82f6',  # Blue
        'positive': '#10b981',  # Green
        'negative': '#ef4444'  # Red
    }

    colors = [color_map[t] for t in types]

    # Create 3D scatter plot
    fig = go.Figure()

    # Add points
    fig.add_trace(go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=12,
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=labels,
        textposition='top center',
        textfont=dict(size=10, color='#1a1a1a'),
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        name='Embeddings'
    ))

    # Add connections from anchor to positive and negatives
    anchor_coords = coords_3d[0]
    for i in range(1, len(coords_3d)):
        target_coords = coords_3d[i]

        # Line color based on type
        line_color = '#10b981' if types[i] == 'positive' else '#ef4444'
        line_width = 3 if types[i] == 'positive' else 1

        fig.add_trace(go.Scatter3d(
            x=[anchor_coords[0], target_coords[0]],
            y=[anchor_coords[1], target_coords[1]],
            z=[anchor_coords[2], target_coords[2]],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            opacity=0.6,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb', showbackground=False),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb', showbackground=False),
            zaxis=dict(showgrid=True, gridcolor='#e5e7eb', showbackground=False),
            bgcolor='white'
        ),
        showlegend=False,
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='white',
        font=dict(family='Inter', size=12)
    )

    return fig

def create_2d_visualization(embeddings_data: Dict):
    """Create 2D visualization for download using UMAP"""
    embeddings = embeddings_data['embeddings']
    labels = embeddings_data['labels']
    types = embeddings_data['types']

    # Reduce to 2D using UMAP
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings)-1))
    coords_2d = reducer.fit_transform(embeddings)

    # Create DataFrame for Plotly Express
    df = pd.DataFrame({
        'X': coords_2d[:, 0],
        'Y': coords_2d[:, 1],
        'Label': labels,
        'Type': types
    })

    color_map = {
        'anchor': '#3b82f6',
        'positive': '#10b981',
        'negative': '#ef4444'
    }

    fig = px.scatter(
        df, x='X', y='Y', color='Type',
        text='Label',
        color_discrete_map=color_map,
        title='2D Embedding Space (UMAP)'
    )

    fig.update_traces(
        marker=dict(size=15, line=dict(width=2, color='white')),
        textposition='top center',
        textfont=dict(size=11)
    )

    # Add connection lines
    anchor_coords = coords_2d[0]
    for i in range(1, len(coords_2d)):
        target_coords = coords_2d[i]
        line_color = color_map[types[i]]
        line_width = 3 if types[i] == 'positive' else 1

        fig.add_shape(
            type='line',
            x0=anchor_coords[0], y0=anchor_coords[1],
            x1=target_coords[0], y1=target_coords[1],
            line=dict(color=line_color, width=line_width),
            opacity=0.5
        )

    fig.update_layout(
        showlegend=True,
        legend=dict(title='Type', orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='Inter'),
        xaxis=dict(showgrid=True, gridcolor='#e5e7eb'),
        yaxis=dict(showgrid=True, gridcolor='#e5e7eb')
    )

    return fig

def load_dataset(file):
    """Load JSONL dataset"""
    try:
        content = file.read().decode('utf-8')
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        dataset = [json.loads(line) for line in lines]
        return dataset
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

def save_dataset(dataset):
    """Convert dataset to JSONL for download"""
    return '\n'.join([json.dumps(item) for item in dataset])

def get_similarity_color(sim: float) -> str:
    """Get color class based on similarity value"""
    if sim > 0.6:
        return 'sim-high'
    elif sim > 0.4:
        return 'sim-medium'
    else:
        return 'sim-low'

def get_current_item():
    """Get current triplet item"""
    if st.session_state.dataset is None or st.session_state.current_index >= len(st.session_state.dataset):
        return None
    return st.session_state.dataset[st.session_state.current_index]

def save_current_edits(code, positive, negatives):
    """Save edits to current item"""
    if st.session_state.dataset is None:
        return
    idx = st.session_state.current_index
    item = st.session_state.dataset[idx]
    item['code_snippet'] = code
    item['conceptual_feedback'] = positive
    item['negative_feedbacks'] = negatives
    st.session_state.edited_items.add(idx)

def compute_global_embeddings(dataset, n_samples, model):
    """
    Compute embeddings for N random triplets from dataset

    Returns:
    - embeddings: numpy array of all embeddings
    - labels: list of labels for each point
    - types: list of types (anchor/positive/negative)
    - connections: list of tuples (start_idx, end_idx, type) for drawing edges
    """
    import random

    # Sample N random examples
    if n_samples > len(dataset):
        n_samples = len(dataset)

    sampled_items = random.sample(dataset, n_samples)

    all_texts = []
    labels = []
    types = []
    connections = []

    current_idx = 0

    for item_idx, item in enumerate(sampled_items):
        code = item.get('code_snippet', '')
        positive = item.get('conceptual_feedback', item.get('refined_feedback', ''))
        negatives = item.get('negative_feedbacks', [])

        # Handle legacy format
        if not negatives and 'hard_negative_feedback' in item:
            negatives = [item['hard_negative_feedback']]

        # Add anchor (code)
        anchor_idx = current_idx
        all_texts.append(code[:100])  # Truncate for display
        labels.append(f"Code {item_idx+1}")
        types.append('anchor')
        current_idx += 1

        # Add positive
        positive_idx = current_idx
        all_texts.append(positive)
        labels.append(f"Pos {item_idx+1}")
        types.append('positive')
        connections.append((anchor_idx, positive_idx, 'positive'))
        current_idx += 1

        # Add negatives
        for neg_idx, neg in enumerate(negatives):
            negative_idx = current_idx
            all_texts.append(neg)
            labels.append(f"Neg {item_idx+1}.{neg_idx+1}")
            types.append('negative')
            connections.append((anchor_idx, negative_idx, 'negative'))
            current_idx += 1

    # Compute embeddings
    embeddings = model.encode(all_texts)

    return {
        'embeddings': embeddings,
        'labels': labels,
        'types': types,
        'connections': connections
    }

def create_global_3d_visualization(embeddings_data: Dict):
    """Create 3D visualization with N triplets and their connections using UMAP"""
    embeddings = embeddings_data['embeddings']
    labels = embeddings_data['labels']
    types = embeddings_data['types']
    connections = embeddings_data['connections']

    # Reduce to 3D using UMAP
    n_neighbors = min(15, len(embeddings) - 1)
    if n_neighbors < 2:
        n_neighbors = 2

    reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=n_neighbors)
    coords_3d = reducer.fit_transform(embeddings)

    # Color mapping
    color_map = {
        'anchor': '#3b82f6',  # Blue
        'positive': '#10b981',  # Green
        'negative': '#ef4444'  # Red
    }

    colors = [color_map[t] for t in types]

    # Create figure
    fig = go.Figure()

    # Add all connection lines first (so they're behind points)
    for start_idx, end_idx, conn_type in connections:
        start_coords = coords_3d[start_idx]
        end_coords = coords_3d[end_idx]

        line_color = '#10b981' if conn_type == 'positive' else '#ef4444'
        line_width = 3 if conn_type == 'positive' else 1

        fig.add_trace(go.Scatter3d(
            x=[start_coords[0], end_coords[0]],
            y=[start_coords[1], end_coords[1]],
            z=[start_coords[2], end_coords[2]],
            mode='lines',
            line=dict(color=line_color, width=line_width),
            opacity=0.4,
            showlegend=False,
            hoverinfo='skip'
        ))

    # Add points
    fig.add_trace(go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='markers+text',
        marker=dict(
            size=10,
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=labels,
        textposition='top center',
        textfont=dict(size=9, color='#1a1a1a'),
        hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>',
        name='Embeddings'
    ))

    # Layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showgrid=True, gridcolor='#e5e7eb', showbackground=False),
            yaxis=dict(showgrid=True, gridcolor='#e5e7eb', showbackground=False),
            zaxis=dict(showgrid=True, gridcolor='#e5e7eb', showbackground=False),
            bgcolor='white'
        ),
        showlegend=False,
        height=700,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='white',
        font=dict(family='Inter', size=12)
    )

    return fig

# Sidebar
with st.sidebar:
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload JSONL file", type=['jsonl', 'json'], label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.dataset is None:
            dataset = load_dataset(uploaded_file)
            if dataset:
                st.session_state.dataset = dataset
                st.session_state.current_index = 0
                st.session_state.similarities = {}
                st.session_state.embeddings_cache = {}
                st.success(f"Loaded {len(dataset)} examples")

    st.markdown('<div class="section-header">Embedding Model</div>', unsafe_allow_html=True)
    model_options = [
        "sentence-transformers/all-MiniLM-L6-v2",
        "google/embeddinggemma-300m",
        "microsoft/graphcodebert-base"
    ]
    selected_model = st.selectbox("Select model", model_options, index=0, label_visibility="collapsed")

    if st.button("Load Model", use_container_width=True):
        st.session_state.model = load_embedding_model(selected_model)
        st.success("Model loaded")

    if st.session_state.dataset:
        st.markdown('<div class="section-header">Export</div>', unsafe_allow_html=True)
        jsonl_content = save_dataset(st.session_state.dataset)
        st.download_button(
            label="Download Dataset",
            data=jsonl_content,
            file_name="edited_dataset.jsonl",
            mime="application/jsonl",
            use_container_width=True
        )

        if st.session_state.edited_items:
            st.caption(f"{len(st.session_state.edited_items)} items edited")

# Main content
if st.session_state.dataset is None:
    st.markdown('<div class="main-header">FFGen Triplet Viewer</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Visualize, edit and analyze training triplets</div>', unsafe_allow_html=True)

    st.info("Upload a JSONL dataset file from the sidebar to begin")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Visualize**")
        st.caption("3D interactive embeddings")
    with col2:
        st.markdown("**Edit**")
        st.caption("Modify triplets inline")
    with col3:
        st.markdown("**Analyze**")
        st.caption("Compute similarities")

else:
    total = len(st.session_state.dataset)
    st.markdown('<div class="main-header">Triplet Viewer</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subtitle">Example {st.session_state.current_index + 1} of {total}</div>', unsafe_allow_html=True)

    # Navigation
    col1, col2, col3, col4, col5 = st.columns([2, 2, 4, 2, 2])

    with col1:
        if st.button("Previous", use_container_width=True, disabled=st.session_state.current_index == 0):
            st.session_state.current_index -= 1
            st.rerun()

    with col2:
        jump_to = st.number_input("Jump", min_value=1, max_value=total, value=st.session_state.current_index + 1, label_visibility="collapsed")
        if jump_to - 1 != st.session_state.current_index:
            st.session_state.current_index = jump_to - 1
            st.rerun()

    with col4:
        if st.button("Compute Similarities", use_container_width=True):
            if st.session_state.model is None:
                st.warning("Load a model first")
            else:
                item = get_current_item()
                positive = item.get('conceptual_feedback', item.get('refined_feedback', ''))
                negatives = item.get('negative_feedbacks', [])
                with st.spinner("Computing..."):
                    sims = compute_similarities(positive, negatives, st.session_state.model)
                    st.session_state.similarities[st.session_state.current_index] = sims
                st.success("Done")
                st.rerun()

    with col5:
        if st.button("Next", use_container_width=True, disabled=st.session_state.current_index >= total - 1):
            st.session_state.current_index += 1
            st.rerun()

    st.markdown("---")

    item = get_current_item()
    if item is None:
        st.error("Invalid item")
        st.stop()

    # Code section
    st.markdown('<div class="section-header">Code Snippet (Anchor)</div>', unsafe_allow_html=True)
    code_snippet = st.text_area(
        "Code",
        value=item.get('code_snippet', ''),
        height=200,
        label_visibility="collapsed",
        key=f"code_{st.session_state.current_index}"
    )

    # Positive and Negatives
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-header">Conceptual Feedback (Positive)</div>', unsafe_allow_html=True)
        positive_feedback = st.text_area(
            "Positive",
            value=item.get('conceptual_feedback', item.get('refined_feedback', '')),
            height=250,
            label_visibility="collapsed",
            key=f"positive_{st.session_state.current_index}"
        )

    with col_right:
        st.markdown('<div class="section-header">Negative Feedbacks</div>', unsafe_allow_html=True)

        negatives = item.get('negative_feedbacks', [])
        if not negatives and 'hard_negative_feedback' in item:
            negatives = [item['hard_negative_feedback']]

        sims = st.session_state.similarities.get(st.session_state.current_index, [])

        if sims:
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            with metrics_col1:
                st.metric("Mean", f"{np.mean(sims):.3f}")
            with metrics_col2:
                st.metric("Min", f"{np.min(sims):.3f}")
            with metrics_col3:
                st.metric("Max", f"{np.max(sims):.3f}")

        st.caption(f"{len(negatives)} negatives")

        updated_negatives = []
        for idx, neg in enumerate(negatives):
            sim = sims[idx] if idx < len(sims) else None

            if sim is not None:
                color_class = get_similarity_color(sim)
                st.markdown(f'<div style="display:flex; align-items:center; margin-bottom:0.5rem;"><span style="font-weight:600; margin-right:1rem;">Negative {idx + 1}</span><span style="color:#6b7280;">Similarity: {sim:.3f}</span><div class="similarity-bar {color_class}" style="width:{sim*100}%; margin-left:1rem;"></div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f"**Negative {idx + 1}**")

            neg_text = st.text_area(
                f"Negative {idx + 1}",
                value=neg,
                height=100,
                label_visibility="collapsed",
                key=f"neg_{st.session_state.current_index}_{idx}"
            )
            updated_negatives.append(neg_text)

        if st.button("Add Negative"):
            updated_negatives.append("New negative feedback")

    if st.button("Save Changes", type="primary"):
        save_current_edits(code_snippet, positive_feedback, updated_negatives)
        st.success("Saved")

    # Tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Current Stats", "Dataset Stats", "Similarity Analysis", "3D Visualization", "RAG Testing"])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Code Length", f"{len(code_snippet)}")
        with col2:
            st.metric("Feedback Length", f"{len(positive_feedback)}")
        with col3:
            st.metric("Negatives", len(updated_negatives))
        with col4:
            item_id = item.get('id', f'ex_{st.session_state.current_index}')
            st.metric("ID", item_id)

    with tab2:
        if st.button("Refresh Stats"):
            dataset = st.session_state.dataset
            code_lengths = [len(item.get('code_snippet', '')) for item in dataset]
            feedback_lengths = [len(item.get('conceptual_feedback', item.get('refined_feedback', ''))) for item in dataset]
            neg_counts = []
            for item in dataset:
                negs = item.get('negative_feedbacks', [])
                if not negs and 'hard_negative_feedback' in item:
                    negs = [item['hard_negative_feedback']]
                neg_counts.append(len(negs))

            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            with metric_col1:
                st.metric("Examples", len(dataset))
            with metric_col2:
                st.metric("Total Negatives", sum(neg_counts))
            with metric_col3:
                st.metric("Avg Negatives", f"{np.mean(neg_counts):.2f}")
            with metric_col4:
                st.metric("Avg Code Length", f"{np.mean(code_lengths):.0f}")

            dist_col1, dist_col2 = st.columns(2)
            with dist_col1:
                fig = px.histogram(x=code_lengths, nbins=50, title="Code Length Distribution")
                fig.update_traces(marker_color='#3b82f6')
                fig.update_layout(showlegend=False, paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)
            with dist_col2:
                fig = px.histogram(x=neg_counts, nbins=20, title="Negatives Distribution")
                fig.update_traces(marker_color='#ef4444')
                fig.update_layout(showlegend=False, paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        if not st.session_state.similarities:
            st.info("Click 'Compute Similarities' to analyze")
        else:
            all_sims = []
            for sims in st.session_state.similarities.values():
                all_sims.extend(sims)

            if all_sims:
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Mean", f"{np.mean(all_sims):.3f}")
                with metric_col2:
                    st.metric("Median", f"{np.median(all_sims):.3f}")
                with metric_col3:
                    st.metric("Std", f"{np.std(all_sims):.3f}")
                with metric_col4:
                    high_sim_pct = (np.array(all_sims) > 0.5).sum() / len(all_sims) * 100
                    st.metric("Pairs > 0.5", f"{high_sim_pct:.1f}%")

                fig = px.histogram(x=all_sims, nbins=50, title="Similarity Distribution")
                fig.add_vline(x=0.5, line_dash="dash", line_color="#ef4444", annotation_text="Margin")
                fig.add_vline(x=np.mean(all_sims), line_dash="dash", line_color="#10b981", annotation_text="Mean")
                fig.update_traces(marker_color='#3b82f6')
                fig.update_layout(showlegend=False, paper_bgcolor='white')
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("**Quality Assessment**")
                if np.mean(all_sims) > 0.6:
                    st.error("High similarity - negatives too similar to positives")
                elif np.mean(all_sims) > 0.5:
                    st.warning("Medium similarity - negatives could be more distinct")
                else:
                    st.success("Good separation - well-separated negatives")

    with tab4:
        if st.session_state.model is None:
            st.info("Load an embedding model to visualize")
        else:
            viz_col1, viz_col2 = st.columns([3, 1])

            with viz_col2:
                st.markdown("**Settings**")
                st.caption("Dimensionality reduction: UMAP")

                if st.button("Generate Visualization", use_container_width=True):
                    with st.spinner("Computing embeddings..."):
                        emb_data = compute_embeddings_for_visualization(item, st.session_state.model)
                        st.session_state.embeddings_cache[st.session_state.current_index] = emb_data
                    st.success("Done")
                    st.rerun()

            with viz_col1:
                if st.session_state.current_index in st.session_state.embeddings_cache:
                    emb_data = st.session_state.embeddings_cache[st.session_state.current_index]
                    fig_3d = create_3d_visualization(emb_data)
                    st.plotly_chart(fig_3d, use_container_width=True)

                    st.caption("Blue: Anchor (code) | Green: Positive | Red: Negatives")
                    st.caption("Lines show connections from anchor. Hover points for details.")
                else:
                    st.info("Click 'Generate Visualization' to see 3D embedding space")

# Global UMAP Visualization Section
if st.session_state.dataset and st.session_state.model:
    st.markdown("---")
    st.markdown('<div class="section-header">Global Dataset Visualization</div>', unsafe_allow_html=True)
    st.caption("Visualize multiple triplets together in 3D UMAP space")

    col_controls, col_viz = st.columns([1, 3])

    with col_controls:
        st.markdown("**Settings**")
        n_samples = st.number_input(
            "Number of examples",
            min_value=1,
            max_value=min(100, len(st.session_state.dataset)),
            value=min(10, len(st.session_state.dataset)),
            help="Random examples to include in visualization"
        )

        if st.button("Generate Global UMAP", use_container_width=True, type="primary"):
            with st.spinner(f"Computing embeddings for {n_samples} triplets..."):
                global_emb = compute_global_embeddings(
                    st.session_state.dataset,
                    n_samples,
                    st.session_state.model
                )
                st.session_state.global_embeddings = global_emb
            st.success(f"Generated visualization with {len(global_emb['embeddings'])} points")
            st.rerun()

        if st.session_state.global_embeddings:
            st.markdown("**Info**")
            emb_data = st.session_state.global_embeddings
            n_points = len(emb_data['embeddings'])
            n_connections = len(emb_data['connections'])
            st.caption(f"Points: {n_points}")
            st.caption(f"Connections: {n_connections}")
            st.caption(f"Dimensionality: UMAP 3D")

    with col_viz:
        if st.session_state.global_embeddings:
            fig_global = create_global_3d_visualization(st.session_state.global_embeddings)
            st.plotly_chart(fig_global, use_container_width=True)

            st.caption("Blue: Code (anchor) | Green: Positive feedback | Red: Negative feedbacks")
            st.caption("Each code is connected to its positive (thick green) and negatives (thin red)")
        else:
            st.info("Click 'Generate Global UMAP' to visualize multiple triplets together")

    with tab5:
        st.subheader("RAG Testing")

        if not CHROMADB_AVAILABLE:
            st.error("ChromaDB not installed. Install with: `uv pip install chromadb`")
        else:
            st.info("Test your trained model's retrieval performance interactively")

            # Initialize session state for RAG
            if 'rag_corpus' not in st.session_state:
                st.session_state.rag_corpus = []
            if 'rag_embeddings' not in st.session_state:
                st.session_state.rag_embeddings = None
            if 'rag_collection' not in st.session_state:
                st.session_state.rag_collection = None

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("##### Setup")

                # Model selection for RAG
                rag_model_option = st.selectbox(
                    "RAG Model",
                    ["sentence-transformers/all-MiniLM-L6-v2", "google/embeddinggemma-300m", "Custom path"],
                    key="rag_model_selector",
                    help="Select the model to test for retrieval"
                )

                if rag_model_option == "Custom path":
                    custom_rag_model = st.text_input("Model path:", key="custom_rag_model_input")
                    rag_model_to_use = custom_rag_model if custom_rag_model else "sentence-transformers/all-MiniLM-L6-v2"
                else:
                    rag_model_to_use = rag_model_option

                # Corpus source
                corpus_source = st.radio(
                    "Corpus source:",
                    ["Use current dataset", "Upload separate corpus"],
                    key="corpus_source"
                )

                if corpus_source == "Upload separate corpus":
                    corpus_file = st.file_uploader(
                        "Upload corpus JSONL",
                        type=['jsonl'],
                        key="rag_corpus_uploader",
                        help="Upload a JSONL file with documents to index"
                    )
                else:
                    corpus_file = None

                # Index button
                if st.button("Index Corpus", type="primary"):
                    try:
                        with st.spinner("Loading model and indexing corpus..."):
                            # Load model
                            if rag_model_to_use not in st.session_state.models_cache:
                                st.session_state.models_cache[rag_model_to_use] = SentenceTransformer(rag_model_to_use)
                            model = st.session_state.models_cache[rag_model_to_use]

                            # Prepare corpus
                            if corpus_source == "Use current dataset" and 'dataset' in st.session_state:
                                st.session_state.rag_corpus = [
                                    {
                                        "id": item.get('id', f"doc_{i}"),
                                        "text": item.get('conceptual_feedback', ''),
                                        "code": item.get('code_snippet', ''),
                                        "metadata": item
                                    }
                                    for i, item in enumerate(st.session_state.dataset)
                                ]
                            elif corpus_file is not None:
                                st.session_state.rag_corpus = []
                                content = corpus_file.read().decode('utf-8')
                                for line in content.strip().split('\n'):
                                    if line.strip():
                                        item = json.loads(line)
                                        st.session_state.rag_corpus.append({
                                            "id": item.get("id", len(st.session_state.rag_corpus)),
                                            "text": item.get("conceptual_feedback", item.get("text", "")),
                                            "code": item.get("code_snippet", item.get("code", "")),
                                            "metadata": item
                                        })
                            else:
                                st.error("No corpus available")
                                st.stop()

                            # Compute embeddings
                            texts = [doc["text"] for doc in st.session_state.rag_corpus]
                            st.session_state.rag_embeddings = model.encode(
                                texts,
                                batch_size=32,
                                show_progress_bar=False,
                                convert_to_numpy=True,
                                normalize_embeddings=True
                            )

                            # Index in ChromaDB
                            client = chromadb.Client(Settings(anonymized_telemetry=False, allow_reset=True))
                            try:
                                client.delete_collection("ffgen_rag_test")
                            except:
                                pass

                            st.session_state.rag_collection = client.create_collection(
                                name="ffgen_rag_test",
                                metadata={"description": "FFGen RAG testing"}
                            )

                            st.session_state.rag_collection.add(
                                ids=[str(doc["id"]) for doc in st.session_state.rag_corpus],
                                embeddings=st.session_state.rag_embeddings.tolist(),
                                documents=texts,
                                metadatas=[{"index": i} for i in range(len(st.session_state.rag_corpus))]
                            )

                        st.success(f"Indexed {len(st.session_state.rag_corpus)} documents")

                    except Exception as e:
                        st.error(f"Error indexing corpus: {e}")

            with col2:
                st.markdown("##### Search")

                if st.session_state.rag_corpus and st.session_state.rag_embeddings is not None:
                    st.caption(f"Corpus: {len(st.session_state.rag_corpus)} documents indexed")

                    # Query input
                    query = st.text_area(
                        "Enter query:",
                        height=100,
                        placeholder="Type a code snippet or feedback to search...",
                        key="rag_query_input"
                    )

                    top_k = st.slider("Top-k results:", 1, 20, 5, key="rag_top_k")

                    if st.button("Search", type="secondary") and query.strip():
                        try:
                            # Load model
                            if rag_model_to_use not in st.session_state.models_cache:
                                st.session_state.models_cache[rag_model_to_use] = SentenceTransformer(rag_model_to_use)
                            model = st.session_state.models_cache[rag_model_to_use]

                            # Encode query
                            query_embedding = model.encode(
                                query,
                                convert_to_numpy=True,
                                normalize_embeddings=True
                            )

                            # Compute similarities
                            similarities = np.dot(st.session_state.rag_embeddings, query_embedding)
                            top_indices = np.argsort(similarities)[::-1][:top_k]

                            # Display results
                            st.markdown("---")
                            st.markdown(f"##### Top {top_k} Results")

                            for rank, idx in enumerate(top_indices, start=1):
                                score = similarities[idx]
                                doc = st.session_state.rag_corpus[idx]

                                with st.expander(f"#{rank} - Score: {score:.4f} - ID: {doc['id']}", expanded=(rank == 1)):
                                    if doc.get('code'):
                                        st.markdown("**Code:**")
                                        st.code(doc['code'], language='python')

                                    st.markdown("**Feedback:**")
                                    st.write(doc['text'])

                                    st.caption(f"Similarity: {score:.4f} | Index: {idx}")

                        except Exception as e:
                            st.error(f"Error during search: {e}")
                else:
                    st.info("Index a corpus first to enable search")

            # Statistics section
            if st.session_state.rag_corpus and st.session_state.rag_embeddings is not None:
                st.markdown("---")
                st.markdown("##### Corpus Statistics")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Documents", len(st.session_state.rag_corpus))
                with col2:
                    avg_len = np.mean([len(doc['text']) for doc in st.session_state.rag_corpus])
                    st.metric("Avg Text Length", f"{avg_len:.0f}")
                with col3:
                    embedding_dim = st.session_state.rag_embeddings.shape[1]
                    st.metric("Embedding Dim", embedding_dim)
                with col4:
                    st.metric("Model", rag_model_to_use.split('/')[-1][:20])

st.markdown("---")
st.caption("FFGen Triplet Viewer | Built with Streamlit")
