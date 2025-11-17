#!/usr/bin/env python3
"""
FFGen Triplet Viewer - Streamlit App
Beautiful, minimal interface for visualizing and editing training triplets
"""

import streamlit as st

# Backend imports
from backend.data_loader import load_dataset, save_dataset


# Frontend imports
from frontend.styles import apply_custom_css
from frontend.components import (
    render_navigation,
    render_code_section,
    render_positive_feedback,
    render_negative_feedbacks,
    render_statistics_tab,
    render_dataset_stats_tab,
    render_similarity_analysis_tab,
    render_3d_visualization_tab,
    render_rag_testing_tab,
    render_global_visualization
)


# Page configuration
st.set_page_config(
    page_title="FFGen Triplet Viewer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'similarities' not in st.session_state:
    st.session_state.similarities = {}
if 'embeddings_cache' not in st.session_state:
    st.session_state.embeddings_cache = {}
if 'edited_items' not in st.session_state:
    st.session_state.edited_items = set()
if 'global_embeddings' not in st.session_state:
    st.session_state.global_embeddings = None





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


# Sidebar
with st.sidebar:
    st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload JSONL file", type=['jsonl', 'json'], label_visibility="collapsed")

    if uploaded_file:
        if st.session_state.dataset is None:
            try:
                dataset = load_dataset(uploaded_file)
                if dataset:
                    st.session_state.dataset = dataset
                    st.session_state.current_index = 0
                    st.session_state.similarities = {}
                    st.session_state.embeddings_cache = {}
                    st.success(f"Loaded {len(dataset)} examples")
            except Exception as e:
                st.error(f"Error loading dataset: {e}")

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
    new_index = render_navigation(st.session_state.current_index, total)
    if new_index is not None:
        st.session_state.current_index = new_index
        st.rerun()

    st.markdown("---")

    item = st.session_state.dataset[st.session_state.current_index]

    # Code section
    code_snippet = render_code_section(item, st.session_state.current_index)

    # Positive and Negatives
    col_left, col_right = st.columns([1, 1])

    with col_left:
        positive_feedback = render_positive_feedback(item, st.session_state.current_index)

    with col_right:
        sims = st.session_state.similarities.get(st.session_state.current_index, [])
        updated_negatives = render_negative_feedbacks(item, st.session_state.current_index, sims)

    if st.button("Save Changes", type="primary"):
        save_current_edits(code_snippet, positive_feedback, updated_negatives)
        st.success("Saved")

    # Tabs
    st.markdown("---")
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Current Stats",
        "Dataset Stats",
        "Similarity Analysis",
        "3D Visualization",
        "RAG Testing"
    ])

    with tab1:
        render_statistics_tab(item, code_snippet, positive_feedback, updated_negatives)

    with tab2:
        render_dataset_stats_tab(st.session_state.dataset)

    with tab3:
        render_similarity_analysis_tab(st.session_state.similarities)

    with tab4:
        render_3d_visualization_tab(item, st.session_state.current_index)

    with tab5:
        render_rag_testing_tab(st.session_state.dataset)

# Global UMAP Visualization Section
if st.session_state.dataset:
    render_global_visualization(st.session_state.dataset)

st.markdown("---")
st.caption("FFGen Triplet Viewer | Built with Streamlit")
