"""
UI component rendering functions
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import json
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer

from backend.embeddings import compute_embeddings_for_visualization, compute_global_embeddings
from backend.similarity import compute_similarities, get_similarity_color
from backend.rag import RAGTester, CHROMADB_AVAILABLE
from frontend.visualizations import create_3d_visualization, create_global_3d_visualization


def render_navigation(current_index: int, total: int) -> Optional[int]:
    """
    Render navigation controls.

    Args:
        current_index: Current item index
        total: Total number of items

    Returns:
        New index if navigation occurred, None otherwise
    """
    col1, col2, col3, col4, col5 = st.columns([2, 2, 4, 2, 2])

    new_index = None

    with col1:
        if st.button("Previous", use_container_width=True, disabled=current_index == 0):
            new_index = current_index - 1

    with col2:
        jump_to = st.number_input("Jump", min_value=1, max_value=total, value=current_index + 1, label_visibility="collapsed")
        if jump_to - 1 != current_index:
            new_index = jump_to - 1

    with col4:
        if st.button("Compute Similarities", use_container_width=True):
            if st.session_state.model is None:
                st.warning("Load a model first")
            else:
                item = st.session_state.dataset[current_index]
                positive = item.get('conceptual_feedback', item.get('refined_feedback', ''))
                negatives = item.get('negative_feedbacks', [])
                with st.spinner("Computing..."):
                    sims = compute_similarities(positive, negatives, st.session_state.model)
                    st.session_state.similarities[current_index] = sims
                st.success("Done")
                st.rerun()

    with col5:
        if st.button("Next", use_container_width=True, disabled=current_index >= total - 1):
            new_index = current_index + 1

    return new_index


def render_code_section(item: Dict, current_index: int) -> str:
    """
    Render code snippet section.

    Args:
        item: Dataset item
        current_index: Current item index

    Returns:
        Edited code snippet text
    """
    st.markdown('<div class="section-header">Code Snippet (Anchor)</div>', unsafe_allow_html=True)
    code_snippet = st.text_area(
        "Code",
        value=item.get('code_snippet', ''),
        height=200,
        label_visibility="collapsed",
        key=f"code_{current_index}"
    )
    return code_snippet


def render_positive_feedback(item: Dict, current_index: int) -> str:
    """
    Render positive feedback section.

    Args:
        item: Dataset item
        current_index: Current item index

    Returns:
        Edited positive feedback text
    """
    st.markdown('<div class="section-header">Conceptual Feedback (Positive)</div>', unsafe_allow_html=True)
    positive_feedback = st.text_area(
        "Positive",
        value=item.get('conceptual_feedback', item.get('refined_feedback', '')),
        height=250,
        label_visibility="collapsed",
        key=f"positive_{current_index}"
    )
    return positive_feedback


def render_negative_feedbacks(item: Dict, current_index: int, similarities: List[float]) -> List[str]:
    """
    Render negative feedbacks section.

    Args:
        item: Dataset item
        current_index: Current item index
        similarities: List of similarity scores (can be empty)

    Returns:
        List of edited negative feedback texts
    """
    st.markdown('<div class="section-header">Negative Feedbacks</div>', unsafe_allow_html=True)

    negatives = item.get('negative_feedbacks', [])
    if not negatives and 'hard_negative_feedback' in item:
        negatives = [item['hard_negative_feedback']]

    if similarities:
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("Mean", f"{np.mean(similarities):.3f}")
        with metrics_col2:
            st.metric("Min", f"{np.min(similarities):.3f}")
        with metrics_col3:
            st.metric("Max", f"{np.max(similarities):.3f}")

    st.caption(f"{len(negatives)} negatives")

    updated_negatives = []
    for idx, neg in enumerate(negatives):
        sim = similarities[idx] if idx < len(similarities) else None

        if sim is not None:
            color_class = get_similarity_color(sim)
            st.markdown(
                f'<div style="display:flex; align-items:center; margin-bottom:0.5rem;">'
                f'<span style="font-weight:600; margin-right:1rem;">Negative {idx + 1}</span>'
                f'<span style="color:#6b7280;">Similarity: {sim:.3f}</span>'
                f'<div class="similarity-bar {color_class}" style="width:{sim*100}%; margin-left:1rem;"></div>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(f"**Negative {idx + 1}**")

        neg_text = st.text_area(
            f"Negative {idx + 1}",
            value=neg,
            height=100,
            label_visibility="collapsed",
            key=f"neg_{current_index}_{idx}"
        )
        updated_negatives.append(neg_text)

    if st.button("Add Negative"):
        updated_negatives.append("New negative feedback")

    return updated_negatives


def render_statistics_tab(item: Dict, code_snippet: str, positive_feedback: str, updated_negatives: List[str]):
    """Render current item statistics tab"""
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


def render_dataset_stats_tab(dataset: List[Dict]):
    """Render dataset-wide statistics tab"""
    if st.button("Refresh Stats"):
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


def render_similarity_analysis_tab(similarities_dict: Dict):
    """Render similarity analysis tab"""
    if not similarities_dict:
        st.info("Click 'Compute Similarities' to analyze")
    else:
        all_sims = []
        for sims in similarities_dict.values():
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


def render_3d_visualization_tab(item: Dict, current_index: int, model: Optional[SentenceTransformer]):
    """Render 3D visualization tab"""
    if model is None:
        st.info("Load an embedding model to visualize")
    else:
        viz_col1, viz_col2 = st.columns([3, 1])

        with viz_col2:
            st.markdown("**Settings**")
            st.caption("Dimensionality reduction: UMAP")

            if st.button("Generate Visualization", use_container_width=True):
                with st.spinner("Computing embeddings..."):
                    emb_data = compute_embeddings_for_visualization(item, model)
                    st.session_state.embeddings_cache[current_index] = emb_data
                st.success("Done")
                st.rerun()

        with viz_col1:
            if current_index in st.session_state.embeddings_cache:
                emb_data = st.session_state.embeddings_cache[current_index]
                fig_3d = create_3d_visualization(emb_data)
                st.plotly_chart(fig_3d, use_container_width=True)

                st.caption("Blue: Anchor (code) | Green: Positive | Red: Negatives")
                st.caption("Lines show connections from anchor. Hover points for details.")
            else:
                st.info("Click 'Generate Visualization' to see 3D embedding space")


def render_rag_testing_tab(dataset: List[Dict], model: Optional[SentenceTransformer]):
    """Render MIPS-based RAG testing tab: Code as query, Feedback as response"""
    st.subheader("MIPS Retrieval: Code → Feedback")
    st.caption("Maximum Inner Product Search - Query with code, retrieve relevant feedbacks")

    # ChromaDB status
    if CHROMADB_AVAILABLE:
        st.success("✓ ChromaDB available - embeddings will be cached")
    else:
        st.warning("ChromaDB not available - embeddings won't be cached. Install with: `uv pip install chromadb`")

    # Initialize session state for RAG (will recreate with callback when indexing)
    if 'rag_tester' not in st.session_state:
        st.session_state.rag_tester = RAGTester()

    # Progress messages container
    if 'rag_progress_messages' not in st.session_state:
        st.session_state.rag_progress_messages = []

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### Setup")

        # Model selection for RAG
        rag_model_option = st.selectbox(
            "Embedding Model",
            ["sentence-transformers/all-MiniLM-L6-v2", "google/embeddinggemma-300m", "Custom path"],
            key="rag_model_selector",
            help="Select the embedding model for retrieval"
        )

        if rag_model_option == "Custom path":
            custom_rag_model = st.text_input("Model path:", key="custom_rag_model_input")
            rag_model_to_use = custom_rag_model if custom_rag_model else "sentence-transformers/all-MiniLM-L6-v2"
        else:
            rag_model_to_use = rag_model_option

        # Corpus source
        corpus_source = st.radio(
            "Feedback corpus source:",
            ["Use current dataset", "Upload JSONL"],
            key="corpus_source"
        )

        corpus_file = None
        if corpus_source == "Upload JSONL":
            corpus_file = st.file_uploader(
                "Upload feedback corpus",
                type=['jsonl'],
                key="rag_corpus_uploader",
                help="Upload JSONL with code_snippet and conceptual_feedback fields"
            )

        # Index button
        if st.button("Index Feedbacks", type="primary"):
            try:
                # Clear previous messages
                st.session_state.rag_progress_messages = []

                # Progress container
                progress_placeholder = st.empty()
                status_container = st.status("Indexing feedbacks...", expanded=True)

                # Define callback to capture progress
                def progress_callback(msg):
                    st.session_state.rag_progress_messages.append(msg)
                    with status_container:
                        # Show all messages
                        for m in st.session_state.rag_progress_messages:
                            st.write(m)

                with status_container:
                    st.write("📦 Loading embedding model...")

                # Load model
                if 'models_cache' not in st.session_state:
                    st.session_state.models_cache = {}

                if rag_model_to_use not in st.session_state.models_cache:
                    from backend.embeddings import load_embedding_model
                    st.session_state.models_cache[rag_model_to_use] = load_embedding_model(rag_model_to_use)

                rag_model = st.session_state.models_cache[rag_model_to_use]

                with status_container:
                    st.write("✓ Model loaded")

                # Create RAG tester with callback
                st.session_state.rag_tester = RAGTester(progress_callback=progress_callback)

                # Index corpus (with ChromaDB caching if available)
                use_cache = CHROMADB_AVAILABLE

                with status_container:
                    if use_cache:
                        st.write("💾 ChromaDB cache enabled")
                    else:
                        st.write("⚠️ ChromaDB not available - computing without cache")

                    st.write("🚀 Starting indexing...")

                if corpus_source == "Use current dataset":
                    st.session_state.rag_tester.index_corpus(dataset, rag_model, use_chromadb=use_cache)
                else:
                    if corpus_file is not None:
                        st.session_state.rag_tester.index_corpus_from_file(corpus_file, rag_model, use_chromadb=use_cache)
                    else:
                        st.error("No corpus file provided")
                        st.stop()

                with status_container:
                    st.write("✅ Indexing complete!")

                stats = st.session_state.rag_tester.get_corpus_stats()
                status_container.update(label=f"✅ Indexed {stats['n_feedbacks']} feedbacks", state="complete")
                st.success(f"🎉 Ready to search! {stats['n_feedbacks']} feedbacks indexed.")

            except Exception as e:
                st.error(f"❌ Error indexing corpus: {e}")
                import traceback
                st.code(traceback.format_exc())

    with col2:
        st.markdown("##### Search")

        stats = st.session_state.rag_tester.get_corpus_stats()
        if stats is not None:
            st.caption(f"📊 {stats['n_feedbacks']} feedbacks indexed")

            # Query input - CODE SNIPPET
            code_query = st.text_area(
                "Enter code snippet:",
                height=120,
                placeholder="int my_strlen(char *str) {\n    int i = 0;\n    while (str[i])\n        i++;\n    return i;\n}",
                key="rag_code_query"
            )

            top_k = st.slider("Top-k results:", 1, 20, 5, key="rag_top_k")

            if st.button("🔍 Search Feedbacks", type="secondary") and code_query.strip():
                try:
                    results = st.session_state.rag_tester.search(code_query, top_k)

                    # Display results
                    st.markdown("---")
                    st.markdown(f"##### Top {top_k} Retrieved Feedbacks")

                    for rank, similarity, feedback_doc in results:
                        # Color code by similarity score
                        if similarity >= 0.7:
                            badge_color = "🟢"
                        elif similarity >= 0.5:
                            badge_color = "🟡"
                        else:
                            badge_color = "🔴"

                        with st.expander(
                            f"{badge_color} Rank #{rank} | Similarity: {similarity:.4f}",
                            expanded=(rank <= 2)
                        ):
                            # Display similarity score prominently
                            st.markdown(f"**Similarity Score: `{similarity:.4f}`**")
                            st.progress(min(similarity, 1.0))

                            # Display feedback
                            st.markdown("**📝 Retrieved Feedback:**")
                            st.info(feedback_doc['feedback'])

                            # Display associated code
                            if feedback_doc.get('code'):
                                st.markdown("**💻 Associated Code:**")
                                st.code(feedback_doc['code'], language='c')

                            st.caption(f"ID: {feedback_doc['id']}")

                except Exception as e:
                    st.error(f"Error during search: {e}")
        else:
            st.info("👆 Index a feedback corpus first to enable search")

    # Statistics section
    stats = st.session_state.rag_tester.get_corpus_stats()
    if stats is not None:
        st.markdown("---")
        st.markdown("##### Corpus Statistics")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Feedbacks", stats['n_feedbacks'])
        with col2:
            st.metric("Avg Length", f"{stats['avg_feedback_length']:.0f}")
        with col3:
            st.metric("Embedding Dim", stats['embedding_dim'])
        with col4:
            st.metric("Model", rag_model_to_use.split('/')[-1][:20])


def render_global_visualization(dataset: List[Dict], model: Optional[SentenceTransformer]):
    """Render global dataset visualization section"""
    st.markdown("---")
    st.markdown('<div class="section-header">Global Dataset Visualization</div>', unsafe_allow_html=True)
    st.caption("Visualize multiple triplets together in 3D UMAP space")

    col_controls, col_viz = st.columns([1, 3])

    with col_controls:
        st.markdown("**Settings**")
        n_samples = st.number_input(
            "Number of examples",
            min_value=1,
            max_value=min(100, len(dataset)),
            value=min(10, len(dataset)),
            help="Random examples to include in visualization"
        )

        if st.button("Generate Global UMAP", use_container_width=True, type="primary"):
            with st.spinner(f"Computing embeddings for {n_samples} triplets..."):
                global_emb = compute_global_embeddings(dataset, n_samples, model)
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
