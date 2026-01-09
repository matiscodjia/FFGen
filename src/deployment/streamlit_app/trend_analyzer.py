"""
Trend Analysis Module for RAG Feedback Viewer
==============================================

This module analyzes patterns in student code submissions by tracking:
- Recurring difficulties by author_id and code_id
- Common error patterns through feedback clustering
- Topic identification for problematic areas
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go


class TrendAnalyzer:
    """Analyze trends and patterns in code feedback data"""

    def __init__(self):
        self.data_cache = None
        self.cluster_cache = None

    def load_data_from_collection(self, collection, dataset: List[Dict]) -> pd.DataFrame:
        """
        Load and structure data from ChromaDB collection and original dataset

        Args:
            collection: ChromaDB collection
            dataset: Original dataset with code_id and author_id

        Returns:
            DataFrame with all necessary fields
        """
        # Get all data from collection
        results = collection.get(include=['embeddings', 'documents', 'metadatas'])

        # Build DataFrame
        data = []
        for i, (doc_id, embedding, document, metadata) in enumerate(
            zip(results['ids'], results['embeddings'], results['documents'], results['metadatas'])
        ):
            # Extract index from doc_id (format: "doc_123")
            idx = int(doc_id.split('_')[1])

            # Get corresponding dataset entry
            if idx < len(dataset):
                original_entry = dataset[idx]
                data.append({
                    'id': doc_id,
                    'feedback': document,
                    'code': metadata.get('code', ''),
                    'code_id': original_entry.get('code_id', 'unknown'),
                    'author_id': original_entry.get('author_id', 'unknown'),
                    'embedding': embedding
                })

        df = pd.DataFrame(data)
        self.data_cache = df
        return df

    def analyze_authors_cluster_diversity(self, df_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze how many different clusters each author appears in.
        Authors in many clusters = struggling with multiple different concepts.

        Args:
            df_clustered: DataFrame with cluster labels and author_id

        Returns:
            DataFrame with author statistics sorted by cluster diversity
        """
        author_stats = df_clustered.groupby('author_id').agg({
            'cluster': lambda x: list(x),
            'feedback': 'count',
            'code_id': lambda x: list(x)
        }).reset_index()

        # Calculate cluster diversity metrics
        author_stats['unique_clusters'] = author_stats['cluster'].apply(lambda x: len(set(x)))
        author_stats['total_feedbacks'] = author_stats['feedback']
        author_stats['cluster_diversity_ratio'] = author_stats['unique_clusters'] / author_stats['total_feedbacks']
        author_stats['cluster_list'] = author_stats['cluster'].apply(lambda x: sorted(set(x)))

        # Rename columns
        author_stats = author_stats[['author_id', 'total_feedbacks', 'unique_clusters',
                                     'cluster_diversity_ratio', 'cluster_list', 'code_id']]
        author_stats.columns = ['author_id', 'total_feedbacks', 'unique_clusters',
                               'diversity_ratio', 'clusters', 'code_ids']

        # Sort by unique clusters (descending) then by total feedbacks
        author_stats = author_stats.sort_values(['unique_clusters', 'total_feedbacks'],
                                                ascending=[False, False])

        return author_stats

    def find_recurring_authors(self, df: pd.DataFrame, min_submissions: int = 3) -> pd.DataFrame:
        """
        Identify authors with multiple submissions

        Args:
            df: DataFrame with author_id column
            min_submissions: Minimum number of submissions to be considered recurring

        Returns:
            DataFrame with author statistics
        """
        author_stats = df.groupby('author_id').agg({
            'code_id': 'count',
            'feedback': lambda x: list(x)
        }).reset_index()

        author_stats.columns = ['author_id', 'submission_count', 'feedbacks']
        author_stats = author_stats[author_stats['submission_count'] >= min_submissions]
        author_stats = author_stats.sort_values('submission_count', ascending=False)

        return author_stats

    def find_recurring_code_patterns(self, df: pd.DataFrame, min_occurrences: int = 2) -> pd.DataFrame:
        """
        Identify code_ids that appear multiple times (same exercise attempted multiple times)

        Args:
            df: DataFrame with code_id column
            min_occurrences: Minimum occurrences to be considered recurring

        Returns:
            DataFrame with code pattern statistics
        """
        code_stats = df.groupby('code_id').agg({
            'author_id': 'count',
            'feedback': lambda x: list(x),
            'code': 'first'
        }).reset_index()

        code_stats.columns = ['code_id', 'occurrence_count', 'feedbacks', 'sample_code']
        code_stats = code_stats[code_stats['occurrence_count'] >= min_occurrences]
        code_stats = code_stats.sort_values('occurrence_count', ascending=False)

        return code_stats

    def find_optimal_clusters(self, embeddings: np.ndarray, max_k: int = 20) -> int:
        """
        Find optimal number of clusters using silhouette score

        Args:
            embeddings: Embedding matrix
            max_k: Maximum number of clusters to try

        Returns:
            Optimal number of clusters
        """
        from sklearn.metrics import silhouette_score

        # Try different k values
        min_k = max(2, min(5, len(embeddings) // 10))  # At least 2, typically start at 5
        max_k = min(max_k, len(embeddings) // 5)  # Don't have too many small clusters

        best_score = -1
        best_k = 10  # Default fallback

        for k in range(min_k, max_k + 1):
            try:
                clusterer = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = clusterer.fit_predict(embeddings)
                score = silhouette_score(embeddings, labels, metric='cosine', sample_size=min(1000, len(embeddings)))

                if score > best_score:
                    best_score = score
                    best_k = k
            except:
                continue

        return best_k

    def cluster_feedbacks(
        self,
        df: pd.DataFrame,
        n_clusters: int = None,
        method: str = 'kmeans'
    ) -> Tuple[pd.DataFrame, np.ndarray, int]:
        """
        Cluster feedbacks based on embeddings to identify common themes

        Args:
            df: DataFrame with embedding column
            n_clusters: Number of clusters (if None, will find optimal)
            method: 'kmeans' or 'dbscan'

        Returns:
            Tuple of (DataFrame with cluster labels, cluster info, number of clusters used)
        """
        embeddings = np.array(df['embedding'].tolist())

        if method == 'kmeans':
            # Find optimal if not specified
            if n_clusters is None:
                n_clusters = self.find_optimal_clusters(embeddings)

            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings)
            cluster_info = clusterer.cluster_centers_
        else:  # dbscan
            clusterer = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
            labels = clusterer.fit_predict(embeddings)
            cluster_info = labels
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        df_clustered = df.copy()
        df_clustered['cluster'] = labels

        self.cluster_cache = {
            'labels': labels,
            'method': method,
            'cluster_info': cluster_info,
            'n_clusters': n_clusters
        }

        return df_clustered, cluster_info, n_clusters

    def get_cluster_statistics(self, df_clustered: pd.DataFrame) -> pd.DataFrame:
        """
        Get statistics for each cluster

        Args:
            df_clustered: DataFrame with cluster labels

        Returns:
            DataFrame with cluster statistics
        """
        cluster_stats = df_clustered.groupby('cluster').agg({
            'code_id': 'count',
            'author_id': lambda x: len(set(x)),
            'feedback': lambda x: list(x)[:5],  # Top 5 samples
            'code': lambda x: list(x)[:3]  # Top 3 code samples
        }).reset_index()

        cluster_stats.columns = [
            'cluster_id',
            'feedback_count',
            'unique_authors',
            'sample_feedbacks',
            'sample_codes'
        ]

        cluster_stats = cluster_stats.sort_values('feedback_count', ascending=False)

        return cluster_stats

    def find_author_difficulties(
        self,
        df_clustered: pd.DataFrame,
        author_id: str
    ) -> Dict:
        """
        Analyze specific author's difficulty patterns

        Args:
            df_clustered: DataFrame with cluster labels
            author_id: Author ID to analyze

        Returns:
            Dictionary with author difficulty analysis
        """
        author_data = df_clustered[df_clustered['author_id'] == author_id]

        if len(author_data) == 0:
            return {'error': 'Author not found'}

        # Cluster distribution
        cluster_dist = author_data['cluster'].value_counts().to_dict()

        # Most common issues (based on cluster membership)
        primary_clusters = author_data['cluster'].value_counts().head(3).index.tolist()

        return {
            'author_id': author_id,
            'total_submissions': len(author_data),
            'cluster_distribution': cluster_dist,
            'primary_difficulty_clusters': primary_clusters,
            'feedbacks': author_data['feedback'].tolist(),
            'codes': author_data['code'].tolist()
        }

    def identify_common_topics(
        self,
        df_clustered: pd.DataFrame,
        top_n: int = 10,
        use_llm: bool = True
    ) -> List[Dict]:
        """
        Identify most common topics/difficulties across all students

        Args:
            df_clustered: DataFrame with cluster labels
            top_n: Number of top topics to return
            use_llm: Use LLM-based topic extraction (better quality)

        Returns:
            List of topic dictionaries
        """
        cluster_stats = self.get_cluster_statistics(df_clustered)

        topics = []
        for _, row in cluster_stats.head(top_n).iterrows():
            if use_llm:
                # Use LLM to extract meaningful concepts
                concepts = self._extract_concepts_with_llm(row['sample_feedbacks'])
            else:
                # Fallback to basic keyword extraction
                concepts = self._extract_keywords_basic(row['sample_feedbacks'])

            topics.append({
                'cluster_id': int(row['cluster_id']),
                'student_count': int(row['unique_authors']),
                'occurrence_count': int(row['feedback_count']),
                'key_concepts': concepts,
                'sample_feedbacks': row['sample_feedbacks'][:3],
                'sample_codes': row['sample_codes'][:2]
            })

        return topics

    def _extract_concepts_with_llm(self, feedbacks: List[str]) -> List[str]:
        """
        Extract programming concepts using zero-shot classification

        Args:
            feedbacks: List of feedback texts

        Returns:
            List of key programming concepts
        """
        from transformers import pipeline

        # Combine top feedbacks
        combined_text = ' '.join(feedbacks[:5])[:1000]  # Limit length

        # Candidate programming concepts
        candidate_labels = [
            "edge cases",
            "loop conditions",
            "variable initialization",
            "return values",
            "pointer management",
            "memory allocation",
            "array indexing",
            "conditional logic",
            "recursion",
            "null handling",
            "boundary conditions",
            "type conversion",
            "function parameters",
            "error handling",
            "algorithm efficiency",
            "data structure usage",
            "integer overflow",
            "off-by-one errors",
            "scope issues",
            "logic errors"
        ]

        try:
            # Use zero-shot classification
            classifier = pipeline("zero-shot-classification",
                                model="facebook/bart-large-mnli",
                                device=-1)  # CPU

            result = classifier(combined_text, candidate_labels, multi_label=True)

            # Get top 5 concepts with score > 0.3
            concepts = [
                label for label, score in zip(result['labels'], result['scores'])
                if score > 0.3
            ][:5]

            return concepts if concepts else ["general programming"]

        except Exception as e:
            print(f"LLM extraction failed: {e}, falling back to basic")
            return self._extract_keywords_basic(feedbacks)

    def _extract_keywords_basic(self, feedbacks: List[str]) -> List[str]:
        """
        Basic keyword extraction (fallback)

        Args:
            feedbacks: List of feedback texts

        Returns:
            List of keywords
        """
        all_feedbacks = ' '.join(feedbacks)
        words = all_feedbacks.lower().split()

        # Extended stop words - more comprehensive
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
            'can', 'could', 'may', 'might', 'your', 'you', 'this', 'that', 'these',
            'those', 'when', 'where', 'why', 'how', 'which', 'who', 'what',
            'after', 'before', 'during', 'while', 'about', 'into', 'through',
            'their', 'there', 'them', 'they', 'then', 'than', 'such', 'some',
            'it', 'its', 'if', 'else', 'not', 'all', 'any', 'each', 'every'
        }

        # Programming-specific keywords to keep
        prog_keywords = {
            'loop', 'variable', 'function', 'return', 'pointer', 'array',
            'condition', 'edge', 'case', 'null', 'memory', 'allocation',
            'index', 'recursion', 'parameter', 'overflow', 'boundary'
        }

        keywords = []
        for w in words:
            w_clean = w.strip('.,;:!?()')
            if (w_clean in prog_keywords) or (w_clean not in stop_words and len(w_clean) > 4):
                keywords.append(w_clean)

        keyword_counts = Counter(keywords).most_common(5)
        return [kw for kw, _ in keyword_counts]

    def visualize_clusters_2d(
        self,
        df_clustered: pd.DataFrame,
        sample_size: int = 1000
    ) -> go.Figure:
        """
        Create 2D visualization of feedback clusters using t-SNE

        Args:
            df_clustered: DataFrame with embeddings and cluster labels
            sample_size: Max number of points to visualize (for performance)

        Returns:
            Plotly figure
        """
        # Sample data if too large
        if len(df_clustered) > sample_size:
            df_sample = df_clustered.sample(n=sample_size, random_state=42)
        else:
            df_sample = df_clustered

        # Extract embeddings
        embeddings = np.array(df_sample['embedding'].tolist())

        # Reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Create visualization DataFrame with full feedback
        viz_df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'cluster': df_sample['cluster'].astype(str),
            'feedback': df_sample['feedback'].tolist(),  # Full feedback
            'author_id': df_sample['author_id'].tolist(),  # Full author_id
            'code_id': df_sample['code_id'].tolist()
        })

        # Create scatter plot with custom hover template
        fig = go.Figure()

        # Add trace for each cluster
        for cluster_id in sorted(viz_df['cluster'].unique()):
            cluster_data = viz_df[viz_df['cluster'] == cluster_id]

            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                marker=dict(size=8, opacity=0.7),
                customdata=np.column_stack((
                    cluster_data['feedback'],
                    cluster_data['author_id'],
                    cluster_data['code_id']
                )),
                hovertemplate='<b>Cluster %{fullData.name}</b><br><br>' +
                             '<b>Feedback:</b><br>%{customdata[0]}<br><br>' +
                             '<b>Author ID:</b> %{customdata[1]}<br>' +
                             '<b>Code ID:</b> %{customdata[2]}<br>' +
                             '<extra></extra>'
            ))

        fig.update_layout(
            title='Feedback Clusters - 2D Projection (t-SNE)',
            xaxis_title='',  # Remove dimension labels
            yaxis_title='',
            xaxis=dict(showticklabels=False),  # Hide axis ticks
            yaxis=dict(showticklabels=False),
            height=600,
            hovermode='closest',
            legend_title='Cluster'
        )

        return fig

    def get_difficulty_heatmap(
        self,
        df_clustered: pd.DataFrame,
        top_n_authors: int = 20,
        top_n_clusters: int = 10
    ) -> go.Figure:
        """
        Create heatmap showing which authors struggle with which topics

        Args:
            df_clustered: DataFrame with cluster labels and author_id
            top_n_authors: Number of top authors to show
            top_n_clusters: Number of top clusters to show

        Returns:
            Plotly figure
        """
        # Get top authors by submission count
        top_authors = df_clustered['author_id'].value_counts().head(top_n_authors).index

        # Get top clusters by frequency
        top_clusters = df_clustered['cluster'].value_counts().head(top_n_clusters).index

        # Filter data
        df_filtered = df_clustered[
            (df_clustered['author_id'].isin(top_authors)) &
            (df_clustered['cluster'].isin(top_clusters))
        ]

        # Create pivot table
        heatmap_data = df_filtered.groupby(['author_id', 'cluster']).size().reset_index(name='count')
        heatmap_pivot = heatmap_data.pivot(index='author_id', columns='cluster', values='count').fillna(0)

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=[f'Cluster {c}' for c in heatmap_pivot.columns],
            y=[f'Author {a[:8]}...' for a in heatmap_pivot.index],
            colorscale='YlOrRd',
            text=heatmap_pivot.values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Submission Count")
        ))

        fig.update_layout(
            title='Student Difficulty Patterns - Author vs Topic Clusters',
            xaxis_title='Difficulty Cluster',
            yaxis_title='Student ID',
            height=600
        )

        return fig
