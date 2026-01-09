"""
Statistics Dashboard
Displays metrics for the cache system
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from stats_logger import StatsLogger

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Cache Statistics",
    layout="wide"
)

st.title("Cache Performance Statistics")

# ==========================================
# LOAD DATA
# ==========================================

logger = StatsLogger()

# Load data
stats = logger.read_stats()
summary = logger.get_summary_stats()
cache_misses = logger.read_cache_misses()

if not stats:
    st.warning("No data yet. Submit some queries first!")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(stats)

# Convert timestamp to datetime
if 'timestamp' in df.columns:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')

# ==========================================
# KPI METRICS
# ==========================================

st.header("Key Performance Indicators")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Queries",
        f"{summary['total_queries']:,}",
        help="Total number of queries submitted"
    )

with col2:
    st.metric(
        "Cache Hit Rate",
        f"{summary['hit_rate']:.1f}%",
        delta=f"{summary['total_hits']} hits" if summary['total_hits'] > 0 else None,
        help="Percentage of queries resolved via cache"
    )

with col3:
    st.metric(
        "Avg Confidence",
        f"{summary['avg_confidence']:.2f}",
        help="Average confidence score for cache hits"
    )

with col4:
    st.metric(
        "DeepSeek Tokens",
        f"{summary['total_deepseek_tokens']:,}",
        delta=f"{summary['total_misses']} calls",
        delta_color="inverse",
        help="Total tokens consumed via DeepSeek API"
    )

st.divider()

# ==========================================
# TIME SERIES
# ==========================================

st.header("Query Timeline")

col1, col2 = st.columns(2)

with col1:
    # Hit/Miss over time
    fig = px.scatter(
        df,
        x='timestamp',
        y='confidence',
        color='status',
        size='response_time_ms',
        color_discrete_map={'hit': '#10b981', 'miss': '#ef4444'},
        title="Cache Hit/Miss Over Time",
        labels={
            'timestamp': 'Time',
            'confidence': 'Confidence Score',
            'status': 'Status',
            'response_time_ms': 'Response Time (ms)'
        }
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Response time distribution
    fig = px.box(
        df,
        x='status',
        y='response_time_ms',
        color='status',
        color_discrete_map={'hit': '#10b981', 'miss': '#ef4444'},
        title="Response Time Distribution",
        labels={'response_time_ms': 'Response Time (ms)', 'status': 'Cache Status'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# SIMILARITY SCORES
# ==========================================

st.header("Similarity Analysis")

col1, col2 = st.columns(2)

with col1:
    # Similarity distribution
    if 'similarity_score' in df.columns:
        fig = px.histogram(
            df,
            x='similarity_score',
            color='status',
            nbins=30,
            title="Similarity Score Distribution",
            labels={'similarity_score': 'Similarity Score (lower = more similar)'},
            color_discrete_map={'hit': '#10b981', 'miss': '#ef4444'}
        )
        fig.add_vline(x=0.3, line_dash="dash", line_color="orange",
                     annotation_text="Threshold (0.3)")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Confidence vs Similarity
    hits_df = df[df['status'] == 'hit']
    if not hits_df.empty and 'similarity_score' in hits_df.columns:
        fig = px.scatter(
            hits_df,
            x='similarity_score',
            y='confidence',
            size='response_time_ms',
            title="Confidence vs Similarity (Hits Only)",
            labels={
                'similarity_score': 'Similarity Score',
                'confidence': 'Confidence',
                'response_time_ms': 'Response Time (ms)'
            },
            color='confidence',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# ERROR CATEGORIES
# ==========================================

st.header("Error Categories Analysis")

col1, col2 = st.columns(2)

with col1:
    # Top error categories
    if 'error_category' in df.columns:
        error_counts = df['error_category'].value_counts().head(10)
        fig = px.bar(
            x=error_counts.values,
            y=error_counts.index,
            orientation='h',
            title="Top 10 Error Categories",
            labels={'x': 'Count', 'y': 'Error Category'},
            color=error_counts.values,
            color_continuous_scale='blues'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with col2:
    # Difficulty distribution
    if 'difficulty' in df.columns:
        diff_counts = df['difficulty'].value_counts()
        fig = px.pie(
            values=diff_counts.values,
            names=diff_counts.index,
            title="Difficulty Distribution",
            color_discrete_sequence=px.colors.sequential.RdBu
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ==========================================
# CACHE MISSES LOG
# ==========================================

st.header("Recent Cache Misses")

if cache_misses:
    st.info(f"{len(cache_misses)} cache misses logged (ready for retraining)")

    # Display the last 5
    recent_misses = cache_misses[-5:]

    for i, miss in enumerate(reversed(recent_misses), 1):
        with st.expander(f"Miss #{len(cache_misses) - i + 1} - {miss.get('theme', 'N/A')} ({miss.get('error_category', 'N/A')})"):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("**Code:**")
                st.code(miss.get('code', 'N/A'), language='c')

            with col2:
                st.markdown("**Generated Feedback:**")
                st.write(miss.get('feedback', 'N/A'))

            st.markdown(f"**Tokens Used:** {miss.get('tokens_used', 0)}")
            st.markdown(f"**Timestamp:** {miss.get('timestamp', 'N/A')}")
else:
    st.success("No cache misses yet - all queries resolved from cache!")

# ==========================================
# EXPORT DATA
# ==========================================

st.divider()

st.header("Export Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("Download Stats CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download stats.csv",
            data=csv,
            file_name=f"cache_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

with col2:
    if cache_misses and st.button("Download Cache Misses JSONL"):
        import json
        jsonl_content = "\n".join(json.dumps(miss) for miss in cache_misses)
        st.download_button(
            label="Download cache_miss.jsonl",
            data=jsonl_content,
            file_name=f"cache_miss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            mime="application/jsonl"
        )