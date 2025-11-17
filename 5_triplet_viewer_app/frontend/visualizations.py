"""
Visualization utilities for embedding space and similarity analysis
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from umap.umap_ import UMAP
from typing import Dict


def create_3d_visualization(embeddings_data: Dict):
    """
    Create 3D interactive visualization with connections using UMAP.

    Args:
        embeddings_data: Dict with 'embeddings', 'labels', 'types' keys

    Returns:
        Plotly Figure object
    """
    embeddings = embeddings_data['embeddings']
    labels = embeddings_data['labels']
    types = embeddings_data['types']

    # Reduce to 3D using UMAP
    reducer = UMAP(n_components=3, random_state=42, n_neighbors=min(15, len(embeddings)-1))
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
    """
    Create 2D visualization for download using UMAP.

    Args:
        embeddings_data: Dict with 'embeddings', 'labels', 'types' keys

    Returns:
        Plotly Figure object
    """
    embeddings = embeddings_data['embeddings']
    labels = embeddings_data['labels']
    types = embeddings_data['types']

    # Reduce to 2D using UMAP
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings)-1))
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


def create_global_3d_visualization(embeddings_data: Dict):
    """
    Create 3D visualization with N triplets and their connections using UMAP.

    Args:
        embeddings_data: Dict with 'embeddings', 'labels', 'types', 'connections' keys

    Returns:
        Plotly Figure object
    """
    embeddings = embeddings_data['embeddings']
    labels = embeddings_data['labels']
    types = embeddings_data['types']
    connections = embeddings_data['connections']

    # Reduce to 3D using UMAP
    n_neighbors = min(15, len(embeddings) - 1)
    if n_neighbors < 2:
        n_neighbors = 2

    reducer = UMAP(n_components=3, random_state=42, n_neighbors=n_neighbors)
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
