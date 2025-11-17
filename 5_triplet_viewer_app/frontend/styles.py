"""
Custom CSS styling for the triplet viewer interface
"""

import streamlit as st


def apply_custom_css():
    """Apply custom CSS styling to the Streamlit app"""
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
