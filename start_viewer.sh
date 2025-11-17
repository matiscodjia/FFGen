#!/bin/bash
# FFGen Triplet Viewer - Launcher

echo "=========================================="
echo "  FFGen Triplet Viewer"
echo "=========================================="
echo ""
echo "Starting Streamlit app..."
echo ""

STREAMLIT_SERVER_HEADLESS=true .venv/bin/streamlit run triplet_viewer.py --server.port 8501

echo ""
echo "App terminated."
