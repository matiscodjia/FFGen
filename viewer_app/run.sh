#!/bin/bash
# Quick launcher for FFGen Triplet Viewer

cd "$(dirname "$0")"
echo "🔬 Launching FFGen Triplet Viewer..."
echo "📂 Working directory: $(pwd)"
echo ""

# Check if virtual environment exists
if [ ! -d "../.venv" ]; then
    echo "❌ Virtual environment not found. Please run 'make install' first."
    exit 1
fi

# Launch streamlit
../.venv/bin/streamlit run app.py

# Alternative: use uv if preferred
# uv run streamlit run app.py
