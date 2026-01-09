#!/bin/bash
# Quick launcher for RAG Feedback System

cd "$(dirname "$0")"

echo "🚀 Starting RAG Feedback System..."
echo ""
echo "📊 Pages available:"
echo "   - Main: http://localhost:8501"
echo "   - Stats: http://localhost:8501/stats"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run app.py
