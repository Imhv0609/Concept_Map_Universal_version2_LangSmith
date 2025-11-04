#!/bin/bash

# Launch Standalone Streamlit App
# ================================
# Single-page app where you input description and see dynamic concept map

echo "🧠 Launching Dynamic Concept Map Generator..."
echo ""
echo "✨ Features:"
echo "   - Enter description in the app"
echo "   - Watch concepts appear dynamically"
echo "   - Natural voice narration"
echo "   - Smooth animations"
echo ""
echo "🌐 Opening browser at http://localhost:8501"
echo ""
echo "⚠️  Keep this terminal open while using the app"
echo "🛑 Press Ctrl+C to exit"
echo ""

cd "$(dirname "$0")"
streamlit run streamlit_app_standalone.py
