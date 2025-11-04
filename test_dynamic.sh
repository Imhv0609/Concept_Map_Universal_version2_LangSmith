#!/bin/bash

# Test script for dynamic concept map generation
# Dynamic mode is now DEFAULT, so no --dynamic flag needed!

echo "========================================================================"
echo "🧪 Testing Dynamic Concept Map Generation (DEFAULT MODE)"
echo "========================================================================"
echo ""
echo "This test will:"
echo "  1. Make ONE API call to extract all concepts"
echo "  2. Launch Streamlit web interface"
echo "  3. Play TTS narration sentence-by-sentence"
echo "  4. Update the concept map in real-time"
echo ""
echo "⚠️  Instructions:"
echo "  • Wait for the URL to appear (http://localhost:8501)"
echo "  • Open the URL in your browser"
echo "  • Watch the concept map build dynamically!"
echo "  • Press Ctrl+C in this terminal when done"
echo ""
echo "========================================================================"
echo ""

# Run with default dynamic mode (no --dynamic flag needed)
python3 main_universal.py \
  --description "Photosynthesis converts light into energy. Plants use chlorophyll to absorb sunlight." \
  --level "high school" \
  --topic "Photosynthesis"
