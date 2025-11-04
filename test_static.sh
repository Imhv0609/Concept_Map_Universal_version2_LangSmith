#!/bin/bash

# Test script for STATIC concept map generation
# Use --static flag to get JSON + PNG output (no Streamlit)

echo "========================================================================"
echo "🧪 Testing Static Concept Map Generation"
echo "========================================================================"
echo ""
echo "This test will:"
echo "  1. Make ONE API call to extract all concepts"
echo "  2. Generate JSON output file"
echo "  3. Generate PNG visualization"
echo "  4. Save both to output/ directory"
echo ""
echo "NOTE: This is the OLD behavior (no Streamlit, no dynamic reveal)"
echo ""
echo "========================================================================"
echo ""

# Run with --static flag to use original mode
python3 main_universal.py \
  --description "Water evaporates and forms clouds. Rain falls back to Earth." \
  --level "elementary" \
  --topic "Water Cycle" \
  --static

echo ""
echo "========================================================================"
echo "✅ Static mode complete! Check output/ directory for files:"
echo "   - JSON file with concept data"
echo "   - PNG image with concept map visualization"
echo "========================================================================"
