
import json
import os
import sys
import tempfile

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the enhanced visualizer
from streamlit_visualizer_enhanced import run_enhanced_visualization

# Load timeline from temp file
timeline_file = os.path.join(tempfile.gettempdir(), "concept_map_timeline_concurrent.json")

try:
    with open(timeline_file, 'r') as f:
        timeline = json.load(f)
except FileNotFoundError:
    import streamlit as st
    st.error("❌ Timeline file not found. Please run the main script again.")
    st.stop()
except json.JSONDecodeError as e:
    import streamlit as st
    st.error(f"❌ Failed to parse timeline file: {e}")
    st.stop()

# Run the enhanced visualization
run_enhanced_visualization(timeline)
