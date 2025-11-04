"""
Dynamic Orchestrator Module
============================
Coordinates the entire dynamic concept map generation workflow.

Orchestrates: Timeline creation → TTS narration → Streamlit visualization updates
"""

import logging
import subprocess
import sys
import os
from typing import Optional

logger = logging.getLogger(__name__)


def run_dynamic_mode(
    description: str,
    educational_level: str,
    topic_name: str
) -> bool:
    """
    Run the complete dynamic concept map generation workflow.
    
    Workflow:
    1. Create timeline (SINGLE LLM API call)
    2. Pre-compute all assets (audio + layout)
    3. Launch Streamlit app with enhanced timeline
    4. Streamlit app handles smooth playback + animations
    
    Args:
        description: Full description text
        educational_level: Educational level (e.g., "High School")
        topic_name: Topic name for the concept map
        
    Returns:
        True if successful, False otherwise
    """
    from timeline_mapper import create_timeline, print_timeline_summary
    from precompute_engine import PrecomputeEngine
    
    logger.info("=" * 70)
    logger.info("🚀 Starting Enhanced Dynamic Concept Map Generation")
    logger.info("=" * 70)
    
    # Step 1: Create timeline with SINGLE API call
    logger.info("📋 Step 1: Creating timeline (analyzing full description)...")
    try:
        timeline = create_timeline(description, educational_level, topic_name)
        print_timeline_summary(timeline)
    except Exception as e:
        logger.error(f"❌ Failed to create timeline: {e}")
        return False
    
    # Step 2: Pre-compute all assets (NEW!)
    logger.info("🎨 Step 2: Pre-computing assets (audio + layout)...")
    try:
        engine = PrecomputeEngine(voice="en-US-AriaNeural", rate="+0%")
        timeline = engine.precompute_all(timeline)
        logger.info("✅ Pre-computation complete!")
        logger.info(f"   → Generated {len(timeline['sentences'])} audio files")
        logger.info(f"   → Calculated hierarchical layout for {timeline['metadata']['total_concepts']} concepts")
    except Exception as e:
        logger.error(f"❌ Failed to pre-compute assets: {e}")
        logger.warning("⚠️  Falling back to legacy mode without pre-computation")
        # Continue without pre-computation (will use on-the-fly generation)
    
    # Step 3: Save timeline to temporary file for Streamlit to read
    import json
    import tempfile
    
    timeline_file = os.path.join(tempfile.gettempdir(), "concept_map_timeline.json")
    try:
        with open(timeline_file, 'w') as f:
            json.dump(timeline, f, indent=2)
        logger.info(f"💾 Timeline saved to: {timeline_file}")
    except Exception as e:
        logger.error(f"❌ Failed to save timeline: {e}")
        return False
    
    # Step 4: Create Streamlit runner script
    streamlit_script = _create_streamlit_runner_script()
    
    # Step 5: Print instructions and launch Streamlit
    print("\n" + "=" * 70)
    print("🌐 DYNAMIC CONCEPT MAP READY")
    print("=" * 70)
    print("\n📍 Streamlit server will start shortly...")
    print("\n🔗 Open this URL in your browser:")
    print("   http://localhost:8501")
    print("\n⚠️  IMPORTANT: Keep this terminal window open while viewing")
    print("\n🛑 TO EXIT AFTER VIEWING:")
    print("   1. Close the browser tab")
    print("   2. Press Ctrl+C in this terminal")
    print("\n" + "=" * 70 + "\n")
    
    # Launch Streamlit app
    logger.info("🎬 Launching Streamlit app...")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", streamlit_script, "--server.headless=true"],
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Streamlit failed to run: {e}")
        return False
    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("✅ DYNAMIC CONCEPT MAP SESSION ENDED")
        print("=" * 70)
        logger.info("✅ Dynamic concept map session ended by user")
        return True


def _create_streamlit_runner_script() -> str:
    """
    Create a temporary Python script that Streamlit will run.
    This script loads the timeline and calls the visualizer.
    
    Returns:
        Path to the temporary script file
    """
    import tempfile
    
    script_content = '''
import json
import os
import sys
import tempfile

# Add current directory to path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the enhanced visualizer with animations and pre-computed assets
from streamlit_visualizer_enhanced import run_enhanced_visualization

# Load timeline from temp file
timeline_file = os.path.join(tempfile.gettempdir(), "concept_map_timeline.json")

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

# Run the enhanced dynamic visualization with animations
run_enhanced_visualization(timeline)
'''
    
    # Create temporary file in the same directory as the main script
    # This ensures imports work correctly
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "_streamlit_runner_temp.py")
    
    try:
        with open(script_path, 'w') as f:
            f.write(script_content)
        logger.info(f"📝 Created Streamlit runner script: {script_path}")
        return script_path
    except Exception as e:
        logger.error(f"❌ Failed to create runner script: {e}")
        # Fallback to temp directory
        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False,
            dir=script_dir
        ) as f:
            f.write(script_content)
            return f.name


def cleanup_temp_files():
    """
    Clean up temporary files created during dynamic mode.
    Call this on exit/error.
    """
    import tempfile
    
    timeline_file = os.path.join(tempfile.gettempdir(), "concept_map_timeline.json")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runner_script = os.path.join(script_dir, "_streamlit_runner_temp.py")
    
    for filepath in [timeline_file, runner_script]:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"🗑️  Cleaned up: {filepath}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to clean up {filepath}: {e}")


# Register cleanup on exit
import atexit
atexit.register(cleanup_temp_files)
