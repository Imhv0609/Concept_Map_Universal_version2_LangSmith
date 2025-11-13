"""
Standalone Streamlit App for Dynamic Concept Maps
==================================================
Single-page app where you can input a description and see the dynamic concept map.
Version: 3.1 - Smart Grid + JSON Download Button Always Visible
Features: 3x3 grid, halved node count, max 2 incoming edges, prominent download button
Last Updated: 2025-11-13 - Added always-visible JSON download button
"""

import streamlit as st
import sys
import os
import tempfile
import time
import logging
from pathlib import Path
from dotenv import load_dotenv

# CRITICAL: Load environment variables (including GOOGLE_API_KEY)
load_dotenv()

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verify API key is loaded
if not os.getenv('GOOGLE_API_KEY'):
    st.error("⚠️ GOOGLE_API_KEY not found! Please create a .env file with your API key.")
    st.stop()

# Import required modules
from timeline_mapper import create_timeline
from precompute_engine import PrecomputeEngine
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import pygame

# Initialize pygame for audio (with fallback for headless environments)
AUDIO_AVAILABLE = False
try:
    # Try to initialize with dummy driver for headless environments
    os.environ['SDL_AUDIODRIVER'] = 'dummy'
    pygame.mixer.init()
    AUDIO_AVAILABLE = True
    logger.info("Audio system initialized successfully")
except Exception as e:
    logger.warning(f"Audio system not available: {e}. Audio playback will be disabled.")
    AUDIO_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Dynamic Concept Map Generator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .concept-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


def render_graph(G, pos, visible_nodes, new_nodes, alpha_map, scale_map, show_edge_labels=True):
    """
    Render the graph with animations and edge labels.
    
    Args:
        G: NetworkX graph
        pos: Node positions dict
        visible_nodes: Set of visible node names
        new_nodes: Set of newly added nodes
        alpha_map: Dict of node alpha values (for fade-in)
        scale_map: Dict of node scale values (for pop-in)
        show_edge_labels: Whether to show relationship labels on edges
    """
    fig, ax = plt.subplots(figsize=(16, 12), dpi=100)  # Larger canvas, better resolution
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    
    # Draw edges for visible nodes only
    visible_edges = [(u, v) for u, v in G.edges() 
                     if u in visible_nodes and v in visible_nodes]
    
    logger.debug(f"render_graph: {len(G.nodes())} total nodes, {len(G.edges())} total edges, {len(visible_nodes)} visible nodes, {len(visible_edges)} visible edges")
    
    if visible_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=visible_edges,
            edge_color='#5a6c7d',  # Darker gray for better visibility
            alpha=0.6,  # More opaque
            width=2.5,  # Thicker edges
            arrows=True,
            arrowsize=25,  # Larger arrows (increased from 15 to 25)
            arrowstyle='-|>',  # Filled arrow style for better visibility
            ax=ax,
            connectionstyle='arc3,rad=0.05',  # Straighter edges
            node_size=3000,  # Helps arrows appear at proper distance from nodes
            min_source_margin=20,  # Arrow starts away from source node
            min_target_margin=20   # Arrow ends away from target node
        )
    
    # Draw nodes with animations
    for node in visible_nodes:
        if node not in pos:
            continue
            
        x, y = pos[node]
        # NO ANIMATIONS - nodes and labels appear instantly at full opacity
        alpha = 1.0  # Always fully visible
        scale = 1.0  # Always full size
        
        # Node size - no animation
        base_size = 3000
        node_size = base_size
        
        # Node color - distinctive colors for new vs existing nodes
        if node in new_nodes:
            # Bright vibrant orange/gold for NEW nodes - very eye-catching
            color = (1.0, 0.6, 0.0, 1.0)  # Vibrant orange-gold, fully opaque
            edge_color = '#ff6b00'  # Bright orange border
            edge_width = 5  # Thicker border for new nodes
        else:
            # Blue color for EXISTING nodes
            color = (0.2, 0.5, 0.8, 1.0)  # Lighter blue, fully opaque
            edge_color = '#1f77b4'
            edge_width = 2
        
        # Draw node
        ax.scatter([x], [y], s=node_size, c=[color], 
                  edgecolors=edge_color, linewidth=edge_width, zorder=2)
        
        # Draw label INSIDE the node (centered)
        # Label is ALWAYS fully visible (alpha=1.0) even when node is fading in
        ax.text(x, y, node, fontsize=9, fontweight='bold',
               ha='center', va='center', color='white', alpha=1.0, zorder=4,
               bbox=dict(boxstyle='round,pad=0.3', facecolor=(0, 0, 0, 0.3), alpha=0.7, edgecolor='none'))
    
    # Draw edge labels (relationship names) if enabled
    if show_edge_labels and visible_edges:
        edge_labels = {}
        for u, v in visible_edges:
            # Get relationship from edge data
            if G.has_edge(u, v):
                edge_data = G.get_edge_data(u, v)
                rel_type = edge_data.get('relationship', 'related to')
                edge_labels[(u, v)] = rel_type
        
        if edge_labels:
            # Draw edge labels with minimal padding and background
            for (u, v), label in edge_labels.items():
                # Calculate position along the edge
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                
                # Position label at 1/3rd of the distance from source to avoid overlapping
                # This places the label closer to the source node
                label_x = x1 + (x2 - x1) / 3.0  # 1/3rd from source (x1)
                label_y = y1 + (y2 - y1) / 3.0  # 1/3rd from source (y1)
                
                # Add text with minimal background - larger font, tighter padding
                ax.text(
                    label_x, label_y, label,
                    fontsize=9,  # Larger font (was 7)
                    color='#4a5568',  # Slightly darker for better readability
                    ha='center',
                    va='center',
                    bbox=dict(
                        boxstyle='round,pad=0.15',  # Tighter padding (was 0.2)
                        facecolor='white',
                        alpha=0.9,  # More opaque (was 0.85)
                        edgecolor='none'
                    ),
                    zorder=3  # Above edges but below nodes
                )
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def animate_fade_in(graph_placeholder, G, pos, sentence_data, 
                    existing_nodes, show_edge_labels=True, 
                    animation_duration=0.8, steps=15):
    """
    Animate the appearance of new concepts with fade-in and scale effects.
    
    Args:
        graph_placeholder: Streamlit placeholder for graph
        G: NetworkX graph
        pos: Node positions
        sentence_data: Current sentence data with concepts
        existing_nodes: Set of nodes that were already visible
        show_edge_labels: Whether to show relationship labels on edges
        animation_duration: Duration of animation in seconds
        steps: Number of animation frames
    """
    # FIXED: Handle both dict and string formats
    new_concepts = []
    for c in sentence_data.get('concepts', []):
        if isinstance(c, dict):
            name = c.get('name', '')
        else:
            name = str(c)
        if name.strip():
            new_concepts.append(name)
    
    new_nodes = set(new_concepts) - existing_nodes
    
    if not new_nodes:
        return
    
    # Animation loop
    alpha_map = {node: 0.0 for node in new_nodes}
    scale_map = {node: 0.3 for node in new_nodes}
    
    for step in range(steps + 1):
        progress = step / steps
        
        # Update alpha (fade-in: 0 → 1.0)
        for node in new_nodes:
            alpha_map[node] = progress
            
        # Update scale (pop-in: 0.3 → 1.0)
        for node in new_nodes:
            scale_map[node] = 0.3 + (0.7 * progress)
        
        # Render graph with current animation state
        visible_nodes = existing_nodes | new_nodes
        fig = render_graph(G, pos, visible_nodes, new_nodes, alpha_map, scale_map, show_edge_labels)
        
        with graph_placeholder:
            st.pyplot(fig)
        
        plt.close(fig)
        
        # Sleep between frames
        if step < steps:
            time.sleep(animation_duration / steps)


def play_audio(audio_file, wait_for_audio=True):
    """
    Play audio file using Streamlit's audio player (works in cloud!)
    
    Args:
        audio_file: Path to audio file
        wait_for_audio: If True, wait for estimated audio duration before continuing
    
    Returns:
        Audio duration in seconds (or True/False for legacy compatibility)
    """
    try:
        if os.path.exists(audio_file):
            # Use Streamlit's native audio player - works in cloud!
            st.audio(audio_file, format='audio/mp3', start_time=0)
            
            # Get audio duration for timing
            duration = 0
            try:
                import mutagen
                from mutagen.mp3 import MP3
                audio = MP3(audio_file)
                duration = audio.info.length
                logger.info(f"Playing audio: {duration:.2f}s")
            except:
                # Fallback: estimate duration from file size
                file_size = os.path.getsize(audio_file)
                # Rough estimate: ~1KB per 0.1 seconds for MP3
                duration = max(2.0, file_size / 10000)
                logger.info(f"Playing audio file: {audio_file} (estimated {duration:.1f}s)")
            
            # Wait for audio to "play" (give user time to hear it)
            if wait_for_audio and duration > 0:
                time.sleep(duration)
            
            return duration
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
    
    return False


def reveal_concepts_progressively(
    graph_placeholder, G, pos, concepts, elapsed_time, 
    visible_nodes, show_edge_labels=True, animation_duration=0.5
):
    """
    Reveal concepts that should be visible at current elapsed time.
    
    Args:
        graph_placeholder: Streamlit placeholder for graph
        G: NetworkX graph
        pos: Node positions
        concepts: List of all concepts with reveal_time
        elapsed_time: Current elapsed time since audio started
        visible_nodes: Set of already visible nodes
        show_edge_labels: Whether to show relationship labels
        animation_duration: Duration of fade-in animation (seconds)
        
    Returns:
        Updated set of visible nodes
    """
    # Find concepts that should be revealed now
    newly_revealed = []
    for concept in concepts:
        concept_name = concept.get('name', '')
        reveal_time = concept.get('reveal_time', 0.0)
        
        if concept_name and reveal_time <= elapsed_time and concept_name not in visible_nodes:
            newly_revealed.append(concept_name)
            logger.debug(f"     → Revealing '{concept_name}' at {elapsed_time:.2f}s (reveal_time: {reveal_time:.2f}s)")
    
    if not newly_revealed:
        # No new concepts to reveal, but still render current state
        if visible_nodes:
            logger.debug(f"     → No new concepts, rendering {len(visible_nodes)} existing nodes")
            fig = render_graph(G, pos, visible_nodes, set(), {}, {}, show_edge_labels)
            with graph_placeholder:
                st.pyplot(fig)
            plt.close(fig)
        return visible_nodes
    
    # Animate new concepts with fade-in
    new_nodes_set = set(newly_revealed)
    alpha_map = {node: 0.0 for node in new_nodes_set}
    scale_map = {node: 0.3 for node in new_nodes_set}
    
    steps = max(5, int(animation_duration * 10))  # 10 fps
    
    for step in range(steps + 1):
        progress = step / steps
        
        # Update alpha and scale
        for node in new_nodes_set:
            alpha_map[node] = progress
            scale_map[node] = 0.3 + (0.7 * progress)
        
        # Render graph with new nodes for animation
        current_visible = visible_nodes | new_nodes_set
        
        # Render graph (it will calculate visible edges internally)
        fig = render_graph(G, pos, current_visible, new_nodes_set, alpha_map, scale_map, show_edge_labels)
        
        with graph_placeholder:
            st.pyplot(fig)
        
        plt.close(fig)
        
        # Small delay between frames
        if step < steps:
            time.sleep(animation_duration / steps)
    
    # Add newly revealed nodes to visible set
    return visible_nodes | new_nodes_set


def run_dynamic_visualization(timeline, layout_style="hierarchical", show_edge_labels=True):
    """
    Run the dynamic visualization with continuous audio and keyword-timed reveals.
    
    Args:
        timeline: Timeline data structure (continuous format)
        layout_style: Layout algorithm to use
        show_edge_labels: Whether to show relationship labels on edges
    """
    st.markdown("---")
    st.markdown("### 🎬 Dynamic Concept Map (Keyword-Timed)")
    
    # Always-visible Download Button
    import json
    timeline_json = json.dumps(timeline, indent=2)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.download_button(
            label="📥 Download Timeline JSON with Node Timings",
            data=timeline_json,
            file_name=f"timeline_{timeline['metadata'].get('topic', 'concept_map')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    # Debug: Show timeline structure
    with st.expander("🔍 Debug Info (Click to expand)", expanded=False):
        st.write(f"**Format:** Continuous Timeline (single audio file)")
        st.write(f"**Total Words:** {timeline['metadata'].get('word_count', 0)}")
        st.write(f"**Total Duration:** {timeline['metadata'].get('total_duration', 0):.1f}s")
        st.write(f"**Total Concepts:** {timeline['metadata'].get('total_concepts', 0)}")
        st.write(f"**Total Relationships:** {len(timeline.get('relationships', []))}")
        
        # Show Node Timings Table
        st.write("### 📊 Node Timings")
        timing_data = []
        for sentence in timeline.get("sentences", []):
            for concept in sentence.get("concepts", []):
                timing_data.append({
                    "Node": concept.get("name", ""),
                    "Reveal Time (s)": f"{concept.get('reveal_time', 0):.2f}",
                    "Importance": concept.get("importance", 0)
                })
        if timing_data:
            import pandas as pd
            df = pd.DataFrame(timing_data)
            st.dataframe(df, use_container_width=True)
        
        # Download Complete Timeline as JSON
        st.write("### 💾 Download Timeline")
        import json
        timeline_json = json.dumps(timeline, indent=2)
        st.download_button(
            label="📥 Download Complete Timeline JSON",
            data=timeline_json,
            file_name=f"timeline_{timeline['metadata'].get('topic', 'concept_map')}.json",
            mime="application/json"
        )
        
        # Check first sentence structure
        st.write("### 🔍 First Sentence Structure")
        if timeline["sentences"]:
            first_sent = timeline["sentences"][0]
            st.write(f"**First Sentence Concepts:** {len(first_sent.get('concepts', []))}")
            st.json(first_sent)
    
    # Create graph
    G = nx.DiGraph()
    
    # Get concepts and relationships from timeline (continuous format)
    all_concepts = set()
    concepts = timeline.get("concepts", [])
    
    # Extract concept names
    for concept in concepts:
        if isinstance(concept, dict):
            concept_name = concept.get("name", "")
        else:
            concept_name = str(concept)
        
        if concept_name.strip():
            all_concepts.add(concept_name)
    
    logger.info(f"📊 Building graph with {len(all_concepts)} concepts")
    
    # Add nodes to graph
    for concept_name in all_concepts:
        G.add_node(concept_name)
    
    # Add edges from relationships (top-level relationships array)
    relationships = timeline.get("relationships", [])
    logger.info(f"🔗 Processing {len(relationships)} relationships")
    
    edges_added = 0
    for rel in relationships:
        if isinstance(rel, dict):
            from_node = rel.get("from", "")
            to_node = rel.get("to", "")
            rel_type = rel.get("relationship", "related to")
            
            if from_node in all_concepts and to_node in all_concepts:
                G.add_edge(from_node, to_node, relationship=rel_type)
                edges_added += 1
                logger.debug(f"  ➜ Added edge: {from_node} --[{rel_type}]--> {to_node}")
    
    logger.info(f"✅ Graph built: {len(G.nodes())} nodes, {len(G.edges())} edges (added {edges_added})")
    
    # Get pre-computed layout from timeline (preferred) or calculate fallback
    pos = timeline.get("pre_calculated_layout", timeline.get("layout", {}))
    
    logger.info(f"📐 Layout info: pre_calculated_layout exists={bool(pos)}, positions={len(pos) if pos else 0}")
    if pos:
        logger.info(f"   Sample positions: {list(pos.items())[:3]}")
    
    # If no layout provided, calculate one using selected style (fallback only)
    if not pos or len(pos) == 0:
        logger.warning("⚠️ No pre-calculated layout found! Calculating fallback layout...")
        if len(G.nodes()) > 0:
            try:
                if layout_style == "hierarchical":
                    # Use SAME implementation as PrecomputeEngine for consistency
                    from precompute_engine import PrecomputeEngine
                    engine = PrecomputeEngine()
                    pos = engine._create_hierarchical_tree_layout(G)
                    logger.info(f"✅ Created hierarchical layout using PrecomputeEngine (consistent)")
                elif layout_style == "shell":
                    pos = nx.shell_layout(G)
                elif layout_style == "circular":
                    pos = nx.circular_layout(G)
                elif layout_style == "kamada-kawai":
                    pos = nx.kamada_kawai_layout(G)
                else:  # spring
                    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            except Exception as e:
                logger.error(f"Layout calculation failed: {e}")
                # Fallback: simple grid layout
                nodes = list(G.nodes())
                import math
                cols = math.ceil(math.sqrt(len(nodes)))
                pos = {}
                for i, node in enumerate(nodes):
                    row = i // cols
                    col = i % cols
                    pos[node] = (col, -row)
        else:
            pos = {}
    
    # Show warning if no concepts found
    if len(all_concepts) == 0:
        st.warning("⚠️ No concepts extracted! Check your description or try an example.")
        return
    
    # Create containers
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("#### 📊 Concept Map")
        st.caption(f"Total concepts to display: {len(all_concepts)}")
        graph_placeholder = st.empty()
        
        # Initial empty graph
        if len(all_concepts) > 0:
            fig = render_graph(G, pos, set(), set(), {}, {}, show_edge_labels)
            graph_placeholder.pyplot(fig)
            plt.close(fig)
        else:
            graph_placeholder.warning("Waiting for concepts...")
    
    with col2:
        st.markdown("#### 📝 Progress")
        progress_placeholder = st.empty()
        timer_placeholder = st.empty()
        concepts_placeholder = st.empty()
        
        # Audio section - will be populated when visualization starts
        st.markdown("---")
        st.markdown("#### 🔊 Audio Narration")
        audio_placeholder = st.empty()
        audio_control_info = st.empty()
    
    # Button-triggered visualization with state management
    st.markdown("---")
    
    # Get audio file and concepts
    audio_file = timeline.get("audio_file")
    concepts = timeline.get("concepts", [])
    total_duration = timeline.get("actual_audio_duration", 
                                  timeline["metadata"].get("total_duration", 0.0))
    
    if audio_file and os.path.exists(audio_file):
        # Initialize session state for visualization control
        if 'viz_started' not in st.session_state:
            st.session_state.viz_started = False
        if 'viz_completed' not in st.session_state:
            st.session_state.viz_completed = False
        
        # DEBUG: Show current state
        st.write(f"🔍 DEBUG: viz_started = {st.session_state.viz_started}, viz_completed = {st.session_state.viz_completed}")
        
        # Show instructions and button if not started
        if not st.session_state.viz_started:
            st.info("🎧 **Instructions:** Click the button below to start visualization with audio narration")
            
            # Show audio duration info
            with audio_control_info:
                st.info(f"📊 **Ready:** {total_duration:.1f}s duration | {len(concepts)} concepts")
            
            start_button = st.button("🚀 Start Visualization with Audio", type="primary", use_container_width=True, key="start_viz_btn")
            
            if start_button:
                logger.info("🔴 BUTTON CLICKED! Setting viz_started=True")
                st.session_state.viz_started = True
                st.write("✅ Button clicked! State updated. Script will continue...")
                # No st.rerun() needed - button click automatically triggers rerun!
        
        # Run visualization if started but not completed
        elif st.session_state.viz_started and not st.session_state.viz_completed:
            logger.info("🟢 ENTERING VISUALIZATION BLOCK")
            st.write("🟢 Visualization block entered!")
            
            # NOW show the audio player (won't be recreated on subsequent updates)
            with audio_placeholder:
                st.audio(audio_file, format='audio/mp3', autoplay=True)
            
            with audio_control_info:
                st.success(f"🎧 **Playing:** {total_duration:.1f}s | {len(concepts)} concepts")
            
            st.info("⏳ **Playing...** Watch as concepts appear synchronized with the narration!")
            
            # Debug logging with timing information
            logger.info(f"🚀 VISUALIZATION STARTED")
            logger.info(f"   Total concepts: {len(concepts)}")
            logger.info(f"   Total duration: {total_duration:.2f}s")
            logger.info(f"   Graph has {len(G.nodes())} nodes, {len(pos)} positions")
            
            # Log concept timings for debugging
            if timeline.get("metadata", {}).get("timing_scale_factor"):
                scale_factor = timeline["metadata"]["timing_scale_factor"]
                orig_duration = timeline["metadata"].get("original_estimated_duration", 0)
                logger.info(f"   ⚖️ Timings rescaled: {orig_duration:.2f}s → {total_duration:.2f}s (factor: {scale_factor:.3f})")
            
            logger.info("   📍 Concept reveal schedule:")
            for i, concept in enumerate(concepts[:10]):  # Show first 10
                reveal_time = concept.get('reveal_time', 0.0)
                name = concept.get('name', 'Unknown')
                logger.info(f"      {i+1}. '{name}' at {reveal_time:.2f}s")
            if len(concepts) > 10:
                logger.info(f"      ... and {len(concepts) - 10} more concepts")
            
            # Progressive reveal over duration
            visible_nodes = set()
            recently_revealed = {}  # Track when each node was revealed {node: reveal_time}
            highlight_duration = 1.5  # Keep nodes orange for 1.5 seconds after reveal
            
            # Calculate frames per second for smooth animation (targeting 10 FPS)
            fps = 10  # frames per second
            frame_duration = 1.0 / fps  # seconds per frame
            total_frames = int(total_duration * fps)  # total number of frames
            
            logger.info(f"   Will reveal over {total_frames} frames at {fps} FPS ({frame_duration:.3f}s per frame)")
            
            # Start timing for real-time synchronization
            start_time = time.time()
            
            for frame in range(total_frames + 1):
                # Calculate elapsed time based on actual clock time (not frame count)
                # This ensures we stay synchronized with audio even if rendering is slow
                elapsed = time.time() - start_time
                
                # Safety check: don't exceed total duration
                if elapsed > total_duration:
                    elapsed = total_duration
                
                # Update progress
                with progress_placeholder:
                    progress = elapsed / total_duration if total_duration > 0 else 1.0
                    st.progress(progress, text=f"Progress: {int(progress * 100)}%")
                
                # Update timer
                with timer_placeholder:
                    st.metric("⏱️ Elapsed Time", f"{elapsed:.1f}s / {total_duration:.1f}s")
                
                # Reveal concepts that should be visible now
                prev_count = len(visible_nodes)
                for concept in concepts:
                    concept_name = concept.get('name', '')
                    reveal_time = concept.get('reveal_time', 0.0)
                    if concept_name and reveal_time <= elapsed and concept_name not in visible_nodes:
                        visible_nodes.add(concept_name)
                        recently_revealed[concept_name] = elapsed  # Track when revealed
                        logger.info(f"   ✨ Revealing '{concept_name}' at {elapsed:.2f}s")
                
                # Determine which nodes are still "new" (revealed within last 3 seconds)
                new_nodes = {node for node, reveal_time in recently_revealed.items() 
                            if elapsed - reveal_time < highlight_duration}
                
                # Render graph with currently visible nodes and highlight new ones
                if len(visible_nodes) > 0:
                    fig = render_graph(G, pos, visible_nodes, new_nodes, {}, {}, show_edge_labels)
                    with graph_placeholder:
                        st.pyplot(fig)
                    plt.close(fig)
                    logger.debug(f"   📊 Rendered graph with {len(visible_nodes)} nodes ({len(new_nodes)} highlighted)")
                
                # Update concepts counter
                with concepts_placeholder:
                    if len(visible_nodes) > 0:
                        st.success(f"💡 **Revealed:** {len(visible_nodes)}/{len(concepts)} concepts")
                    else:
                        st.info(f"💡 **Waiting for first concept...**")
                
                # Sleep until next frame time (to maintain consistent FPS)
                next_frame_time = start_time + (frame + 1) * frame_duration
                sleep_time = next_frame_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                # Stop if we've reached or exceeded the total duration
                if elapsed >= total_duration:
                    logger.info(f"   ⏹️ Reached total duration: {elapsed:.2f}s")
                    break
            
            # Mark as completed
            st.session_state.viz_completed = True
            
            # Final view
            with progress_placeholder:
                st.success("✅ Complete!")
            
            with timer_placeholder:
                st.balloons()
                st.success(f"🎉 **All {len(visible_nodes)} concepts revealed!**")
            
            with concepts_placeholder:
                st.info(f"📊 **Concepts:** {', '.join(sorted(visible_nodes))}")
            
            logger.info(f"✅ VISUALIZATION COMPLETED")
        
        # Show completed state
        else:
            st.success("✅ **Visualization completed!** Generate a new concept map to see another animation.")
            with progress_placeholder:
                st.success("✅ Complete!")
            with timer_placeholder:
                st.success(f"🎉 **All concepts revealed!**")
        
    else:
        st.error("❌ Audio file not found. Cannot start visualization.")
        logger.error(f"Audio file missing: {audio_file}")


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Dynamic Concept Map Generator</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Enter a description and watch concepts come alive!</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        educational_level = st.selectbox(
            "Educational Level",
            ["elementary", "middle school", "high school", "college", "graduate"],
            index=2
        )
        
        topic_name = st.text_input(
            "Topic Name (optional)",
            placeholder="Auto-detected if empty"
        )
        
        st.markdown("---")
        st.markdown("### 🗺️ Layout Options")
        
        layout_style = st.selectbox(
            "Graph Layout",
            ["hierarchical", "shell", "circular", "kamada-kawai", "spring"],
            index=0,
            help="Choose how concepts are arranged in the graph"
        )
        
        show_edge_labels = st.checkbox(
            "Show Relationship Labels",
            value=True,
            help="Display relationship names on edges"
        )
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Enter your description (4-12 sentences work best)
        2. Click **Generate Concept Map**
        3. Watch as concepts appear dynamically!
        
        **Tips:**
        - More sentences = better animations
        - Use clear, educational language
        - Include key terms and relationships
        """)
        
        st.markdown("---")
        st.markdown("### 🎨 Features")
        st.markdown("""
        ✅ Fade-in animations  
        ✅ Natural voice narration  
        ✅ Hierarchical layout  
        ✅ Real-time concept reveal  
        ✅ Progress tracking  
        """)
    
    # Main content area
    description = st.text_area(
        "Enter your description:",
        height=200,
        placeholder="Example: Photosynthesis converts light energy into chemical energy.Chlorophyll molecules absorb sunlight in plant cells.Water molecules split to release oxygen.The Calvin cycle uses carbon dioxide.Glucose is produced as the final product.",
        help="Enter 4-12 sentences for best results. Spaces after periods are optional - we handle that!"
    )
    
    # Generate button
    if st.button("🚀 Generate Concept Map", type="primary"):
        if not description.strip():
            st.error("⚠️ Please enter a description first!")
            return
        
        # Clear previous session state to ensure fresh generation
        if 'timeline' in st.session_state:
            del st.session_state.timeline
        if 'viz_started' in st.session_state:
            del st.session_state.viz_started
        if 'viz_completed' in st.session_state:
            del st.session_state.viz_completed
        logger.info("🧹 Cleared previous session state for fresh generation")
        
        # Show loading
        with st.spinner("🔄 Processing..."):
            try:
                # Step 1: Create timeline
                with st.status("📋 Creating timeline...", expanded=True) as status:
                    st.write("🔥 Analyzing description with AI...")
                    timeline = create_timeline(
                        description,
                        educational_level,
                        topic_name if topic_name.strip() else None
                    )
                    
                    # Validate timeline
                    num_sentences = len(timeline.get('sentences', []))
                    total_concepts = timeline.get('metadata', {}).get('total_concepts', 0)
                    
                    st.write(f"✅ Found {num_sentences} sentences")
                    st.write(f"✅ Extracted {total_concepts} concepts")
                    
                    # Warning if no concepts
                    if total_concepts == 0:
                        st.warning("⚠️ No concepts extracted! This might affect visualization.")
                    
                    # Show sample concepts
                    if num_sentences > 0 and timeline['sentences'][0].get('concepts'):
                        sample_concepts = [c.get('name', str(c)) if isinstance(c, dict) else str(c) 
                                         for c in timeline['sentences'][0]['concepts'][:3]]
                        if sample_concepts:
                            st.write(f"📝 Sample concepts: {', '.join(sample_concepts)}")
                    
                    status.update(label="✅ Timeline created!", state="complete")
                
                # Step 2: Pre-compute assets with selected layout
                with st.status("🎨 Generating audio and layout...", expanded=True) as status:
                    st.write("🎤 Generating natural voice narration...")
                    st.write(f"📐 Using '{layout_style}' layout algorithm...")
                    engine = PrecomputeEngine(layout_style=layout_style)
                    timeline = engine.precompute_all(timeline)
                    st.write(f"✅ Generated {len(timeline['sentences'])} audio files")
                    st.write(f"✅ Calculated {layout_style} graph layout")
                    status.update(label="✅ Assets ready!", state="complete")
                
                # Step 3: Store timeline in session state for persistence
                st.session_state.timeline = timeline
                st.session_state.layout_style = layout_style
                st.session_state.show_edge_labels = show_edge_labels
                st.session_state.engine = engine
                
                # Step 4: Run visualization with selected options
                run_dynamic_visualization(timeline, layout_style, show_edge_labels)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.exception("Error during generation")
    
    # If timeline exists in session state, continue showing visualization
    elif 'timeline' in st.session_state and st.session_state.timeline:
        timeline = st.session_state.timeline
        layout_style = st.session_state.get('layout_style', 'hierarchical')
        show_edge_labels = st.session_state.get('show_edge_labels', True)
        run_dynamic_visualization(timeline, layout_style, show_edge_labels)
    
    # Example descriptions
    with st.expander("📚 Example Descriptions (Click to use)"):
        st.markdown("**Click any example to use it:**")
        
        examples = [
            {
                "title": "🌿 Photosynthesis (5 sentences)",
                "text": "Photosynthesis converts light energy into chemical energy.Chlorophyll molecules absorb sunlight in plant cells.Water molecules split to release oxygen.The Calvin cycle uses carbon dioxide.Glucose is produced as the final product."
            },
            {
                "title": "💧 Water Cycle (7 sentences)",
                "text": "The water cycle moves water across Earth's surface.Water evaporates from oceans due to solar energy.Water vapor rises and forms clouds through condensation.Precipitation falls as rain or snow.Surface runoff carries water to rivers.Groundwater infiltrates into soil and rocks.Transpiration releases water from plants."
            },
            {
                "title": "🌍 Climate Change (8 sentences)",
                "text": "Climate change affects global temperatures.The greenhouse effect traps heat naturally.Carbon dioxide levels have increased dramatically.Fossil fuels release greenhouse gases.Polar ice caps are melting rapidly.Sea levels are rising worldwide.Extreme weather events occur more frequently.Renewable energy can reduce emissions."
            },
            {
                "title": "⚛️ Newton's Laws (6 sentences)",
                "text": "Newton's laws describe motion and forces.The first law states objects resist changes.The second law defines force as mass times acceleration.The third law describes action-reaction pairs.Momentum is conserved in collisions.These principles form classical mechanics."
            }
        ]
        
        for example in examples:
            if st.button(example["title"], key=example["title"]):
                st.session_state.description = example["text"]
                st.rerun()
    
    # Auto-fill from session state if available
    if "description" in st.session_state and st.session_state.description:
        st.text_area(
            "Auto-filled description:",
            value=st.session_state.description,
            height=150,
            disabled=True
        )
        if st.button("Use this description"):
            description = st.session_state.description


if __name__ == "__main__":
    main()
