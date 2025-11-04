"""
Standalone Streamlit App for Dynamic Concept Maps
==================================================
Single-page app where you can input a description and see the dynamic concept map.
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
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor('#ffffff')
    fig.patch.set_facecolor('#ffffff')
    
    # Draw edges for visible nodes only
    visible_edges = [(u, v) for u, v in G.edges() 
                     if u in visible_nodes and v in visible_nodes]
    
    if visible_edges:
        nx.draw_networkx_edges(
            G, pos,
            edgelist=visible_edges,
            edge_color='#34495e',
            alpha=0.7,
            width=2.5,
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            ax=ax,
            connectionstyle='arc3,rad=0.1'
        )
    
    # Draw nodes with animations
    for node in visible_nodes:
        if node not in pos:
            continue
            
        x, y = pos[node]
        alpha = alpha_map.get(node, 1.0)
        scale = scale_map.get(node, 1.0)
        
        # Node size with scale animation
        base_size = 3000
        node_size = base_size * scale
        
        # Node color with alpha
        if node in new_nodes:
            # Gold color for new nodes
            color = (1.0, 0.84, 0.0, alpha)  # Gold with alpha
            edge_color = 'gold'
            edge_width = 4
        else:
            # Blue color for existing nodes
            color = (0.12, 0.47, 0.71, alpha)  # Blue with alpha
            edge_color = '#1f77b4'
            edge_width = 2
        
        # Draw node
        ax.scatter([x], [y], s=node_size, c=[color], 
                  edgecolors=edge_color, linewidth=edge_width, zorder=2)
        
        # Draw label with alpha
        ax.text(x, y, node, fontsize=10, fontweight='bold',
               ha='center', va='center', color='white', alpha=alpha, zorder=3)
    
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
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=9,
                font_color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'),
                ax=ax
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


def play_audio(audio_file):
    """Play audio file using pygame"""
    if not AUDIO_AVAILABLE:
        logger.info("Audio playback skipped - no audio device available")
        return False
        
    try:
        if os.path.exists(audio_file):
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for audio to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            return True
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
    
    return False


def run_dynamic_visualization(timeline, layout_style="hierarchical", show_edge_labels=True):
    """
    Run the dynamic visualization with animations and audio.
    
    Args:
        timeline: Timeline data structure
        layout_style: Layout algorithm to use
        show_edge_labels: Whether to show relationship labels on edges
    """
    st.markdown("---")
    st.markdown("### 🎬 Dynamic Concept Map")
    
    # Debug: Show timeline structure
    with st.expander("🔍 Debug Info (Click to expand)", expanded=False):
        st.write(f"**Total Sentences:** {len(timeline['sentences'])}")
        st.write(f"**Total Concepts in Metadata:** {timeline['metadata'].get('total_concepts', 0)}")
        
        # Check first sentence structure
        if timeline["sentences"]:
            first_sent = timeline["sentences"][0]
            st.write(f"**First Sentence Concepts:** {len(first_sent.get('concepts', []))}")
            st.json(first_sent)
    
    # Create graph
    G = nx.DiGraph()
    
    # Create layout - IMPROVED: Handle both dict and list formats
    all_concepts = set()
    for sent in timeline["sentences"]:
        concepts = sent.get("concepts", [])
        
        # Handle both list of dicts and list of strings
        for concept in concepts:
            if isinstance(concept, dict):
                concept_name = concept.get("name", "")
            else:
                concept_name = str(concept)
            
            if concept_name.strip():
                all_concepts.add(concept_name)
    
    # Add nodes to graph
    for concept in all_concepts:
        G.add_node(concept)
    
    # Add edges from relationships
    for sent in timeline["sentences"]:
        relationships = sent.get("relationships", [])
        for rel in relationships:
            if isinstance(rel, dict):
                from_node = rel.get("from", "")
                to_node = rel.get("to", "")
                if from_node in all_concepts and to_node in all_concepts:
                    G.add_edge(from_node, to_node)
    
    # Get pre-computed layout from timeline (preferred) or calculate fallback
    pos = timeline.get("pre_calculated_layout", timeline.get("layout", {}))
    
    # If no layout provided, calculate one using selected style (fallback only)
    if not pos or len(pos) == 0:
        if len(G.nodes()) > 0:
            try:
                if layout_style == "hierarchical":
                    # Try to create hierarchical layout
                    try:
                        from networkx.drawing.nx_agraph import graphviz_layout
                        pos = graphviz_layout(G, prog='dot')
                    except:
                        # Fallback to spring layout if graphviz not available
                        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
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
        
        # Initial empty graph (or show message if no concepts)
        if len(all_concepts) > 0:
            fig = render_graph(G, pos, set(), set(), {}, {}, show_edge_labels)
            graph_placeholder.pyplot(fig)
            plt.close(fig)
        else:
            graph_placeholder.warning("Waiting for concepts...")
    
    with col2:
        st.markdown("#### 📝 Narration Progress")
        progress_placeholder = st.empty()
        sentence_placeholder = st.empty()
        concepts_placeholder = st.empty()
    
    # Animation state
    visible_nodes = set()
    
    # Iterate through sentences
    for idx, sentence_data in enumerate(timeline["sentences"]):
        # Update progress
        with progress_placeholder:
            progress = (idx + 1) / len(timeline["sentences"])
            st.progress(progress, text=f"Sentence {idx + 1}/{len(timeline['sentences'])}")
        
        # Show current sentence
        with sentence_placeholder:
            st.info(f"🗣️ **{sentence_data['text']}**")
        
        # Show concepts being revealed - FIXED: Handle both dict and string formats
        concepts = sentence_data.get('concepts', [])
        concept_names = []
        for c in concepts:
            if isinstance(c, dict):
                name = c.get('name', '')
            else:
                name = str(c)
            if name.strip():
                concept_names.append(name)
        
        with concepts_placeholder:
            if concept_names:
                st.success(f"💡 **Concepts:** {', '.join(concept_names)}")
            else:
                st.info("💡 **Concepts:** (None in this sentence)")
        
        # Play audio if available
        audio_file = sentence_data.get('audio_file')
        if audio_file and os.path.exists(audio_file):
            play_audio(audio_file)
        else:
            # Fallback: estimate duration
            time.sleep(sentence_data.get('estimated_tts_duration', 2.0))
        
        # Animate new concepts
        animate_fade_in(graph_placeholder, G, pos, sentence_data, 
                       visible_nodes, show_edge_labels, 
                       animation_duration=0.8, steps=15)
        
        # Update visible nodes - FIXED: Handle both dict and string formats
        for concept in sentence_data.get('concepts', []):
            if isinstance(concept, dict):
                name = concept.get('name', '')
            else:
                name = str(concept)
            if name.strip():
                visible_nodes.add(name)
        
        # Pause before next sentence
        time.sleep(0.5)
    
    # Final view
    with progress_placeholder:
        st.success("✅ Complete!")
    
    with sentence_placeholder:
        st.balloons()
        st.success(f"🎉 **Concept map complete!** ({len(visible_nodes)} concepts)")
    
    with concepts_placeholder:
        st.info(f"📊 **All Concepts:** {', '.join(sorted(visible_nodes))}")


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
                
                # Step 3: Run visualization with selected options
                run_dynamic_visualization(timeline, layout_style, show_edge_labels)
                
                # Cleanup
                engine.cleanup()
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                logger.exception("Error during generation")
    
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
