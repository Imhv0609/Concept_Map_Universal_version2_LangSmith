"""
Enhanced Streamlit Visualizer with Pre-computation & Animations
================================================================
Uses pre-generated audio and layouts for smooth, lag-free visualization.
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from typing import Dict, List, Tuple
import logging
import time
import os
import pygame

# Use non-interactive backend for Matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class EnhancedStreamlitVisualizer:
    """
    Enhanced visualizer with smooth animations and pre-computed assets.
    """
    
    def __init__(self, topic_name: str, educational_level: str, layout: Dict[str, Tuple[float, float]]):
        """
        Initialize enhanced visualizer.
        
        Args:
            topic_name: Topic name
            educational_level: Educational level
            layout: Pre-calculated node positions
        """
        self.topic_name = topic_name
        self.educational_level = educational_level
        self.graph = nx.DiGraph()
        self.layout = layout
        self.node_colors = {}
        self.node_alphas = {}  # For fade-in animations
        self.node_scales = {}  # For scale animation (pop-in effect)
        self.newly_added_nodes = set()  # Track new nodes for highlighting
        
        self.concept_types_colors = {
            "process": "#3498db",
            "entity": "#2ecc71",
            "property": "#f39c12",
            "event": "#e74c3c",
            "concept": "#9b59b6",
            "default": "#95a5a6"
        }
        
        # Initialize pygame mixer for audio playback
        try:
            pygame.mixer.init()
            logger.info("🎵 Pygame mixer initialized for audio playback")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize pygame mixer: {e}")
    
    def add_concepts(self, concepts: List[Dict], animate: bool = True):
        """
        Add concepts with optional animation support.
        
        Args:
            concepts: List of concept dicts
            animate: Whether to mark as newly added for animation
        """
        for concept in concepts:
            concept_name = concept.get('name', '')
            concept_type = concept.get('type', 'concept').lower()
            
            if concept_name and concept_name not in self.graph.nodes:
                self.graph.add_node(concept_name, **concept)
                
                # Assign color
                color = self.concept_types_colors.get(
                    concept_type,
                    self.concept_types_colors["default"]
                )
                self.node_colors[concept_name] = color
                
                # Set initial alpha and scale for animation
                if animate:
                    self.node_alphas[concept_name] = 0.0  # Start transparent
                    self.node_scales[concept_name] = 0.3  # Start small (30% size)
                    self.newly_added_nodes.add(concept_name)
                else:
                    self.node_alphas[concept_name] = 1.0
                    self.node_scales[concept_name] = 1.0
                
                logger.debug(f"Added concept: {concept_name} (type: {concept_type})")
    
    def add_relationships(self, relationships: List[Dict]):
        """
        Add relationships (edges) to the graph.
        
        Args:
            relationships: List of relationship dicts
        """
        for rel in relationships:
            from_node = rel.get('from', '')
            to_node = rel.get('to', '')
            relationship_type = rel.get('relationship', 'related to')
            
            if from_node in self.graph.nodes and to_node in self.graph.nodes:
                self.graph.add_edge(
                    from_node,
                    to_node,
                    relationship=relationship_type
                )
                logger.debug(f"Added edge: {from_node} -> {to_node}")
    
    def animate_fade_in(self, graph_placeholder, duration: float = 0.5, steps: int = 10):
        """
        Animate newly added nodes fading in with scale effect and real-time rendering.
        
        Args:
            graph_placeholder: Streamlit placeholder for graph updates
            duration: Total animation duration in seconds
            steps: Number of animation steps
        """
        if not self.newly_added_nodes:
            return
        
        step_duration = duration / steps
        alpha_increment = 1.0 / steps
        
        # Scale goes from 0.3 to 1.0, so increment is 0.7 / steps
        scale_increment = 0.7 / steps
        
        for step in range(steps):
            # Update alpha and scale values for dramatic effect
            for node in self.newly_added_nodes:
                # Fade in (0.0 -> 1.0)
                current_alpha = self.node_alphas.get(node, 0.0)
                self.node_alphas[node] = min(current_alpha + alpha_increment, 1.0)
                
                # Scale up (0.3 -> 1.0) for "pop-in" effect
                current_scale = self.node_scales.get(node, 0.3)
                self.node_scales[node] = min(current_scale + scale_increment, 1.0)
            
            # Render and display updated graph
            with graph_placeholder:
                fig = self.render_graph()
                st.pyplot(fig)
                plt.close(fig)
            
            time.sleep(step_duration)
        
        # Mark nodes as fully visible and full size
        for node in self.newly_added_nodes:
            self.node_alphas[node] = 1.0
            self.node_scales[node] = 1.0
        
        self.newly_added_nodes.clear()
    
    def render_graph(self) -> plt.Figure:
        """
        Render the graph with pre-calculated layout and alpha blending.
        
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(16, 12))
        
        if not self.graph.nodes:
            ax.text(0.5, 0.5, "Waiting for concepts...", 
                   ha='center', va='center', fontsize=16, color='gray')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Use pre-calculated layout
        pos = {node: self.layout.get(node, (0, 0)) for node in self.graph.nodes}
        
        # Prepare node colors with alpha and edge colors for highlighting
        node_colors_with_alpha = []
        edge_colors = []
        edge_widths = []
        
        for node in self.graph.nodes:
            base_color = self.node_colors.get(node, self.concept_types_colors["default"])
            alpha = self.node_alphas.get(node, 1.0)
            
            # Convert hex to RGBA
            import matplotlib.colors as mcolors
            rgb = mcolors.hex2color(base_color)
            rgba = (*rgb, alpha)
            node_colors_with_alpha.append(rgba)
            
            # Highlight newly added nodes with yellow glow
            if node in self.newly_added_nodes and alpha < 1.0:
                edge_colors.append('#FFD700')  # Gold color for new nodes
                edge_widths.append(4)
            else:
                edge_colors.append('white')
                edge_widths.append(2)
        
        # Draw nodes with alpha blending, dynamic edge colors, and scale animation
        base_node_size = 3500
        for i, node in enumerate(self.graph.nodes):
            # Get scale factor for this node (for pop-in effect)
            scale = self.node_scales.get(node, 1.0)
            animated_size = base_node_size * scale
            
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=[node],
                node_color=[node_colors_with_alpha[i]],
                node_size=animated_size,  # Dynamic size based on animation
                alpha=1.0,  # Alpha already in color
                ax=ax,
                edgecolors=edge_colors[i],
                linewidths=edge_widths[i]
            )
        
        # Draw node labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=11,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color='#34495e',
            width=2.5,
            alpha=0.7,
            arrows=True,
            arrowsize=25,
            arrowstyle='->',
            ax=ax,
            connectionstyle='arc3,rad=0.1',
            node_size=3500
        )
        
        # Draw edge labels
        edge_labels = {}
        for u, v, data in self.graph.edges(data=True):
            rel_type = data.get('relationship', 'related to')
            edge_labels[(u, v)] = rel_type
        
        nx.draw_networkx_edge_labels(
            self.graph,
            pos,
            edge_labels=edge_labels,
            font_size=9,
            font_color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='#bdc3c7'),
            ax=ax
        )
        
        ax.set_title(
            f"🧠 {self.topic_name}\n[{self.educational_level} Level]",
            fontsize=18,
            fontweight='bold',
            pad=25,
            color='#2c3e50'
        )
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def play_audio(self, audio_file: str) -> float:
        """
        Play pre-generated audio file and return its duration.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            Duration of audio playback in seconds
        """
        if not audio_file or not os.path.exists(audio_file):
            logger.warning(f"⚠️  Audio file not found: {audio_file}")
            return 0.0
        
        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            start_time = time.time()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
            
            duration = time.time() - start_time
            logger.info(f"🎵 Played audio: {os.path.basename(audio_file)} ({duration:.1f}s)")
            return duration
            
        except Exception as e:
            logger.error(f"❌ Error playing audio: {e}")
            return 0.0


def run_enhanced_visualization(timeline: Dict):
    """
    Run enhanced visualization with pre-computed assets and smooth animations.
    
    Args:
        timeline: Timeline with pre-computed audio files and layout
    """
    metadata = timeline["metadata"]
    topic_name = metadata["topic_name"]
    educational_level = metadata["educational_level"]
    pre_calculated_layout = timeline.get("pre_calculated_layout", {})
    
    # Initialize page
    st.set_page_config(
        page_title=f"Dynamic Concept Map: {topic_name}",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title(f"🧠 Dynamic Concept Map: {topic_name}")
    st.markdown(f"**Educational Level:** {educational_level}")
    st.markdown("---")
    
    # Initialize visualizer with pre-calculated layout
    visualizer = EnhancedStreamlitVisualizer(topic_name, educational_level, pre_calculated_layout)
    
    # Create placeholders
    status_placeholder = st.empty()
    current_sentence_placeholder = st.empty()
    graph_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    # Show initial empty graph
    with graph_placeholder:
        fig = visualizer.render_graph()
        st.pyplot(fig)
        plt.close(fig)
    
    logger.info("🎬 Starting enhanced visualization...")
    
    # Main loop
    total_sentences = len(timeline["sentences"])
    
    for sentence_data in timeline["sentences"]:
        sentence_idx = sentence_data["index"]
        sentence_text = sentence_data["text"]
        concepts = sentence_data["concepts"]
        relationships = sentence_data["relationships"]
        audio_file = sentence_data.get("audio_file")
        
        # Update progress
        progress = (sentence_idx + 1) / total_sentences
        progress_placeholder.progress(progress, text=f"Sentence {sentence_idx + 1}/{total_sentences}")
        
        # Show current sentence
        current_sentence_placeholder.info(f"🎙️ **Speaking:** \"{sentence_text}\"")
        status_placeholder.warning("⏳ Playing narration...")
        
        # Play pre-generated audio
        logger.info(f"🎵 Playing sentence {sentence_idx}: \"{sentence_text[:50]}...\"")
        audio_duration = visualizer.play_audio(audio_file)
        
        # Add concepts and relationships
        if concepts:
            status_placeholder.warning("✨ Revealing new concepts...")
            visualizer.add_concepts(concepts, animate=True)
            logger.info(f"   → Added concepts: {[c['name'] for c in concepts]}")
        
        if relationships:
            visualizer.add_relationships(relationships)
            logger.info(f"   → Added {len(relationships)} relationships")
        
        # Animate fade-in with real-time rendering (snappier: 0.8s)
        if concepts:
            status_placeholder.info("🎬 Animating concept reveal...")
            visualizer.animate_fade_in(graph_placeholder, duration=0.8, steps=15)
            status_placeholder.success("✅ Concepts revealed!")
        
        # Final render to ensure everything is shown
        with graph_placeholder:
            fig = visualizer.render_graph()
            st.pyplot(fig)
            plt.close(fig)
        
        # Brief pause for absorption (reduced from 1.0s)
        time.sleep(0.5)
    
    # Final status
    progress_placeholder.progress(1.0, text="✅ Complete!")
    status_placeholder.success(f"✅ Concept map complete! ({metadata['total_concepts']} concepts)")
    current_sentence_placeholder.empty()
    
    logger.info("✅ Enhanced visualization complete!")
    
    # Show completion message
    st.markdown("---")
    st.success("🎉 **Concept map generation complete!** You can now close this tab.")
    st.info("💡 **Tip:** Press **Ctrl+C** in the terminal to stop the server and exit.")
    
    # Show legend
    st.markdown("---")
    st.markdown("### 📊 Concept Types")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🔵 **Process** - Actions or procedures")
        st.markdown("🟢 **Entity** - Objects or things")
    with col2:
        st.markdown("🟠 **Property** - Characteristics or attributes")
        st.markdown("🔴 **Event** - Occurrences or happenings")
    with col3:
        st.markdown("🟣 **Concept** - Abstract ideas")
        st.markdown("⚪ **Other** - General concepts")
    
    # Done button
    st.markdown("---")
    if st.button("🛑 I'm Done - Close This Tab", type="primary", use_container_width=True):
        st.balloons()
        st.success("✅ You can now close this tab and press Ctrl+C in the terminal!")
