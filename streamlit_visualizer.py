"""
Streamlit Visualizer Module
============================
Provides real-time concept map visualization using Streamlit and NetworkX.

Uses st.empty() placeholders for live updates during dynamic concept map generation.
"""

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List
import logging

# Use non-interactive backend for Matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


class StreamlitConceptMapVisualizer:
    """
    Real-time concept map visualizer using Streamlit.
    """
    
    def __init__(self, topic_name: str, educational_level: str):
        """
        Initialize the visualizer.
        
        Args:
            topic_name: Name of the topic
            educational_level: Educational level
        """
        self.topic_name = topic_name
        self.educational_level = educational_level
        self.graph = nx.DiGraph()
        self.node_colors = {}
        self.concept_types_colors = {
            "process": "#3498db",  # Blue
            "entity": "#2ecc71",   # Green
            "property": "#f39c12", # Orange
            "event": "#e74c3c",    # Red
            "concept": "#9b59b6",  # Purple
            "default": "#95a5a6"   # Gray
        }
        
    def add_concepts(self, concepts: List[Dict]):
        """
        Add concepts (nodes) to the graph.
        
        Args:
            concepts: List of concept dicts with 'name' and 'type' keys
        """
        for concept in concepts:
            concept_name = concept.get('name', '')
            concept_type = concept.get('type', 'concept').lower()
            
            if concept_name and concept_name not in self.graph.nodes:
                self.graph.add_node(concept_name)
                # Assign color based on type
                color = self.concept_types_colors.get(
                    concept_type,
                    self.concept_types_colors["default"]
                )
                self.node_colors[concept_name] = color
                logger.debug(f"Added concept: {concept_name} (type: {concept_type})")
    
    def add_relationships(self, relationships: List[Dict]):
        """
        Add relationships (edges) to the graph.
        
        Args:
            relationships: List of relationship dicts with 'from', 'to', 'relationship' keys
        """
        for rel in relationships:
            from_node = rel.get('from', '')
            to_node = rel.get('to', '')
            relationship_type = rel.get('relationship', 'related to')
            
            # Only add edge if both nodes exist
            if from_node in self.graph.nodes and to_node in self.graph.nodes:
                self.graph.add_edge(
                    from_node,
                    to_node,
                    relationship=relationship_type
                )
                logger.debug(f"Added relationship: {from_node} -> {to_node} ({relationship_type})")
    
    def render_graph(self) -> plt.Figure:
        """
        Render the current graph state as a matplotlib figure.
        
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(14, 10))
        
        if not self.graph.nodes:
            # Empty graph - show placeholder
            ax.text(
                0.5, 0.5,
                "Waiting for concepts...",
                ha='center', va='center',
                fontsize=16, color='gray'
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            return fig
        
        # Use hierarchical layout if possible, otherwise spring layout
        try:
            pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        except:
            pos = nx.spring_layout(self.graph, seed=42)
        
        # Get node colors in order
        node_colors_list = [
            self.node_colors.get(node, self.concept_types_colors["default"])
            for node in self.graph.nodes
        ]
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors_list,
            node_size=3000,
            alpha=0.9,
            ax=ax
        )
        
        # Draw node labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            font_size=10,
            font_weight='bold',
            font_color='white',
            ax=ax
        )
        
        # Draw edges
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edge_color='#34495e',
            width=2,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            ax=ax,
            connectionstyle='arc3,rad=0.1'
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
            font_size=8,
            font_color='#2C3E50',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7),
            ax=ax
        )
        
        ax.set_title(
            f"Concept Map: {self.topic_name}\n[{self.educational_level} Level]",
            fontsize=16,
            fontweight='bold',
            pad=20
        )
        ax.axis('off')
        plt.tight_layout()
        
        return fig


def initialize_streamlit_page(topic_name: str, educational_level: str):
    """
    Initialize Streamlit page configuration and layout.
    
    Args:
        topic_name: Name of the topic
        educational_level: Educational level
    """
    st.set_page_config(
        page_title=f"Dynamic Concept Map: {topic_name}",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title(f"🧠 Dynamic Concept Map: {topic_name}")
    st.markdown(f"**Educational Level:** {educational_level}")
    st.markdown("---")


def run_dynamic_visualization(timeline: Dict):
    """
    Run the dynamic visualization loop synchronized with TTS.
    
    This is the main entry point for Streamlit-based dynamic visualization.
    
    Args:
        timeline: Timeline dict from timeline_mapper.create_timeline()
    """
    import time
    from tts_handler import TTSHandler
    
    # Extract metadata
    metadata = timeline["metadata"]
    topic_name = metadata["topic_name"]
    educational_level = metadata["educational_level"]
    
    # Initialize Streamlit page
    initialize_streamlit_page(topic_name, educational_level)
    
    # Initialize visualizer
    visualizer = StreamlitConceptMapVisualizer(topic_name, educational_level)
    
    # Initialize TTS handler
    tts = TTSHandler(rate=190, volume=0.9)
    
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    current_sentence_placeholder = st.empty()
    graph_placeholder = st.empty()
    progress_placeholder = st.empty()
    
    # Show initial empty graph
    with graph_placeholder:
        fig = visualizer.render_graph()
        st.pyplot(fig)
        plt.close(fig)
    
    logger.info("🎤 Starting dynamic visualization with TTS...")
    
    # Main loop: Process each sentence
    total_sentences = len(timeline["sentences"])
    
    for sentence_data in timeline["sentences"]:
        sentence_idx = sentence_data["index"]
        sentence_text = sentence_data["text"]
        concepts = sentence_data["concepts"]
        relationships = sentence_data["relationships"]
        
        # Update progress
        progress = (sentence_idx + 1) / total_sentences
        progress_placeholder.progress(progress, text=f"Processing sentence {sentence_idx + 1}/{total_sentences}")
        
        # Show current sentence
        current_sentence_placeholder.info(f"🎙️ **Speaking:** \"{sentence_text}\"")
        status_placeholder.warning("⏳ Listening to narration...")
        
        # Speak sentence and wait for it to FULLY complete
        logger.info(f"🎙️ Speaking sentence {sentence_idx}: \"{sentence_text}\"")
        start_time = time.time()
        
        # Use blocking TTS call
        tts.speak_sentence(sentence_text)
        
        # Calculate actual duration
        actual_duration = time.time() - start_time
        
        # Add extra buffer to ensure TTS is fully complete
        # Estimate minimum time based on word count (0.4s per word at 150 wpm)
        word_count = len(sentence_text.split())
        estimated_duration = word_count * 0.4
        
        # If actual duration is suspiciously short, wait for estimated duration
        if actual_duration < estimated_duration * 0.5:
            wait_time = estimated_duration - actual_duration
            logger.warning(f"⚠️  TTS finished too quickly ({actual_duration:.1f}s), waiting additional {wait_time:.1f}s")
            time.sleep(wait_time)
            actual_duration = time.time() - start_time
        
        logger.info(f"✅ Finished speaking (took {actual_duration:.1f}s)")
        
        # Additional buffer to ensure speech is fully complete
        time.sleep(0.5)
        
        # Update status
        status_placeholder.success("✨ Revealing concepts...")
        
        # Add concepts and relationships to graph
        if concepts:
            visualizer.add_concepts(concepts)
            logger.info(f"   → Added concepts: {[c['name'] for c in concepts]}")
        
        if relationships:
            visualizer.add_relationships(relationships)
            logger.info(f"   → Added {len(relationships)} relationships")
        
        # Update visualization
        with graph_placeholder:
            fig = visualizer.render_graph()
            st.pyplot(fig)
            plt.close(fig)
        
        # Pause for visual absorption
        time.sleep(1.0)
    
    # Final status
    progress_placeholder.progress(1.0, text="✅ Complete!")
    status_placeholder.success(f"✅ Dynamic concept map complete! ({metadata['total_concepts']} concepts revealed)")
    current_sentence_placeholder.empty()
    
    logger.info("✅ Dynamic visualization complete!")
    
    # Show completion message
    st.markdown("---")
    st.success("🎉 **Concept map generation complete!** You can now close this browser tab.")
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
    
    # Add stop button (this won't actually stop Streamlit server, just a visual indicator)
    st.markdown("---")
    if st.button("🛑 I'm Done - Close This Tab", type="primary", use_container_width=True):
        st.balloons()
        st.success("✅ You can now close this browser tab and press Ctrl+C in the terminal!")
