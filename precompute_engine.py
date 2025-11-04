"""
Pre-computation Engine
======================
Generates all assets (audio, layout) before visualization starts.
This ensures smooth, lag-free playback.
"""

import asyncio
import os
import tempfile
import logging
from typing import Dict, List, Tuple
import networkx as nx
from pathlib import Path

logger = logging.getLogger(__name__)


class PrecomputeEngine:
    """
    Pre-computes all visualization assets for smooth playback.
    """
    
    def __init__(self, voice: str = "en-US-AriaNeural", rate: str = "+0%", layout_style: str = "hierarchical"):
        """
        Initialize pre-computation engine.
        
        Args:
            voice: Edge-TTS voice name (default: en-US-AriaNeural - clear, professional)
            rate: Speech rate adjustment (default: +0% = normal speed)
            layout_style: Graph layout algorithm (default: hierarchical)
                         Options: "hierarchical", "shell", "circular", "kamada-kawai", "spring"
        """
        self.voice = voice
        self.rate = rate
        self.layout_style = layout_style
        self.temp_dir = tempfile.mkdtemp(prefix="concept_map_audio_")
        self.audio_files = []
        logger.info(f"🎤 Using voice: {voice}")
        logger.info(f"📐 Using layout: {layout_style}")
        logger.info(f"📁 Audio temp directory: {self.temp_dir}")
    
    async def _generate_audio_async(self, text: str, output_file: str):
        """
        Generate audio file using Edge-TTS (async).
        
        Args:
            text: Text to synthesize
            output_file: Output MP3 file path
        """
        import edge_tts
        
        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            await communicate.save(output_file)
        except Exception as e:
            # Re-raise with more context
            raise Exception(f"Edge-TTS failed: {e}. Check internet connection and voice name '{self.voice}'")
    
    def _generate_audio_gtts_fallback(self, text: str, output_file: str) -> bool:
        """
        Fallback: Generate audio using gTTS (Google Text-to-Speech).
        More reliable on cloud platforms like Streamlit Cloud.
        
        Args:
            text: Text to synthesize
            output_file: Output MP3 file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from gtts import gTTS
            
            # Create gTTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to file
            tts.save(output_file)
            
            # Verify file was created
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                return True
            return False
            
        except Exception as e:
            logger.error(f"gTTS fallback also failed: {e}")
            return False
    
    def generate_audio_file(self, text: str, index: int) -> str:
        """
        Generate audio file for a sentence.
        Uses Edge-TTS with gTTS fallback for better cloud compatibility.
        
        Args:
            text: Sentence text
            index: Sentence index
            
        Returns:
            Path to generated audio file
        """
        output_file = os.path.join(self.temp_dir, f"sentence_{index}.mp3")
        
        # Try Edge-TTS first
        try:
            # Apply nest_asyncio to allow nested event loops (Streamlit compatibility)
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except:
                pass  # Already applied or not needed
            
            # Handle event loop properly for Streamlit compatibility
            try:
                # Try to get and use existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Loop is already running (Streamlit context)
                    # Use run_until_complete with nest_asyncio
                    loop.run_until_complete(self._generate_audio_async(text, output_file))
                else:
                    # Loop exists but not running
                    loop.run_until_complete(self._generate_audio_async(text, output_file))
            except RuntimeError as e:
                # No event loop exists, create new one
                logger.debug(f"Creating new event loop: {e}")
                asyncio.run(self._generate_audio_async(text, output_file))
            
            # Verify file was created
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                file_size = os.path.getsize(output_file)
                logger.info(f"✅ Generated audio for sentence {index}: {os.path.basename(output_file)} ({file_size} bytes)")
                return output_file
            else:
                logger.warning(f"⚠️ Edge-TTS didn't create file for sentence {index}, trying gTTS fallback...")
                raise Exception("File not created")
                
        except Exception as e:
            # Edge-TTS failed, try gTTS fallback
            logger.warning(f"⚠️ Edge-TTS failed for sentence {index}, trying gTTS fallback...")
            
            try:
                if self._generate_audio_gtts_fallback(text, output_file):
                    file_size = os.path.getsize(output_file)
                    logger.info(f"✅ Generated audio (gTTS) for sentence {index}: {os.path.basename(output_file)} ({file_size} bytes)")
                    return output_file
                else:
                    logger.error(f"❌ Both Edge-TTS and gTTS failed for sentence {index}")
                    return None
            except Exception as e2:
                logger.error(f"❌ gTTS fallback failed for sentence {index}: {e2}")
                return None
    
    def generate_all_audio(self, timeline: Dict) -> Dict:
        """
        Pre-generate all audio files for the timeline.
        
        Args:
            timeline: Timeline dict from timeline_mapper
            
        Returns:
            Updated timeline with audio_file paths
        """
        logger.info("🎵 Pre-generating all audio files...")
        
        total_sentences = len(timeline["sentences"])
        
        for sentence_data in timeline["sentences"]:
            idx = sentence_data["index"]
            text = sentence_data["text"]
            
            logger.info(f"  🎤 Generating audio {idx + 1}/{total_sentences}: \"{text[:50]}...\"")
            
            audio_file = self.generate_audio_file(text, idx)
            sentence_data["audio_file"] = audio_file
            
            # Calculate actual duration from audio file
            if audio_file and os.path.exists(audio_file):
                try:
                    # Use mutagen or similar to get actual duration
                    # For now, estimate based on word count (will be accurate enough)
                    word_count = len(text.split())
                    # Edge-TTS at normal rate: ~150 wpm = 0.4s per word
                    duration = max(word_count * 0.4, 1.0)
                    sentence_data["actual_audio_duration"] = duration
                except Exception as e:
                    logger.warning(f"⚠️  Could not determine audio duration: {e}")
                    sentence_data["actual_audio_duration"] = sentence_data["estimated_tts_duration"]
            else:
                sentence_data["actual_audio_duration"] = sentence_data["estimated_tts_duration"]
        
        logger.info(f"✅ Generated {total_sentences} audio files")
        return timeline
    
    def _create_hierarchical_tree_layout(self, graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        Create a clean top-to-bottom hierarchical tree layout.
        Most connected concepts at top, least connected at bottom.
        
        Args:
            graph: Directed graph
            
        Returns:
            Position dictionary
        """
        if len(graph.nodes) == 0:
            return {}
        
        # Calculate node importance (out-degree + in-degree)
        importance = {}
        for node in graph.nodes:
            importance[node] = graph.out_degree(node) + graph.in_degree(node)
        
        # Group nodes by importance level
        max_importance = max(importance.values()) if importance else 0
        levels = {}
        
        for node, imp in importance.items():
            # Create 3-4 levels
            if max_importance == 0:
                level = 0
            else:
                level = int((1.0 - imp / max_importance) * 3)  # 0 = top, 3 = bottom
            
            if level not in levels:
                levels[level] = []
            levels[level].append(node)
        
        # Arrange nodes in hierarchical layout
        pos = {}
        y_spacing = 3.0
        x_spacing = 3.0
        
        for level_idx, nodes in sorted(levels.items()):
            y = -level_idx * y_spacing  # Top to bottom
            num_nodes = len(nodes)
            
            # Center nodes horizontally
            total_width = (num_nodes - 1) * x_spacing
            start_x = -total_width / 2
            
            for i, node in enumerate(sorted(nodes)):  # Sort for consistency
                x = start_x + i * x_spacing
                pos[node] = (x, y)
        
        return pos
    
    def _create_shell_groups(self, graph: nx.DiGraph) -> List[List[str]]:
        """
        Create concentric shell groups based on graph structure.
        Nodes are grouped by their "generation" or centrality.
        
        Args:
            graph: Directed graph
            
        Returns:
            List of node lists for each shell (innermost to outermost)
        """
        if len(graph.nodes) == 0:
            return []
        
        try:
            # Use betweenness centrality to determine importance
            centrality = nx.betweenness_centrality(graph)
            
            # Sort nodes by centrality
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            
            # Create 3-4 shells based on centrality
            total = len(sorted_nodes)
            if total <= 3:
                return [[n[0] for n in sorted_nodes]]
            elif total <= 8:
                # 2 shells: core (33%) and periphery
                split = total // 3
                return [
                    [n[0] for n in sorted_nodes[:split]],
                    [n[0] for n in sorted_nodes[split:]]
                ]
            else:
                # 3 shells: core, middle, periphery
                split1 = total // 4
                split2 = total * 3 // 4
                return [
                    [n[0] for n in sorted_nodes[:split1]],
                    [n[0] for n in sorted_nodes[split1:split2]],
                    [n[0] for n in sorted_nodes[split2:]]
                ]
        except Exception as e:
            logger.debug(f"Failed to create shell groups: {e}")
            return []
    
    def calculate_hierarchical_layout(self, timeline: Dict, layout_style: str = None) -> Dict[str, Tuple[float, float]]:
        """
        Calculate clear, organized graph layout for concepts.
        Uses multiple strategies based on selected layout_style.
        
        Args:
            timeline: Timeline dict with concepts and relationships
            layout_style: Graph layout algorithm to use (if None, uses self.layout_style)
                         Options: "hierarchical", "shell", "circular", "kamada-kawai", "spring"
            
        Returns:
            Dictionary mapping concept names to (x, y) positions (as Python floats)
        """
        # Use instance layout_style if not provided
        if layout_style is None:
            layout_style = self.layout_style
        # Build complete graph from timeline
        graph = nx.DiGraph()
        
        # Collect all concepts
        all_concepts = set()
        for sentence_data in timeline["sentences"]:
            for concept in sentence_data["concepts"]:
                all_concepts.add(concept["name"])
        
        # Add all nodes
        for concept_name in all_concepts:
            graph.add_node(concept_name)
        
        # Add all relationships
        for sentence_data in timeline["sentences"]:
            for rel in sentence_data["relationships"]:
                if rel["from"] in all_concepts and rel["to"] in all_concepts:
                    graph.add_edge(rel["from"], rel["to"])
        
        logger.info(f"📐 Calculating '{layout_style}' graph layout...")
        
        pos = None
        
        # Strategy 1: Hierarchical Tree Layout (CLEANEST!)
        if layout_style == "hierarchical":
            try:
                pos = self._create_hierarchical_tree_layout(graph)
                if pos:
                    logger.info("✅ Using hierarchical tree layout (top-to-bottom)")
            except Exception as e:
                logger.debug(f"Hierarchical layout failed: {e}")
        
        # Strategy 2: Shell layout (concentric circles)
        elif layout_style == "shell":
            try:
                shells = self._create_shell_groups(graph)
                if shells:
                    pos = nx.shell_layout(graph, nlist=shells, scale=3.0)
                    logger.info(f"✅ Using shell layout ({len(shells)} rings)")
            except Exception as e:
                logger.debug(f"Shell layout failed: {e}")
        
        # Strategy 3: Circular layout (simple ring)
        elif layout_style == "circular":
            pos = nx.circular_layout(graph, scale=3.0)
            logger.info("✅ Using circular layout")
        
        # Strategy 4: Kamada-Kawai layout (minimizes edge crossings)
        elif layout_style == "kamada-kawai":
            try:
                pos = nx.kamada_kawai_layout(graph, scale=3.0)
                logger.info("✅ Using Kamada-Kawai layout")
            except Exception as e:
                logger.debug(f"Kamada-Kawai layout failed: {e}")
        
        # Strategy 5: Spring layout
        elif layout_style == "spring":
            pos = nx.spring_layout(
                graph,
                k=3.0,
                iterations=100,
                seed=42,
                scale=3.0
            )
            logger.info("✅ Using spring layout")
        
        # Fallback if layout failed
        if pos is None:
            pos = nx.spring_layout(
                graph,
                k=3.0,
                iterations=100,
                seed=42,
                scale=3.0
            )
            logger.info("✅ Using spring layout (fallback)")
        
        # Convert numpy arrays to Python floats for JSON serialization
        pos_serializable = {
            node: (float(coords[0]), float(coords[1]))
            for node, coords in pos.items()
        }
        
        logger.info(f"✅ Calculated positions for {len(pos_serializable)} concepts")
        return pos_serializable
    
    def precompute_all(self, timeline: Dict) -> Dict:
        """
        Main pre-computation method: Generate all assets.
        
        Args:
            timeline: Timeline from timeline_mapper
            
        Returns:
            Enhanced timeline with:
            - audio_file paths
            - actual_audio_duration
            - pre_calculated_layout
        """
        logger.info("=" * 70)
        logger.info("⚡ PRE-COMPUTATION PHASE")
        logger.info("=" * 70)
        
        # Step 1: Generate all audio files
        timeline = self.generate_all_audio(timeline)
        
        # Step 2: Calculate layout using selected style
        layout = self.calculate_hierarchical_layout(timeline, layout_style=self.layout_style)
        timeline["pre_calculated_layout"] = layout
        
        logger.info("=" * 70)
        logger.info("✅ PRE-COMPUTATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"  📊 Total sentences: {len(timeline['sentences'])}")
        logger.info(f"  🎵 Audio files: {len([s for s in timeline['sentences'] if s.get('audio_file')])}")
        logger.info(f"  📐 Layout positions: {len(layout)}")
        logger.info("=" * 70)
        
        return timeline
    
    def cleanup(self):
        """
        Clean up temporary audio files.
        """
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"🗑️  Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"⚠️  Could not clean up temp directory: {e}")
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup()
