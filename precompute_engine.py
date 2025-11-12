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
    
    async def _get_word_timings_from_edgetts_async(self, text: str) -> List[Dict]:
        """
        Get precise word-level timings from Edge-TTS (async).
        This uses Edge-TTS's SubMaker to get exact timing data.
        
        Args:
            text: Text to get timings for
            
        Returns:
            List of dicts with format: [{"word": "hello", "start_time": 0.0, "end_time": 0.5}, ...]
        """
        import edge_tts
        from edge_tts import SubMaker
        
        try:
            communicate = edge_tts.Communicate(text, self.voice, rate=self.rate)
            submaker = SubMaker()
            
            word_timings = []
            last_offset = 0
            
            # Stream to get timing information
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    pass  # We don't need the actual audio here
                elif chunk["type"] == "WordBoundary":
                    # Get word boundary data directly
                    offset_ms = chunk["offset"]
                    duration_ms = chunk.get("duration", 0)
                    word_text = chunk["text"]
                    
                    # Calculate start and end times
                    start_time = offset_ms / 10000.0  # Edge-TTS uses 100-nanosecond units
                    
                    # If we have the next word's offset, use it to calculate duration
                    # Otherwise, estimate based on word length
                    if duration_ms > 0:
                        end_time = start_time + (duration_ms / 10000.0)
                    else:
                        # Estimate: average 0.4s per word
                        end_time = start_time + 0.4
                    
                    word_timings.append({
                        "word": word_text.strip(),
                        "start_time": start_time,
                        "end_time": end_time
                    })
                    
                    last_offset = offset_ms
            
            if word_timings:
                logger.info(f"✅ Got {len(word_timings)} word timings from Edge-TTS")
            
            return word_timings
            
        except Exception as e:
            logger.warning(f"⚠️ Could not get word timings from Edge-TTS: {e}")
            return []
    
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
    
    def get_word_timings_from_edgetts(self, text: str) -> List[Dict]:
        """
        Get precise word-level timings from Edge-TTS (synchronous wrapper).
        
        Args:
            text: Text to get timings for
            
        Returns:
            List of dicts with format: [{"word": "hello", "start_time": 0.0, "end_time": 0.5}, ...]
        """
        try:
            # Apply nest_asyncio for Streamlit compatibility
            try:
                import nest_asyncio
                nest_asyncio.apply()
            except:
                pass
            
            # Handle event loop properly
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    return loop.run_until_complete(self._get_word_timings_from_edgetts_async(text))
                else:
                    return loop.run_until_complete(self._get_word_timings_from_edgetts_async(text))
            except RuntimeError:
                return asyncio.run(self._get_word_timings_from_edgetts_async(text))
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to get Edge-TTS timings: {e}")
            return []
    
    def _update_concept_timings_with_precise_data(self, timeline: Dict, precise_timings: List[Dict]) -> Dict:
        """
        Update concept reveal times using precise word timings from Edge-TTS.
        
        Args:
            timeline: Timeline dict with concepts
            precise_timings: List of word timings from Edge-TTS
            
        Returns:
            Updated timeline with precise reveal times
        """
        full_text = timeline.get("full_text", "").lower()
        concepts = timeline.get("concepts", [])
        
        for concept in concepts:
            concept_name = concept.get("name", "").lower()
            if not concept_name:
                continue
            
            # Find the last word of the concept name in the text
            concept_words = concept_name.split()
            if not concept_words:
                continue
            
            last_word = concept_words[-1]
            
            # Find this word in precise timings
            for i, timing in enumerate(precise_timings):
                word = timing["word"].lower().strip('.,!?;:')
                if word == last_word:
                    # Check if previous words match (for multi-word concepts)
                    if len(concept_words) > 1:
                        match = True
                        for j, concept_word in enumerate(reversed(concept_words[:-1])):
                            idx = i - j - 1
                            if idx < 0 or precise_timings[idx]["word"].lower().strip('.,!?;:') != concept_word:
                                match = False
                                break
                        if match:
                            concept["reveal_time"] = timing["end_time"]
                            logger.debug(f"   ✅ Updated '{concept.get('name')}': {timing['end_time']:.2f}s (precise)")
                            break
                    else:
                        # Single word concept
                        concept["reveal_time"] = timing["end_time"]
                        logger.debug(f"   ✅ Updated '{concept.get('name')}': {timing['end_time']:.2f}s (precise)")
                        break
        
        logger.info(f"✅ Updated {len(concepts)} concepts with precise timings")
        return timeline
    
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
        Pre-generate audio file for the continuous timeline.
        Uses Edge-TTS to get precise word timings, then generates audio with best available TTS.
        
        Args:
            timeline: Timeline dict from timeline_mapper
            
        Returns:
            Updated timeline with audio_file path, actual duration, and precise word timings
        """
        logger.info("🎵 Pre-generating audio for continuous timeline...")
        
        # Get full text from timeline
        full_text = timeline.get("full_text", "")
        
        if not full_text:
            logger.warning("⚠️ No full_text in timeline, falling back to sentences")
            # Fallback: use legacy sentence structure
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
                        word_count = len(text.split())
                        duration = max(word_count * 0.35, 0.5)
                        sentence_data["actual_audio_duration"] = duration
                    except Exception as e:
                        logger.warning(f"⚠️  Could not determine audio duration: {e}")
                        sentence_data["actual_audio_duration"] = sentence_data["estimated_tts_duration"]
                else:
                    sentence_data["actual_audio_duration"] = sentence_data["estimated_tts_duration"]
            
            logger.info(f"✅ Generated {total_sentences} audio files (legacy mode)")
            return timeline
        
        # STEP 1: NOTE - Edge-TTS word boundaries are not reliably available
        # Using estimated timings (0.40s per word) which work well for both Edge-TTS and gTTS
        logger.info("ℹ️ Using estimated word timings (0.40s/word, tuned for gTTS)")
        
        # STEP 2: Generate audio file (Edge-TTS or gTTS fallback)
        logger.info(f"  🎤 Generating audio for full text: \"{full_text[:100]}...\"")
        audio_file = self.generate_audio_file(full_text, 0)
        
        if audio_file and os.path.exists(audio_file):
            # Try to get actual duration from audio file
            actual_duration = None
            try:
                from mutagen.mp3 import MP3
                audio = MP3(audio_file)
                actual_duration = audio.info.length
                logger.info(f"✅ Audio file duration: {actual_duration:.2f}s")
            except Exception as e:
                logger.warning(f"⚠️ Could not read audio duration with mutagen: {e}")
                # Try alternative method using pydub
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_mp3(audio_file)
                    actual_duration = len(audio) / 1000.0  # Convert ms to seconds
                    logger.info(f"✅ Audio file duration (pydub): {actual_duration:.2f}s")
                except Exception as e2:
                    logger.warning(f"⚠️ Could not read audio duration with pydub: {e2}")
                    # Fallback: use estimated duration from timeline
                    actual_duration = timeline["metadata"].get("total_duration", 0.0)
                    logger.info(f"✅ Using estimated duration: {actual_duration:.2f}s")
            
            # CRITICAL: Scale concept reveal times to match actual audio duration
            estimated_duration = timeline["metadata"].get("total_duration", 0.0)
            if actual_duration and estimated_duration > 0 and abs(actual_duration - estimated_duration) > 0.5:
                # Audio is significantly different from estimate - rescale all timings
                scale_factor = actual_duration / estimated_duration
                logger.info(f"🔄 Rescaling concept timings: {estimated_duration:.2f}s → {actual_duration:.2f}s (factor: {scale_factor:.3f})")
                
                for concept in timeline.get("concepts", []):
                    old_time = concept.get("reveal_time", 0.0)
                    new_time = old_time * scale_factor
                    concept["reveal_time"] = new_time
                    logger.debug(f"   📍 {concept.get('name')}: {old_time:.2f}s → {new_time:.2f}s")
                
                # Update metadata with actual duration
                timeline["metadata"]["total_duration"] = actual_duration
                timeline["metadata"]["original_estimated_duration"] = estimated_duration
                timeline["metadata"]["timing_scale_factor"] = scale_factor
                
                logger.info(f"✅ Rescaled {len(timeline.get('concepts', []))} concept timings to match audio")
            else:
                logger.info(f"✓ Estimated duration close enough to actual ({estimated_duration:.2f}s ≈ {actual_duration:.2f}s)")
            
            # Store in timeline
            timeline["audio_file"] = audio_file
            timeline["actual_audio_duration"] = actual_duration
            
            # Also store in sentences[0] for backward compatibility
            if timeline["sentences"]:
                timeline["sentences"][0]["audio_file"] = audio_file
                timeline["sentences"][0]["actual_audio_duration"] = actual_duration
            
            logger.info(f"✅ Generated audio file: {os.path.basename(audio_file)}")
        else:
            logger.error("❌ Failed to generate audio file")
            timeline["audio_file"] = None
            timeline["actual_audio_duration"] = timeline["metadata"].get("total_duration", 0.0)
        
        return timeline
    
    def _resolve_node_overlaps(self, pos: Dict[str, Tuple[float, float]], min_distance: float = 5.0) -> Dict[str, Tuple[float, float]]:
        """
        Resolve overlapping nodes using force-directed adjustment.
        Ensures minimum distance between all node pairs.
        
        Args:
            pos: Initial position dictionary
            min_distance: Minimum distance between node centers (default 5.0 to prevent overlap with node size ~3000)
            
        Returns:
            Adjusted position dictionary with no overlaps
        """
        import math
        
        adjusted_pos = pos.copy()
        max_iterations = 200  # Increased for better convergence
        
        for iteration in range(max_iterations):
            moved = False
            
            # Check all pairs of nodes
            nodes = list(adjusted_pos.keys())
            for i, node1 in enumerate(nodes):
                for node2 in nodes[i+1:]:
                    x1, y1 = adjusted_pos[node1]
                    x2, y2 = adjusted_pos[node2]
                    
                    # Calculate distance
                    dx = x2 - x1
                    dy = y2 - y1
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    # If nodes are too close, push them apart
                    if distance < min_distance:
                        moved = True
                        
                        # Calculate push direction
                        if distance < 0.01:  # Avoid division by zero
                            # Nodes are at same position, push in random direction
                            angle = hash(node1 + node2) % 360
                            dx = math.cos(math.radians(angle))
                            dy = math.sin(math.radians(angle))
                            distance = 0.01
                        
                        # Push amount (half to each node)
                        push = (min_distance - distance) / 2
                        push_x = (dx / distance) * push
                        push_y = (dy / distance) * push
                        
                        # Move nodes apart
                        adjusted_pos[node1] = (x1 - push_x, y1 - push_y)
                        adjusted_pos[node2] = (x2 + push_x, y2 + push_y)
            
            # If no nodes moved, we're done
            if not moved:
                logger.info(f"   Resolved overlaps in {iteration + 1} iterations")
                break
        
        return adjusted_pos
    
    def _create_hierarchical_tree_layout(self, graph: nx.DiGraph) -> Dict[str, Tuple[float, float]]:
        """
        True hierarchical tree layout - root at top, children below.
        Uses NetworkX's built-in tree layout with overlap prevention.
        
        Args:
            graph: Directed graph
            
        Returns:
            Position dictionary
        """
        if len(graph.nodes) == 0:
            return {}
        
        # SMART GRID LAYOUT (Always used - replaces Graphviz)
        # Find the root node (most connected)
        importance = {}
        for node in graph.nodes:
            importance[node] = graph.out_degree(node) + graph.in_degree(node)
        
        # Sort by importance
        sorted_nodes = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        pos = {}
        num_nodes = len(sorted_nodes)
        
        if num_nodes == 0:
            return {}
        
        # SMART GRID LAYOUT
        # Root node at top center (position 0, 0)
        root_node = sorted_nodes[0][0]
        pos[root_node] = (0.0, 0.0)
        
        # Grid parameters
        grid_columns = 3  # Fixed 3 columns
        horizontal_spacing = 8.0  # Fixed spacing between columns
        vertical_spacing = 7.0  # Row height (same as before)
        
        # Arrange remaining nodes in 3-column grid below root
        remaining_nodes = [node for node, imp in sorted_nodes[1:]]  # Exclude root
        
        for idx, node in enumerate(remaining_nodes):
            # Calculate grid position (row, column)
            row = idx // grid_columns  # Integer division for row
            col = idx % grid_columns   # Modulo for column (0, 1, 2)
            
            # Calculate actual (x, y) coordinates
            # X: Center the grid, columns at -8, 0, +8
            x = (col - 1) * horizontal_spacing  # col 0→-8, col 1→0, col 2→+8
            
            # Y: Start first row at -7, then -14, -21, etc.
            y = -(row + 1) * vertical_spacing
            
            pos[node] = (x, y)
        
        # Resolve any remaining overlaps with sufficient spacing for node size 3000
        pos = self._resolve_node_overlaps(pos, min_distance=5.0)
        
        num_rows = (len(remaining_nodes) + grid_columns - 1) // grid_columns  # Ceiling division
        logger.info(f"   Created smart grid layout: root + {len(remaining_nodes)} nodes in {num_rows} rows × {grid_columns} columns")
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
    
    def _filter_edges_by_incoming_limit(self, graph: nx.DiGraph, max_incoming: int = 2) -> nx.DiGraph:
        """
        Limit incoming edges per node to create cleaner hierarchy.
        Root node gets 0 incoming edges, all others get max 2.
        Prioritizes edges from more important (higher degree) source nodes.
        
        Args:
            graph: Directed graph
            max_incoming: Maximum incoming edges per non-root node (default: 2)
            
        Returns:
            Filtered graph with edge constraints applied
        """
        # Find root node (highest importance = out_degree + in_degree)
        if len(graph.nodes()) == 0:
            return graph
        
        importance = {}
        for node in graph.nodes():
            importance[node] = graph.out_degree(node) + graph.in_degree(node)
        
        root_node = max(importance.items(), key=lambda x: x[1])[0]
        logger.info(f"   Identified root node: '{root_node}' (importance: {importance[root_node]})")
        
        # Remove ALL incoming edges to root node
        incoming_to_root = list(graph.in_edges(root_node))
        for source, target in incoming_to_root:
            graph.remove_edge(source, target)
        
        if incoming_to_root:
            logger.info(f"   Removed {len(incoming_to_root)} incoming edges from root node")
        
        # For all other nodes, limit to max_incoming edges (prioritize by source importance)
        edges_removed_count = 0
        for node in graph.nodes():
            if node == root_node:
                continue  # Skip root, already handled
            
            incoming_edges = list(graph.in_edges(node))
            
            if len(incoming_edges) > max_incoming:
                # Calculate importance of each incoming edge based on source node
                edge_importance = []
                for source, target in incoming_edges:
                    source_importance = importance.get(source, 0)
                    edge_importance.append((source, target, source_importance))
                
                # Sort by source importance (descending) - keep most important sources
                edge_importance.sort(key=lambda x: x[2], reverse=True)
                
                # Keep only top max_incoming edges
                edges_to_keep = set((src, tgt) for src, tgt, imp in edge_importance[:max_incoming])
                
                # Remove excess edges
                for source, target in incoming_edges:
                    if (source, target) not in edges_to_keep:
                        graph.remove_edge(source, target)
                        edges_removed_count += 1
        
        if edges_removed_count > 0:
            logger.info(f"   Removed {edges_removed_count} excess edges (max {max_incoming} incoming per node)")
        
        return graph
    
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
        edges_added = 0
        for sentence_data in timeline["sentences"]:
            for rel in sentence_data["relationships"]:
                if rel["from"] in all_concepts and rel["to"] in all_concepts:
                    graph.add_edge(rel["from"], rel["to"])
                    edges_added += 1
        
        logger.info(f"📐 Calculating '{layout_style}' graph layout...")
        logger.info(f"   Graph: {len(graph.nodes())} nodes, {len(graph.edges())} edges (added {edges_added})")
        
        # Apply edge constraints: max 2 incoming edges per node (0 for root)
        graph = self._filter_edges_by_incoming_limit(graph, max_incoming=2)
        logger.info(f"   After edge filtering: {len(graph.edges())} edges (limited to max 2 incoming per node)")
        
        if len(graph.edges()) == 0:
            logger.warning("⚠️ Graph has NO edges! Hierarchical layout will not work well. Check LLM relationship extraction!")
        
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
