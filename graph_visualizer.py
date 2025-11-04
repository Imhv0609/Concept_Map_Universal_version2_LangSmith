"""
Graph Visualizer for Concept Maps
Generates visual representations of concept maps using NetworkX and Matplotlib
"""

import json
import logging
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConceptMapVisualizer:
    """Visualizes concept maps as hierarchical graphs using NetworkX and Matplotlib"""
    
    def __init__(self):
        """Initialize the visualizer"""
        logger.info("ConceptMapVisualizer initialized")
        self.graph = nx.DiGraph()
        self.concepts = {}
        self.relationships = []
        self.hierarchy = {}
        
    def load_from_json(self, json_file_path: str) -> bool:
        """
        Load concept map data from JSON file
        
        Args:
            json_file_path: Path to the JSON file containing concept map data
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading concept map from: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract concepts - handle both list and dict formats
            concepts_data = data.get('extracted_concepts', [])
            if isinstance(concepts_data, list):
                # Convert list format to dict format
                self.concepts = {concept['name']: concept for concept in concepts_data}
            else:
                self.concepts = concepts_data
            logger.info(f"Extracted {len(self.concepts)} concepts")
            
            # Extract relationships
            self.relationships = data.get('concept_relationships', [])
            logger.info(f"Extracted {len(self.relationships)} relationships")
            
            # Extract hierarchy - handle both list and dict formats
            hierarchy_data = data.get('concept_hierarchy', [])
            if isinstance(hierarchy_data, list):
                # Convert list format to dict format
                self.hierarchy = {'levels': hierarchy_data}
            else:
                self.hierarchy = hierarchy_data
            hierarchy_levels = len(self.hierarchy.get('levels', []))
            logger.info(f"Extracted {hierarchy_levels} hierarchy levels")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading JSON file: {e}")
            return False
    
    def extract_graph_data(self) -> Dict[str, Any]:
        """
        Extract and process graph data for visualization
        
        Returns:
            dict: Graph statistics and processed data
        """
        # Create NetworkX graph
        self.graph.clear()
        
        # Add concept nodes
        for concept_name, concept_data in self.concepts.items():
            self.graph.add_node(
                concept_name,
                type=concept_data.get('type', 'concept'),
                importance=concept_data.get('importance', 'medium'),
                definition=concept_data.get('definition', ''),
                level=self._get_concept_level(concept_name)
            )
        
        # Add relationship edges
        for relationship in self.relationships:
            source = relationship.get('from_concept', '')
            target = relationship.get('to_concept', '')
            rel_type = relationship.get('relationship_type', 'related')
            strength = relationship.get('strength', 'medium')
            
            if source in self.concepts and target in self.concepts:
                self.graph.add_edge(
                    source,
                    target,
                    relationship=rel_type,
                    strength=strength,
                    description=relationship.get('description', '')
                )
        
        logger.info(f"Created NetworkX graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Gather statistics
        concept_types = list(set(data.get('type', 'concept') for data in self.concepts.values()))
        relationship_types = list(set(rel.get('relationship_type', 'related') for rel in self.relationships))
        
        stats = {
            'num_concepts': len(self.concepts),
            'num_relationships': len(self.relationships),
            'num_hierarchy_levels': len(self.hierarchy.get('levels', [])),
            'concept_types': concept_types,
            'relationship_types': relationship_types
        }
        
        return stats
    
    def _get_concept_level(self, concept_name: str) -> int:
        """
        Get the hierarchy level of a concept
        
        Args:
            concept_name: Name of the concept
            
        Returns:
            int: Hierarchy level (0-based)
        """
        levels = self.hierarchy.get('levels', [])
        for i, level in enumerate(levels):
            # Handle both dict and list formats for level concepts
            if isinstance(level, dict):
                concepts_in_level = level.get('concepts', [])
                # Extract concept names from concept objects
                concept_names = []
                for concept in concepts_in_level:
                    if isinstance(concept, dict):
                        concept_names.append(concept.get('name', ''))
                    else:
                        concept_names.append(str(concept))
            else:
                # If level is a list, assume it contains concept names directly
                concept_names = level if isinstance(level, list) else []
            
            if concept_name in concept_names:
                return i
        return 0  # Default to level 0 if not found
    
    def create_hierarchical_layout(self) -> Dict[str, Tuple[float, float]]:
        """
        Create a hierarchical layout for the graph
        
        Returns:
            dict: Node positions {node_name: (x, y)}
        """
        pos = {}
        levels = self.hierarchy.get('levels', [])
        
        if not levels:
            # Fallback to spring layout if no hierarchy
            logger.info("No hierarchy found, using spring layout")
            return nx.spring_layout(self.graph, k=3, iterations=50)
        
        logger.info(f"Created hierarchical layout for {len(self.concepts)} nodes")
        
        # Calculate positions based on hierarchy levels
        level_height = 2.0
        for level_idx, level in enumerate(levels):
            # Handle both dict and list formats for level concepts
            if isinstance(level, dict):
                concepts_in_level = level.get('concepts', [])
                # Extract concept names from concept objects
                concept_names = []
                for concept in concepts_in_level:
                    if isinstance(concept, dict):
                        concept_names.append(concept.get('name', ''))
                    else:
                        concept_names.append(str(concept))
            else:
                # If level is a list, assume it contains concept names directly
                concept_names = level if isinstance(level, list) else []
                
            y_position = (len(levels) - level_idx - 1) * level_height
            
            # Distribute concepts horizontally within the level
            if concept_names:
                x_spacing = 4.0 if len(concept_names) > 1 else 0
                start_x = -(len(concept_names) - 1) * x_spacing / 2
                
                for i, concept in enumerate(concept_names):
                    if concept in self.graph.nodes():
                        x_position = start_x + i * x_spacing
                        pos[concept] = (x_position, y_position)
        
        # Handle any concepts not in hierarchy levels
        for node in self.graph.nodes():
            if node not in pos:
                pos[node] = (0, 0)
        
        return pos
    
    def generate_graph_image(self, output_path: str = None) -> str:
        """
        Generate and save the concept map visualization
        
        Args:
            output_path: Custom output path (optional)
            
        Returns:
            str: Path to the generated image file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output/concept_map_{timestamp}.png"
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Create the plot
        plt.figure(figsize=(16, 12))
        plt.clf()
        
        # Get hierarchical layout
        pos = self.create_hierarchical_layout()
        
        # Debug: Check if we have nodes and positions
        logger.info(f"Number of nodes: {len(self.graph.nodes())}")
        logger.info(f"Number of edges: {len(self.graph.edges())}")
        logger.info(f"Position count: {len(pos)}")
        
        if len(self.graph.nodes()) == 0:
            logger.warning("No nodes to visualize!")
            # Create a simple error message plot
            plt.text(0.5, 0.5, 'No concepts to visualize', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=plt.gca().transAxes, fontsize=16)
        else:
            # Define colors and styles
            node_colors = {
                'fundamental': '#FF6B6B',  # Red
                'principle': '#4ECDC4',     # Teal
                'process': '#45B7D1',       # Blue
                'application': '#96CEB4',   # Green
                'concept': '#FECA57'        # Yellow (default)
            }
            
            edge_colors = {
                'strong': '#2C3E50',        # Dark blue
                'medium': '#7F8C8D',        # Gray
                'weak': '#BDC3C7'           # Light gray
            }
            
            # Draw nodes - get all node attributes first
            node_colors_list = []
            node_sizes_list = []
            for node in self.graph.nodes():
                node_data = self.graph.nodes[node]
                node_type = node_data.get('type', 'concept')
                importance = node_data.get('importance', 'medium')
                
                color = node_colors.get(node_type, node_colors['concept'])
                size = 3000 if importance == 'high' else 2000 if importance == 'medium' else 1500
                
                node_colors_list.append(color)
                node_sizes_list.append(size)
            
            logger.info(f"Drawing {len(node_colors_list)} nodes")
            
            # Draw all nodes at once
            nx.draw_networkx_nodes(
                self.graph, pos,
                node_color=node_colors_list,
                node_size=node_sizes_list,
                alpha=0.8,
                edgecolors='black',
                linewidths=2
            )
            
            # Draw all edges at once if they exist
            if len(self.graph.edges()) > 0:
                logger.info(f"Drawing {len(self.graph.edges())} edges")
                nx.draw_networkx_edges(
                    self.graph, pos,
                    edge_color='#7F8C8D',  # Gray
                    width=2,
                    alpha=0.7,
                    arrows=True,
                    arrowsize=20,
                    arrowstyle='->'
                )
                
                # Draw edge labels (relationship types)
                edge_labels = {}
                for u, v, data in self.graph.edges(data=True):
                    rel_type = data.get('relationship', 'related')
                    edge_labels[(u, v)] = rel_type
                
                logger.info(f"Drawing {len(edge_labels)} edge labels")
                nx.draw_networkx_edge_labels(
                    self.graph, pos,
                    edge_labels=edge_labels,
                    font_size=8,
                    font_color='#2C3E50',  # Dark blue
                    font_weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.7)
                )
            
            # Draw node labels
            labels = {node: node for node in self.graph.nodes()}
            nx.draw_networkx_labels(
                self.graph, pos,
                labels,
                font_size=10,
                font_weight='bold',
                font_color='black'
            )
        
        # Add legend
        legend_elements = []
        for node_type, color in node_colors.items():
            legend_elements.append(patches.Patch(color=color, label=f'{node_type.title()} Concept'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))
        
        # Add title and formatting
        plt.title('Concept Map Visualization', fontsize=16, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        
        # Save the image
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        logger.info(f"Concept map saved to: {output_path}")
        return output_path


def main():
    """Test the visualizer with sample data"""
    visualizer = ConceptMapVisualizer()
    
    # Find the most recent JSON file
    output_dir = Path("output")
    if output_dir.exists():
        json_files = list(output_dir.glob("*.json"))
        if json_files:
            latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
            print(f"Testing with: {latest_file}")
            
            if visualizer.load_from_json(str(latest_file)):
                stats = visualizer.extract_graph_data()
                print(f"Graph data loaded successfully!")
                print(f"Graph statistics: {stats}")
                
                image_path = visualizer.generate_graph_image()
                print(f"Concept map generated: {image_path}")
            else:
                print("Failed to load concept map data")
        else:
            print("No JSON files found in output directory")
    else:
        print("Output directory not found")


if __name__ == "__main__":
    main()