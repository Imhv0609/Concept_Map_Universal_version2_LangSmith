"""
States for Universal Concept Map Teaching Agent

This module defines the state structure for the description-based concept mapping workflow
that can extract concepts directly from user descriptions of any length (1 word to 3000+ words).
"""

from typing import Dict, List, Any, TypedDict


class ConceptMapState(TypedDict):
    """State structure for the description-based concept mapping workflow"""
    
    # Primary inputs
    description: str                   # The main description text (1 word to 3000+ words) - PRIMARY INPUT
    educational_level: str             # Target educational level (e.g., "elementary", "high school", "graduate")
    topic_name: str                    # Topic name (auto-extracted if not provided)
    
    # Description analysis results
    description_analysis: Dict[str, Any]  # Results from description_analyzer.analyze_description_complexity()
    complexity_config: Dict[str, Any]     # Adjusted complexity configuration for this description
    
    # Extracted concepts (NEW APPROACH - Direct extraction from description)
    extracted_concepts: List[Dict[str, Any]]        # Key concepts extracted directly from description
    concept_relationships: List[Dict[str, Any]]     # Relationships between extracted concepts
    concept_hierarchy: List[Dict[str, Any]]         # Hierarchical organization of concepts
    
    # Educational enrichment
    enriched_concepts: Dict[str, Dict[str, Any]]    # Concepts enhanced with educational metadata
    learning_objectives: List[str]                   # Generated learning objectives
    teaching_strategies: List[Dict[str, Any]]       # Recommended teaching approaches
    
    # Legacy fields (for backward compatibility - may be deprecated)
    raw_subtopics: List[str]                        # DEPRECATED: Use extracted_concepts instead
    subtopic_concepts: Dict[str, List[Dict[str, Any]]]  # DEPRECATED
    key_concepts_per_subtopic: Dict[str, List[Dict[str, Any]]]  # DEPRECATED
    subtopic_hierarchies: Dict[str, List[Dict[str, Any]]]  # DEPRECATED
    cross_subtopic_links: List[Dict[str, Any]]      # DEPRECATED: Use concept_relationships instead
    enriched_subtopics: Dict[str, Dict[str, Any]]   # DEPRECATED: Use enriched_concepts instead
    
    # Metadata
    processing_log: List[str]         # Step-by-step processing log
    errors: List[str]                 # Any errors encountered
    success: bool                     # Overall success status
    timestamp: str                    # Processing timestamp