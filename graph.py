"""
Description-Based Concept Map Graph Definition

This module defines the LangGraph workflow for the description-based concept mapping system
that extracts concepts directly from user descriptions (1 word to 3000+ words).
"""

from langgraph.graph import StateGraph, END
from states import ConceptMapState
from nodes import (
    extract_concepts_from_description,
    analyze_concept_relationships,
    build_concept_hierarchy,
    enrich_with_educational_metadata,
    initialize_legacy_fields,
    extract_all_in_one_call  # NEW: Combined node
)


def create_description_based_concept_map_graph():
    """
    Create the LangGraph workflow for description-based concept mapping
    
    ULTRA-OPTIMIZED 1-Node Workflow:
    1. Combined Extraction → Extract concepts, relationships, AND hierarchy in ONE LLM call
    
    Previous optimizations:
    - Removed educational enrichment (Node 4) - saved 77s & 2717 tokens
    - Combined 3 nodes into 1 - saves ~60s additional
    
    Expected performance:
    - Previous: 141s with 3 nodes
    - Current: 60-80s with 1 node (50% faster!)
    
    Returns:
        Compiled LangGraph workflow
    """
    
    # Create the graph
    workflow = StateGraph(ConceptMapState)
    
    # Add ONLY the combined node
    workflow.add_node("combined_extraction", extract_all_in_one_call)
    workflow.add_node("initialize_legacy", initialize_legacy_fields)
    
    # Define the ultra-optimized flow (just 1 LLM call + legacy init)
    workflow.set_entry_point("combined_extraction")
    workflow.add_edge("combined_extraction", "initialize_legacy")
    workflow.add_edge("initialize_legacy", END)
    
    # Compile the workflow
    return workflow.compile()


# Legacy function for backward compatibility
def create_universal_concept_map_graph():
    """Legacy function - redirects to new description-based approach"""
    return create_description_based_concept_map_graph()


def print_description_based_workflow_summary():
    """
    Print a summary of the description-based workflow
    """
    print("🔄 ULTRA-OPTIMIZED 1-Node Concept Map Workflow:")
    print("   � Combined Extraction - Extract concepts, relationships, AND hierarchy in ONE call")
    print("   ⚡ Expected: 60-80s (vs 141s with 3 nodes = 50% faster!)")
    print()


# Legacy function for backward compatibility  
def print_universal_workflow_summary():
    """Legacy function - redirects to new description-based summary"""
    print_description_based_workflow_summary()


def print_universal_workflow_summary():
    """Print a summary of the universal concept mapping workflow"""
    print("🔄 Universal Concept Map Workflow:")
    print("  1. Topic Analysis    → Extract key subtopics from any topic")
    print("  2. Concept Generation → Generate specific concepts for each subtopic")
    print("  3. Key Identification → Identify most important concepts per subtopic")
    print("  4. Relationship Building → Create hierarchies and cross-links")
    print("  5. Educational Enrichment → Add teaching metadata and strategies")
    print()