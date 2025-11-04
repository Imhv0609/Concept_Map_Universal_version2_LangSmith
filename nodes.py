"""
Description-Based Concept Map Nodes

This module contains the 4-node processing workflow for description-based concept mapping
that extracts concepts directly from user descriptions (1 word to 3000+ words).

New 4-Node Workflow:
1. extract_concepts_from_description - Extract key concepts from description text
2. analyze_concept_relationships - Identify relationships between concepts  
3. build_concept_hierarchy - Organize concepts into learning hierarchy
4. enrich_with_educational_metadata - Add educational metadata and strategies
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime
import google.generativeai as genai
import os
import re
from dotenv import load_dotenv
from states import ConceptMapState
from description_analyzer import (
    analyze_description_complexity, 
    adjust_complexity_for_educational_level, 
    extract_topic_name_from_description
)
from token_tracker import log_token_usage, get_tracker

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def clean_json_response(response_text: str) -> str:
    """
    Clean and extract JSON from AI response text
    
    Args:
        response_text (str): Raw response from AI model
        
    Returns:
        str: Cleaned JSON text
    """
    # Extract JSON from code blocks
    if "```json" in response_text:
        json_start = response_text.find("```json") + 7
        json_end = response_text.find("```", json_start)
        if json_end == -1:
            json_end = len(response_text)
        json_text = response_text[json_start:json_end].strip()
    else:
        json_text = response_text.strip()
    
    # Remove any trailing commas before closing braces/brackets
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    
    # Remove any comments (lines starting with //)
    json_text = re.sub(r'//.*\n', '', json_text)
    
    # Fix common JSON issues
    # Remove trailing commas in arrays and objects
    json_text = re.sub(r',(\s*[}\]])', r'\1', json_text)
    
    # Fix missing commas between objects in arrays
    json_text = re.sub(r'}\s*{', r'},{', json_text)
    
    # Fix misplaced escape characters in property names
    # Pattern: "property\": "value" -> "property": "value"
    json_text = re.sub(r'"([^"]+)\\"\s*:\s*\\"([^"]+)"', r'"\1": "\2"', json_text)
    
    # Fix escaped quotes that shouldn't be escaped
    # Pattern: "educational_level\": \"elementary" -> "educational_level": "elementary"
    json_text = re.sub(r'([^\\])\\"\s*:\s*\\"([^"]+)"', r'\1": "\2"', json_text)
    
    # Remove extra whitespace
    json_text = json_text.strip()
    
    return json_text


def safe_json_parse(json_text: str, fallback_value: Any = None) -> Any:
    """
    Safely parse JSON with error handling
    
    Args:
        json_text (str): JSON text to parse
        fallback_value (Any): Value to return if parsing fails
        
    Returns:
        Any: Parsed JSON or fallback value
    """
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parsing failed: {e}")
        logger.warning(f"Problematic JSON text: {json_text[:500]}...")
        return fallback_value


def extract_all_in_one_call(state: ConceptMapState) -> ConceptMapState:
    """
    OPTIMIZED COMBINED NODE: Extract concepts, relationships, and hierarchy in ONE LLM call
    
    This replaces the previous 3 separate nodes:
    - Node 1: Extract concepts
    - Node 2: Analyze relationships  
    - Node 3: Build hierarchy
    
    This approach reduces inference time by ~50% by eliminating 2 LLM calls.
    """
    logger.info(f"🚀 Extracting all data (concepts, relationships, hierarchy) in ONE call")
    logger.info(f"📝 Description: {len(state['description'])} characters")
    
    try:
        # Analyze description complexity
        description_analysis = analyze_description_complexity(state['description'])
        
        # Adjust complexity for educational level
        base_complexity = description_analysis['complexity']
        adjusted_complexity = adjust_complexity_for_educational_level(base_complexity, state['educational_level'])
        
        # Auto-extract topic name if not provided
        if not state.get('topic_name'):
            state['topic_name'] = extract_topic_name_from_description(state['description'])
        
        # Store analysis results
        state['description_analysis'] = description_analysis
        state['complexity_config'] = adjusted_complexity
        
        # Log complexity decisions
        state['processing_log'].append(
            f"📊 Description analysis: {description_analysis['word_count']} words → "
            f"{adjusted_complexity['target_concepts']} concepts ({adjusted_complexity['detail_level']} level)"
        )
        
        model = genai.GenerativeModel('gemini-2.5-flash-lite')  # FASTEST MODEL for maximum speed
        
        # Create dynamic prompt based on description analysis
        target_concepts = adjusted_complexity['target_concepts']
        detail_level = adjusted_complexity['detail_level']
        
        # OPTIMIZED COMPRESSED PROMPT - 60% token reduction
        prompt = f"""Extract concepts, relationships, and hierarchy from the description as JSON.

DESCRIPTION: "{state['description']}"
EDUCATIONAL LEVEL: {state['educational_level']}
TARGET CONCEPTS: {target_concepts}

JSON FORMAT:
{{
  "extracted_concepts": [{{"name": str, "type": "fundamental|process|application|principle", "importance": "high|medium|low", "definition": str}}],
  "concept_relationships": [{{"from_concept": str, "to_concept": str, "relationship_type": "enables|requires|produces|is_part_of|influences", "relationship_description": str, "strength": "strong|medium|weak"}}],
  "concept_hierarchy": [{{"level": int, "level_name": str, "level_description": str, "concepts": [{{"name": str}}], "difficulty": "easy|moderate|challenging"}}]
}}

RULES:
- Extract exactly {target_concepts} concepts
- Create meaningful relationships
- Organize into 2-4 hierarchy levels
- Match {state['educational_level']} level
"""
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Track token usage
        token_info = log_token_usage("extract_all_in_one_call", prompt, response_text)
        get_tracker().add_node("combined_extraction", token_info)
        
        # Parse JSON response
        json_text = clean_json_response(response_text)
        result = safe_json_parse(json_text, {})
        
        if result:
            # Extract all three components from single response
            extracted_concepts = result.get('extracted_concepts', [])
            concept_relationships = result.get('concept_relationships', [])
            concept_hierarchy = result.get('concept_hierarchy', [])
            
            # Store in state
            state['extracted_concepts'] = extracted_concepts
            state['concept_relationships'] = concept_relationships
            state['concept_hierarchy'] = concept_hierarchy
            
            # Log results
            state['processing_log'].append(
                f"✅ Combined extraction: {len(extracted_concepts)} concepts, "
                f"{len(concept_relationships)} relationships, "
                f"{len(concept_hierarchy)} hierarchy levels"
            )
            
            logger.info(f"✅ Combined extraction completed")
            logger.info(f"📝 {len(extracted_concepts)} concepts, "
                       f"{len(concept_relationships)} relationships, "
                       f"{len(concept_hierarchy)} hierarchy levels")
            
            # Log concept names
            concept_names = [c['name'] for c in extracted_concepts]
            logger.info(f"📝 Concepts: {', '.join(concept_names)}")
        else:
            error_msg = "Failed to parse combined extraction JSON"
            state['errors'].append(error_msg)
            state['extracted_concepts'] = []
            state['concept_relationships'] = []
            state['concept_hierarchy'] = []
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"Error in combined extraction: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


def extract_concepts_from_description(state: ConceptMapState) -> ConceptMapState:
    """
    Node 1: Extract key concepts directly from description text
    
    This node analyzes the user's description and extracts the most important concepts
    for educational concept mapping, with complexity scaled to description length.
    """
    logger.info(f"🔍 Extracting concepts from description ({len(state['description'])} chars)")
    
    try:
        # Analyze description complexity
        description_analysis = analyze_description_complexity(state['description'])
        
        # Adjust complexity for educational level
        base_complexity = description_analysis['complexity']
        adjusted_complexity = adjust_complexity_for_educational_level(base_complexity, state['educational_level'])
        
        # Auto-extract topic name if not provided
        if not state.get('topic_name'):
            state['topic_name'] = extract_topic_name_from_description(state['description'])
        
        # Store analysis results
        state['description_analysis'] = description_analysis
        state['complexity_config'] = adjusted_complexity
        
        # Log complexity decisions
        state['processing_log'].append(
            f"📊 Description analysis: {description_analysis['word_count']} words → "
            f"{adjusted_complexity['target_concepts']} concepts ({adjusted_complexity['detail_level']} level)"
        )
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Create dynamic prompt based on description analysis
        target_concepts = adjusted_complexity['target_concepts']
        detail_level = adjusted_complexity['detail_level']
        
        prompt = f"""
        You are an expert educational concept extraction system. Analyze the following description and extract the most important concepts for creating an educational concept map.
        
        DESCRIPTION TO ANALYZE:
        "{state['description']}"
        
        EXTRACTION PARAMETERS:
        - Educational Level: {state['educational_level']}
        - Target Concepts: Extract exactly {target_concepts} key concepts
        - Detail Level: {detail_level}
        - Description Length: {description_analysis['word_count']} words
        
        EXTRACTION RULES:
        1. Focus on NOUNS and KEY PROCESSES that are central to understanding the description
        2. Extract concepts that are explicitly mentioned or strongly implied in the description
        3. Prioritize concepts that can be meaningfully connected to each other
        4. Ensure concepts are appropriate for {state['educational_level']} level
        5. Include both concrete concepts and abstract principles mentioned in the description
        6. Avoid overly general terms unless they're domain-specific in this context
        
        CONCEPT TYPES TO CONSIDER:
        - Fundamental: Basic building blocks and prerequisites
        - Process: Actions, procedures, or mechanisms described
        - Application: Real-world uses or examples mentioned
        - Principle: Underlying rules or theories referenced
        
        OUTPUT FORMAT (JSON):
        {{
            "extracted_concepts": [
                {{
                    "name": "Concept Name (as mentioned or implied in description)",
                    "type": "fundamental|process|application|principle",
                    "importance": "high|medium|low",
                    "definition": "Brief definition based on the description context",
                    "mentioned_explicitly": true|false,
                    "educational_level": "appropriate level for this concept"
                }}
            ],
            "extraction_summary": {{
                "total_concepts": {target_concepts},
                "description_coverage": "percentage of description concepts captured",
                "confidence": "high|medium|low"
            }}
        }}
        
        EXAMPLE (for "Photosynthesis converts sunlight into energy in plant cells"):
        {{
            "extracted_concepts": [
                {{"name": "Photosynthesis", "type": "process", "importance": "high", "definition": "Process converting sunlight to energy", "mentioned_explicitly": true, "educational_level": "basic"}},
                {{"name": "Sunlight", "type": "fundamental", "importance": "high", "definition": "Light energy from the sun", "mentioned_explicitly": true, "educational_level": "basic"}},
                {{"name": "Energy Conversion", "type": "process", "importance": "high", "definition": "Transformation of one energy form to another", "mentioned_explicitly": true, "educational_level": "intermediate"}},
                {{"name": "Plant Cells", "type": "fundamental", "importance": "medium", "definition": "Basic structural units of plants", "mentioned_explicitly": true, "educational_level": "basic"}}
            ]
        }}
        
        Extract concepts that will create a meaningful, educational concept map based on this specific description.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Track token usage
        token_info = log_token_usage("extract_concepts_from_description", prompt, response_text)
        get_tracker().add_node("extract_concepts", token_info)
        
        # Parse JSON response with improved error handling
        json_text = clean_json_response(response_text)
        
        result = safe_json_parse(json_text, {})
        if result:
            extracted_concepts = result.get('extracted_concepts', [])
            
            state['extracted_concepts'] = extracted_concepts
            state['processing_log'].append(f"✅ Extracted {len(extracted_concepts)} concepts from description")
            
            logger.info(f"✅ Extracted {len(extracted_concepts)} concepts")
            
            # Log extracted concept names for visibility
            concept_names = [concept['name'] for concept in extracted_concepts]
            logger.info(f"📝 Concepts: {', '.join(concept_names)}")
        else:
            error_msg = f"Failed to parse extracted concepts JSON - using empty concepts"
            state['errors'].append(error_msg)
            state['extracted_concepts'] = []
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"Error in concept extraction: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


def analyze_concept_relationships(state: ConceptMapState) -> ConceptMapState:
    """
    Node 2: Identify relationships between extracted concepts
    
    This node analyzes the extracted concepts and determines how they relate to each other
    based on the original description context.
    """
    logger.info("🔗 Analyzing relationships between concepts")
    
    try:
        extracted_concepts = state.get('extracted_concepts', [])
        if not extracted_concepts:
            error_msg = "No extracted concepts found for relationship analysis"
            state['errors'].append(error_msg)
            logger.error(error_msg)
            return state
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        concept_names = [concept['name'] for concept in extracted_concepts]
        
        prompt = f"""
        You are an expert in educational concept mapping. Analyze the relationships between these concepts based on the original description context.
        
        ORIGINAL DESCRIPTION:
        "{state['description']}"
        
        EXTRACTED CONCEPTS:
        {json.dumps(extracted_concepts, indent=2)}
        
        TASK: Identify meaningful relationships between these concepts as they appear in the original description.
        
        RELATIONSHIP TYPES TO CONSIDER:
        - "enables" / "allows" - One concept makes another possible
        - "requires" / "needs" - One concept is prerequisite for another  
        - "produces" / "creates" - One concept generates another
        - "is_part_of" / "contains" - Hierarchical containment
        - "influences" / "affects" - One concept impacts another
        - "similar_to" / "contrasts_with" - Comparison relationships
        - "occurs_in" / "located_in" - Spatial/temporal relationships
        - "measured_by" / "quantified_by" - Measurement relationships
        
        OUTPUT FORMAT (JSON):
        {{
            "concept_relationships": [
                {{
                    "from_concept": "Source Concept Name",
                    "to_concept": "Target Concept Name", 
                    "relationship_type": "enables|requires|produces|is_part_of|influences|similar_to|occurs_in|measured_by",
                    "relationship_description": "Brief explanation of how they relate in this context",
                    "strength": "strong|medium|weak",
                    "bidirectional": true|false,
                    "evidence_in_description": "Quote or reference from original description"
                }}
            ],
            "relationship_summary": {{
                "total_relationships": "number",
                "most_connected_concept": "concept with most relationships",
                "relationship_density": "high|medium|low"
            }}
        }}
        
        GUIDELINES:
        1. Only create relationships that are supported by the original description
        2. Focus on educationally meaningful connections
        3. Avoid creating too many weak relationships
        4. Ensure relationships help explain the description content
        5. Consider the educational level: {state['educational_level']}
        
        Create relationships that will help students understand how these concepts work together based on the description.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Track token usage
        token_info = log_token_usage("analyze_concept_relationships", prompt, response_text)
        get_tracker().add_node("analyze_relationships", token_info)
        
        # Parse JSON response with improved error handling
        json_text = clean_json_response(response_text)
        
        result = safe_json_parse(json_text, {})
        if result:
            concept_relationships = result.get('concept_relationships', [])
            
            state['concept_relationships'] = concept_relationships
            state['processing_log'].append(f"✅ Identified {len(concept_relationships)} concept relationships")
            
            logger.info(f"✅ Identified {len(concept_relationships)} relationships")
        else:
            error_msg = f"Failed to parse concept relationships JSON - using empty relationships"
            state['errors'].append(error_msg)
            state['concept_relationships'] = []
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"Error in relationship analysis: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


def build_concept_hierarchy(state: ConceptMapState) -> ConceptMapState:
    """
    Node 3: Organize concepts into learning hierarchy
    
    This node creates a logical learning progression and hierarchy 
    from the extracted concepts and their relationships.
    """
    logger.info("🏗️ Building concept hierarchy")
    
    try:
        extracted_concepts = state.get('extracted_concepts', [])
        concept_relationships = state.get('concept_relationships', [])
        
        if not extracted_concepts:
            error_msg = "No extracted concepts found for hierarchy building"
            state['errors'].append(error_msg)
            logger.error(error_msg)
            return state
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert in educational curriculum design. Create a logical learning hierarchy from these concepts and relationships.
        
        ORIGINAL DESCRIPTION:
        "{state['description']}"
        
        EDUCATIONAL LEVEL: {state['educational_level']}
        
        EXTRACTED CONCEPTS:
        {json.dumps(extracted_concepts, indent=2)}
        
        CONCEPT RELATIONSHIPS:
        {json.dumps(concept_relationships, indent=2)}
        
        TASK: Organize these concepts into a clear learning hierarchy that shows the optimal order for teaching/learning.
        
        HIERARCHY PRINCIPLES:
        1. Prerequisites come before dependent concepts
        2. Fundamental concepts form the foundation
        3. Complex concepts build on simpler ones
        4. Application concepts come after theory
        5. Consider the educational level for appropriate grouping
        
        OUTPUT FORMAT (JSON):
        {{
            "concept_hierarchy": [
                {{
                    "level": 1,
                    "level_name": "Foundation|Building|Application|Advanced",
                    "level_description": "What students learn at this level",
                    "concepts": [
                        {{
                            "name": "Concept Name",
                            "justification": "Why this concept belongs at this level",
                            "prerequisites": ["prerequisite concept names"],
                            "enables": ["concepts this unlocks"]
                        }}
                    ],
                    "estimated_learning_time": "time to master this level",
                    "difficulty": "easy|moderate|challenging|advanced"
                }}
            ],
            "learning_path": [
                {{
                    "step": 1,
                    "focus": "What to teach/learn in this step",
                    "concepts_involved": ["concept names"],
                    "teaching_approach": "How to teach this step"
                }}
            ],
            "hierarchy_summary": {{
                "total_levels": "number of hierarchy levels",
                "foundational_concepts": ["most basic concepts"],
                "capstone_concepts": ["most advanced concepts"]
            }}
        }}
        
        GUIDELINES:
        - Create 2-5 hierarchy levels appropriate for the educational level
        - Ensure each level builds logically on previous levels  
        - Group related concepts within levels
        - Provide clear learning progression
        - Consider cognitive load for each level
        
        Create a hierarchy that optimizes learning based on the description content and educational level.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Track token usage
        token_info = log_token_usage("build_concept_hierarchy", prompt, response_text)
        get_tracker().add_node("build_hierarchy", token_info)
        
        # Parse JSON response with improved error handling
        json_text = clean_json_response(response_text)
        
        result = safe_json_parse(json_text, {})
        if result:
            concept_hierarchy = result.get('concept_hierarchy', [])
            
            state['concept_hierarchy'] = concept_hierarchy
            state['processing_log'].append(f"✅ Built {len(concept_hierarchy)}-level concept hierarchy")
            
            logger.info(f"✅ Built {len(concept_hierarchy)}-level hierarchy")
        else:
            error_msg = f"Failed to parse concept hierarchy JSON - using empty hierarchy"
            state['errors'].append(error_msg)
            state['concept_hierarchy'] = []
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"Error in hierarchy building: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


def enrich_with_educational_metadata(state: ConceptMapState) -> ConceptMapState:
    """
    Node 4: Add educational metadata and teaching strategies
    
    This node enriches the concept map with educational metadata,
    teaching strategies, learning objectives, and assessment methods.
    """
    logger.info("🎓 Enriching with educational metadata")
    
    try:
        extracted_concepts = state.get('extracted_concepts', [])
        concept_hierarchy = state.get('concept_hierarchy', [])
        complexity_config = state.get('complexity_config', {})
        
        if not extracted_concepts:
            error_msg = "No extracted concepts found for educational enrichment"
            state['errors'].append(error_msg)
            logger.error(error_msg)
            return state
        
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an expert educational content designer. Enrich this concept map with comprehensive educational metadata and teaching strategies.
        
        ORIGINAL DESCRIPTION:
        "{state['description']}"
        
        EDUCATIONAL LEVEL: {state['educational_level']}
        TARGET TOPIC: {state['topic_name']}
        
        EXTRACTED CONCEPTS:
        {json.dumps(extracted_concepts, indent=2)}
        
        CONCEPT HIERARCHY:
        {json.dumps(concept_hierarchy, indent=2)}
        
        COMPLEXITY CONFIG:
        {json.dumps(complexity_config, indent=2)}
        
        TASK: Create comprehensive educational enrichment for this concept map.
        
        OUTPUT FORMAT (JSON):
        {{
            "enriched_concepts": {{
                "Concept Name": {{
                    "definition": "Clear definition appropriate for educational level",
                    "difficulty_level": "easy|moderate|challenging|advanced",
                    "estimated_learning_time": "time to understand this concept",
                    "prerequisites": ["concepts needed before this"],
                    "learning_objectives": ["specific learning goals"],
                    "common_misconceptions": ["typical student errors"],
                    "teaching_strategies": ["effective teaching approaches"],
                    "assessment_methods": ["ways to test understanding"],
                    "real_world_examples": ["practical applications"],
                    "memory_aids": ["mnemonics or memory techniques"],
                    "extension_activities": ["ways to deepen understanding"]
                }}
            }},
            "overall_learning_objectives": [
                "After studying this concept map, students will be able to..."
            ],
            "teaching_sequence": [
                {{
                    "phase": "Introduction|Development|Application|Assessment",
                    "focus": "What to focus on in this phase",
                    "suggested_activities": ["specific teaching activities"],
                    "estimated_time": "time for this phase"
                }}
            ],
            "assessment_strategy": {{
                "formative_assessments": ["ongoing assessment methods"],
                "summative_assessments": ["final assessment methods"],
                "rubric_criteria": ["key evaluation criteria"]
            }},
            "differentiation_strategies": {{
                "for_struggling_learners": ["support strategies"],
                "for_advanced_learners": ["extension strategies"],
                "for_different_learning_styles": ["varied approaches"]
            }},
            "resources_and_materials": {{
                "required_materials": ["essential resources"],
                "recommended_readings": ["additional resources"],
                "digital_tools": ["helpful technology"],
                "hands_on_activities": ["practical activities"]
            }}
        }}
        
        GUIDELINES:
        - Tailor all content to {state['educational_level']} level
        - Ensure practical, actionable teaching strategies
        - Include age-appropriate examples and activities
        - Focus on deep understanding, not just memorization
        - Consider diverse learning styles and needs
        - Align with educational best practices
        
        Create comprehensive educational support for teaching this concept map effectively.
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Track token usage
        token_info = log_token_usage("enrich_with_educational_metadata", prompt, response_text)
        get_tracker().add_node("educational_enrichment", token_info)
        
        # Parse JSON response with improved error handling
        json_text = clean_json_response(response_text)
        
        result = safe_json_parse(json_text, {})
        if result:
            # Store enrichment data
            state['enriched_concepts'] = result.get('enriched_concepts', {})
            state['learning_objectives'] = result.get('overall_learning_objectives', [])
            state['teaching_strategies'] = result.get('teaching_sequence', [])
            
            state['processing_log'].append(f"✅ Added educational metadata for {len(state['enriched_concepts'])} concepts")
            
            logger.info(f"✅ Educational enrichment completed")
        else:
            error_msg = f"Failed to parse educational enrichment JSON - using empty enrichment"
            state['errors'].append(error_msg)
            state['enriched_concepts'] = {}
            state['learning_objectives'] = []
            state['teaching_strategies'] = []
            logger.error(error_msg)
            
    except Exception as e:
        error_msg = f"Error in educational enrichment: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


# Initialize legacy fields for backward compatibility
def initialize_legacy_fields(state: ConceptMapState) -> ConceptMapState:
    """Initialize legacy fields to maintain backward compatibility with existing code"""
    state['raw_subtopics'] = []
    state['subtopic_concepts'] = {}
    state['key_concepts_per_subtopic'] = {}
    state['subtopic_hierarchies'] = {}
    state['cross_subtopic_links'] = []
    state['enriched_subtopics'] = {}
    return state