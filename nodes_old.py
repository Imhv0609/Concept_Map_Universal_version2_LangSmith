"""
Description-Based Concept Map Nodes

This module contains the 4-node processing workflow for description-based concept mapping
that extracts concepts directly from user descriptions (1 word to 3000+ words).
"""

import json
import logging
from typing import Dict, List, Any
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
from states import ConceptMapState
from description_analyzer import analyze_description_complexity, adjust_complexity_for_educational_level, extract_topic_name_from_description

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Configure Gemini
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


def analyze_topic_and_extract_subtopics(state: ConceptMapState) -> ConceptMapState:
    """
    Node 1: Analyze the given topic and extract relevant subtopics
    
    This node takes any topic/chapter and identifies the main subtopics
    that should be covered for comprehensive understanding.
    """
    logger.info(f"🔍 Analyzing topic: {state['topic_name']}")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        You are an educational expert analyzing topics for concept mapping.
        
        Topic: {state['topic_name']}
        Educational Level: {state['educational_level']}
        {f"Additional Context: {state['topic_description']}" if state['topic_description'] else ""}
        
        Your task is to identify 4-8 main subtopics that comprehensively cover this topic.
        These subtopics should be:
        1. Fundamental concepts that students need to understand
        2. Logically organized from basic to advanced
        3. Suitable for the specified educational level
        4. Comprehensive enough to cover the entire topic
        
        Please provide your response in the following JSON format:
        {{
            "subtopics": [
                {{
                    "name": "Subtopic Name",
                    "description": "Brief description of what this subtopic covers",
                    "importance": "High/Medium/Low",
                    "difficulty": "Beginner/Intermediate/Advanced",
                    "estimated_learning_time": "X minutes/hours"
                }}
            ]
        }}
        
        Focus on creating subtopics that are:
        - Conceptually distinct but interconnected
        - Appropriate for building concept maps
        - Educationally meaningful and practical
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse JSON response
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text
        
        try:
            result = json.loads(json_text)
            subtopics_data = result.get('subtopics', [])
            
            # Extract just the subtopic names for processing
            subtopic_names = [subtopic['name'] for subtopic in subtopics_data]
            
            state['raw_subtopics'] = subtopic_names
            state['processing_log'].append(f"✅ Identified {len(subtopic_names)} subtopics")
            
            logger.info(f"✅ Identified {len(subtopic_names)} subtopics")
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse subtopics JSON: {e}"
            state['errors'].append(error_msg)
            logger.error(error_msg)
            state['success'] = False
            
    except Exception as e:
        error_msg = f"Error in subtopic analysis: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


def generate_concepts_for_each_subtopic(state: ConceptMapState) -> ConceptMapState:
    """
    Node 2: For each subtopic, generate detailed concepts that should be covered
    
    This node takes each subtopic and generates specific concepts, terms,
    and ideas that students need to learn within that subtopic.
    """
    logger.info("🧠 Generating concepts for each subtopic")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        subtopic_concepts = {}
        
        for subtopic in state['raw_subtopics']:
            logger.info(f"   📝 Processing subtopic: {subtopic}")
            
            prompt = f"""
            You are an educational content expert creating detailed concept lists.
            
            Main Topic: {state['topic_name']}
            Subtopic: {subtopic}
            Educational Level: {state['educational_level']}
            
            Your task is to identify 6-12 specific concepts, terms, or ideas that students need to learn within this subtopic.
            These concepts should be:
            1. Specific and well-defined
            2. Appropriate for the educational level
            3. Essential for understanding the subtopic
            4. Suitable for creating concept maps (not too broad, not too narrow)
            
            Please provide your response in the following JSON format:
            {{
                "concepts": [
                    {{
                        "name": "Concept Name",
                        "definition": "Clear definition of the concept",
                        "importance": "High/Medium/Low",
                        "type": "Fundamental/Process/Application/Theory",
                        "keywords": ["related", "terms", "synonyms"]
                    }}
                ]
            }}
            
            Focus on concepts that are:
            - Clear and specific
            - Interconnected with other concepts
            - Educationally valuable
            - Mappable in a visual concept map
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            try:
                result = json.loads(json_text)
                concepts = result.get('concepts', [])
                subtopic_concepts[subtopic] = concepts
                
                logger.info(f"   ✅ Generated {len(concepts)} concepts for {subtopic}")
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse concepts JSON for {subtopic}: {e}"
                state['errors'].append(error_msg)
                logger.error(error_msg)
                subtopic_concepts[subtopic] = []
        
        state['subtopic_concepts'] = subtopic_concepts
        total_concepts = sum(len(concepts) for concepts in subtopic_concepts.values())
        state['processing_log'].append(f"✅ Generated {total_concepts} concepts across all subtopics")
        
        logger.info(f"✅ Generated concepts for {len(subtopic_concepts)} subtopics")
        
    except Exception as e:
        error_msg = f"Error in concept generation: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


def identify_key_concepts_per_subtopic(state: ConceptMapState) -> ConceptMapState:
    """
    Node 3: For each subtopic, identify the most important key concepts
    
    This node analyzes all concepts within each subtopic and identifies
    the 3-6 most critical concepts that form the foundation of that subtopic.
    """
    logger.info("🎯 Identifying key concepts for each subtopic")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        key_concepts_per_subtopic = {}
        
        for subtopic, concepts in state['subtopic_concepts'].items():
            if not concepts:
                continue
                
            logger.info(f"   🔍 Analyzing key concepts for: {subtopic}")
            
            concepts_text = "\n".join([f"- {concept['name']}: {concept['definition']}" for concept in concepts])
            
            prompt = f"""
            You are an educational expert identifying the most critical concepts for learning.
            
            Main Topic: {state['topic_name']}
            Subtopic: {subtopic}
            Educational Level: {state['educational_level']}
            
            All concepts in this subtopic:
            {concepts_text}
            
            Your task is to identify the 3-6 MOST IMPORTANT concepts from the list above that:
            1. Are absolutely essential for understanding this subtopic
            2. Serve as foundations for other concepts
            3. Are most frequently referenced or built upon
            4. Would be the core concepts a student MUST understand
            
            Please provide your response in the following JSON format:
            {{
                "key_concepts": [
                    {{
                        "name": "Concept Name (must match exactly from the list above)",
                        "priority": "Critical/High/Medium",
                        "reason": "Why this concept is key to understanding the subtopic",
                        "foundational_for": ["other concepts it supports"]
                    }}
                ]
            }}
            
            Select concepts that are:
            - Central to the subtopic's understanding
            - Prerequisites for other concepts
            - Most educationally significant
            - Essential for concept map visualization
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Parse JSON response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            try:
                result = json.loads(json_text)
                key_concepts = result.get('key_concepts', [])
                key_concepts_per_subtopic[subtopic] = key_concepts
                
                logger.info(f"   ✅ Identified {len(key_concepts)} key concepts for {subtopic}")
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse key concepts JSON for {subtopic}: {e}"
                state['errors'].append(error_msg)
                logger.error(error_msg)
                key_concepts_per_subtopic[subtopic] = []
        
        state['key_concepts_per_subtopic'] = key_concepts_per_subtopic
        total_key_concepts = sum(len(concepts) for concepts in key_concepts_per_subtopic.values())
        state['processing_log'].append(f"✅ Identified {total_key_concepts} key concepts across all subtopics")
        
        logger.info(f"✅ Identified key concepts for {len(key_concepts_per_subtopic)} subtopics")
        
    except Exception as e:
        error_msg = f"Error in key concept identification: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


def build_hierarchies_and_cross_links(state: ConceptMapState) -> ConceptMapState:
    """
    Node 4: Build hierarchical relationships within subtopics and cross-links between subtopics
    
    This node creates:
    1. Hierarchical relationships within each subtopic (parent-child concept relationships)
    2. Cross-links between concepts across different subtopics
    """
    logger.info("🏗️ Building hierarchies and cross-links")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Build hierarchies within each subtopic
        subtopic_hierarchies = {}
        for subtopic, key_concepts in state['key_concepts_per_subtopic'].items():
            if len(key_concepts) < 2:
                subtopic_hierarchies[subtopic] = []
                continue
                
            logger.info(f"   🔗 Building hierarchy for: {subtopic}")
            
            concepts_text = "\n".join([f"- {concept['name']}: {concept['reason']}" for concept in key_concepts])
            
            prompt = f"""
            You are an educational expert creating learning hierarchies.
            
            Subtopic: {subtopic}
            Key concepts in this subtopic:
            {concepts_text}
            
            Your task is to create hierarchical relationships between these concepts that show:
            1. Which concepts are prerequisites for others
            2. Which concepts build upon others
            3. The logical learning sequence
            
            Please provide your response in the following JSON format:
            {{
                "relationships": [
                    {{
                        "parent": "Parent Concept Name",
                        "child": "Child Concept Name", 
                        "relationship_type": "prerequisite_for/builds_upon/component_of/leads_to",
                        "strength": "strong/medium/weak",
                        "description": "Brief explanation of the relationship"
                    }}
                ]
            }}
            
            Create relationships that are:
            - Educationally logical
            - Clear learning progressions
            - Meaningful for concept map visualization
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            try:
                result = json.loads(json_text)
                relationships = result.get('relationships', [])
                subtopic_hierarchies[subtopic] = relationships
                
                logger.info(f"   ✅ Built {len(relationships)} relationships for {subtopic}")
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse hierarchy JSON for {subtopic}: {e}"
                state['errors'].append(error_msg)
                logger.error(error_msg)
                subtopic_hierarchies[subtopic] = []
        
        state['subtopic_hierarchies'] = subtopic_hierarchies
        
        # Build cross-links between subtopics
        logger.info("   🌉 Building cross-subtopic links")
        
        all_subtopics_text = ""
        for subtopic, key_concepts in state['key_concepts_per_subtopic'].items():
            concepts_list = ", ".join([concept['name'] for concept in key_concepts])
            all_subtopics_text += f"\n{subtopic}: {concepts_list}"
        
        prompt = f"""
        You are an educational expert identifying connections between different subtopics.
        
        Main Topic: {state['topic_name']}
        All subtopics and their key concepts:
        {all_subtopics_text}
        
        Your task is to identify meaningful cross-links between concepts from DIFFERENT subtopics that show:
        1. How concepts from one subtopic relate to concepts in another subtopic
        2. Interdisciplinary connections
        3. Applied relationships across subtopics
        
        Please provide your response in the following JSON format:
        {{
            "cross_links": [
                {{
                    "concept1": "Concept from Subtopic A",
                    "subtopic1": "Subtopic A Name",
                    "concept2": "Concept from Subtopic B", 
                    "subtopic2": "Subtopic B Name",
                    "relationship": "how they relate",
                    "strength": "strong/medium/weak",
                    "description": "explanation of the cross-subtopic connection"
                }}
            ]
        }}
        
        Focus on connections that are:
        - Educationally meaningful
        - Help students see the bigger picture
        - Show practical applications across subtopics
        """
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            json_text = response_text[json_start:json_end].strip()
        else:
            json_text = response_text
        
        try:
            result = json.loads(json_text)
            cross_links = result.get('cross_links', [])
            state['cross_subtopic_links'] = cross_links
            
            logger.info(f"   ✅ Built {len(cross_links)} cross-subtopic links")
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse cross-links JSON: {e}"
            state['errors'].append(error_msg)
            logger.error(error_msg)
            state['cross_subtopic_links'] = []
        
        total_relationships = sum(len(relations) for relations in subtopic_hierarchies.values())
        state['processing_log'].append(f"✅ Built {total_relationships} hierarchical relationships and {len(cross_links)} cross-links")
        
        logger.info("✅ Completed hierarchy and cross-link building")
        
    except Exception as e:
        error_msg = f"Error in hierarchy building: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state


def enrich_subtopics_with_educational_metadata(state: ConceptMapState) -> ConceptMapState:
    """
    Node 5: Enrich each subtopic with educational metadata for teaching
    
    This node adds comprehensive educational information for each subtopic including
    difficulty levels, learning objectives, assessment suggestions, and teaching tips.
    """
    logger.info("✨ Enriching subtopics with educational metadata")
    
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        enriched_subtopics = {}
        
        for subtopic in state['raw_subtopics']:
            logger.info(f"   📚 Enriching subtopic: {subtopic}")
            
            # Get concepts for this subtopic
            concepts = state['subtopic_concepts'].get(subtopic, [])
            key_concepts = state['key_concepts_per_subtopic'].get(subtopic, [])
            
            concepts_text = "\n".join([f"- {concept['name']}" for concept in concepts])
            key_concepts_text = "\n".join([f"- {concept['name']}" for concept in key_concepts])
            
            prompt = f"""
            You are an educational curriculum expert creating comprehensive teaching metadata.
            
            Main Topic: {state['topic_name']}
            Subtopic: {subtopic}
            Educational Level: {state['educational_level']}
            
            All concepts in this subtopic:
            {concepts_text}
            
            Key concepts:
            {key_concepts_text}
            
            Your task is to create comprehensive educational metadata for this subtopic.
            
            Please provide your response in the following JSON format:
            {{
                "educational_metadata": {{
                    "difficulty_level": "Beginner/Intermediate/Advanced",
                    "estimated_learning_time": "X hours/days",
                    "importance_score": 1-10,
                    "learning_objectives": [
                        "Specific learning objective 1",
                        "Specific learning objective 2"
                    ],
                    "prerequisites": [
                        "Required prior knowledge or subtopic"
                    ],
                    "common_misconceptions": [
                        "Common student misconception 1",
                        "Common student misconception 2"
                    ],
                    "teaching_strategies": [
                        "Effective teaching approach 1",
                        "Effective teaching approach 2"
                    ],
                    "assessment_methods": [
                        "Suggested assessment method 1",
                        "Suggested assessment method 2"
                    ],
                    "real_world_applications": [
                        "Practical application 1",
                        "Practical application 2"
                    ],
                    "memory_aids": [
                        "Mnemonic or memory technique 1",
                        "Mnemonic or memory technique 2"
                    ],
                    "extension_activities": [
                        "Advanced activity or project 1",
                        "Advanced activity or project 2"
                    ]
                }}
            }}
            
            Make the metadata:
            - Practical and actionable for teachers
            - Age-appropriate for the educational level
            - Comprehensive yet concise
            - Educationally sound and research-based
            """
            
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            try:
                result = json.loads(json_text)
                metadata = result.get('educational_metadata', {})
                enriched_subtopics[subtopic] = metadata
                
                logger.info(f"   ✅ Enriched {subtopic}")
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse enrichment JSON for {subtopic}: {e}"
                state['errors'].append(error_msg)
                logger.error(error_msg)
                enriched_subtopics[subtopic] = {}
        
        state['enriched_subtopics'] = enriched_subtopics
        state['processing_log'].append(f"✅ Enriched {len(enriched_subtopics)} subtopics with educational metadata")
        
        logger.info(f"✅ Enriched {len(enriched_subtopics)} subtopics")
        
    except Exception as e:
        error_msg = f"Error in subtopic enrichment: {e}"
        state['errors'].append(error_msg)
        logger.error(error_msg)
        state['success'] = False
    
    return state