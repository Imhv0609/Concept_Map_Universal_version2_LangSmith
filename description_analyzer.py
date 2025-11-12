"""
Description Analysis Module for Universal Concept Map Teaching Agent

This module analyzes description text to determine optimal concept map complexity
and extracts key concepts directly from the user's description.
"""

import re
import math
from typing import Dict, List, Any, Tuple


def analyze_description_complexity(description: str) -> Dict[str, Any]:
    """
    Analyze description text and determine optimal concept map complexity
    Uses logarithmic scaling to prevent exponential growth
    
    Args:
        description (str): The user's description text (1 word to 3000+ words)
        
    Returns:
        Dict containing analysis results and complexity configuration
    """
    
    # Clean and count words
    words = re.findall(r'\b\w+\b', description.lower())
    word_count = len(words)
    
    # Calculate unique words for additional insight
    unique_words = len(set(words))
    
    # Estimate sentence complexity
    sentences = re.split(r'[.!?]+', description)
    sentence_count = len([s for s in sentences if s.strip()])
    avg_words_per_sentence = word_count / max(sentence_count, 1)
    
    # Logarithmic scaling for concept map complexity - REDUCED TO HALF
    if word_count <= 5:  # Single word to short phrase
        complexity = {
            "target_concepts": 2,  # Reduced from 3 to 2 (minimum for visualization)
            "target_subtopics": 1,
            "detail_level": "basic",
            "connection_density": "low",
            "educational_depth": "introductory"
        }
    elif word_count <= 20:  # Very short description (one sentence)
        # More conservative scaling for very short text
        base_concepts = 1.5 + math.log2(word_count) * 0.4  # Halved: 3→1.5, 0.8→0.4
        complexity = {
            "target_concepts": int(max(2, min(base_concepts, 3))),  # Cap reduced from 5 to 3 (min 2)
            "target_subtopics": 1,
            "detail_level": "basic",
            "connection_density": "low",
            "educational_depth": "basic"
        }
    elif word_count <= 50:  # Short description (1-2 sentences)
        base_concepts = 2 + math.log2(word_count) * 0.6  # Halved: 4→2, 1.2→0.6
        complexity = {
            "target_concepts": int(max(2, min(base_concepts, 4))),  # Cap reduced from 8 to 4 (min 2)
            "target_subtopics": 2,
            "detail_level": "moderate",
            "connection_density": "medium",
            "educational_depth": "basic"
        }
    elif word_count <= 200:  # Medium description (paragraph)
        base_concepts = 3 + math.log10(word_count) * 2  # Halved: 6→3, 4→2
        complexity = {
            "target_concepts": int(max(2, min(base_concepts, 8))),  # Cap reduced from 15 to 8 (min 2)
            "target_subtopics": 3,
            "detail_level": "detailed",
            "connection_density": "high",
            "educational_depth": "comprehensive"
        }
    elif word_count <= 1000:  # Long description (multiple paragraphs)
        base_concepts = 5 + math.log10(word_count) * 1.75  # Halved: 10→5, 3.5→1.75
        complexity = {
            "target_concepts": int(max(2, min(base_concepts, 10))),  # Cap reduced from 20 to 10 (min 2)
            "target_subtopics": 4,
            "detail_level": "comprehensive",
            "connection_density": "very_high",
            "educational_depth": "advanced"
        }
    else:  # Very long description (academic/research level)
        base_concepts = 7.5 + math.log10(word_count) * 1.25  # Halved: 15→7.5, 2.5→1.25
        complexity = {
            "target_concepts": int(max(2, min(base_concepts, 13))),  # Cap reduced from 25 to 13 (min 2)
            "target_subtopics": 5,
            "detail_level": "expert",
            "connection_density": "maximum",
            "educational_depth": "expert"
        }
    
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "sentence_count": sentence_count,
        "avg_words_per_sentence": round(avg_words_per_sentence, 1),
        "complexity": complexity,
        "scaling_factor": min(word_count / 100, 3.0)  # For fine-tuning, capped at 3x
    }


def extract_topic_name_from_description(description: str) -> str:
    """
    Auto-extract a topic name from the description if not provided
    
    Args:
        description (str): The user's description text
        
    Returns:
        str: Extracted topic name
    """
    
    # For very short descriptions, use the description itself
    if len(description.split()) <= 3:
        return description.title()
    
    # For longer descriptions, try to extract the main subject
    # This is a simple approach - could be enhanced with NLP
    words = description.split()
    
    # Look for capitalized words (potential proper nouns)
    capitalized = [word for word in words if word[0].isupper() and len(word) > 2]
    if capitalized:
        return capitalized[0]
    
    # Look for scientific/technical terms (words longer than 6 characters)
    technical_terms = [word for word in words if len(word) > 6 and word.isalpha()]
    if technical_terms:
        return technical_terms[0].title()
    
    # Fallback: use first few words
    return " ".join(words[:3]).title()


def adjust_complexity_for_educational_level(complexity: Dict[str, Any], educational_level: str) -> Dict[str, Any]:
    """
    Adjust the complexity configuration based on educational level
    
    Args:
        complexity (Dict): Base complexity from description analysis
        educational_level (str): Target educational level
        
    Returns:
        Dict: Adjusted complexity configuration
    """
    
    # Define education level modifiers
    level_modifiers = {
        "elementary": {
            "concept_multiplier": 0.6,
            "max_concepts": 8,
            "force_basic": True
        },
        "middle school": {
            "concept_multiplier": 0.8,
            "max_concepts": 12,
            "force_basic": False
        },
        "high school": {
            "concept_multiplier": 1.0,
            "max_concepts": 18,
            "force_basic": False
        },
        "undergraduate": {
            "concept_multiplier": 1.2,
            "max_concepts": 22,
            "force_basic": False
        },
        "graduate": {
            "concept_multiplier": 1.4,
            "max_concepts": 25,
            "force_basic": False
        },
        "professional": {
            "concept_multiplier": 1.3,
            "max_concepts": 25,
            "force_basic": False
        },
        "general audience": {
            "concept_multiplier": 0.9,
            "max_concepts": 15,
            "force_basic": False
        }
    }
    
    modifier = level_modifiers.get(educational_level.lower(), level_modifiers["high school"])
    
    # Adjust target concepts
    adjusted_concepts = int(complexity["target_concepts"] * modifier["concept_multiplier"])
    complexity["target_concepts"] = min(adjusted_concepts, modifier["max_concepts"])
    
    # Force basic detail level for elementary
    if modifier["force_basic"]:
        complexity["detail_level"] = "basic"
        complexity["connection_density"] = "low"
        complexity["educational_depth"] = "introductory"
    
    # Ensure minimum concepts
    complexity["target_concepts"] = max(complexity["target_concepts"], 3)
    
    return complexity


def get_complexity_summary(analysis: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of the complexity analysis
    
    Args:
        analysis (Dict): Analysis results from analyze_description_complexity
        
    Returns:
        str: Human-readable summary
    """
    
    word_count = analysis["word_count"]
    complexity = analysis["complexity"]
    
    if word_count <= 5:
        length_desc = "Very short"
    elif word_count <= 50:
        length_desc = "Short"
    elif word_count <= 200:
        length_desc = "Medium"
    elif word_count <= 1000:
        length_desc = "Long"
    else:
        length_desc = "Very long"
    
    return f"""{length_desc} description ({word_count} words) → {complexity['target_concepts']} concepts
Detail level: {complexity['detail_level']}
Connection density: {complexity['connection_density']}"""


# Example usage and testing
if __name__ == "__main__":
    # Test with different description lengths
    test_descriptions = [
        "Photosynthesis",
        "How plants make food using sunlight",
        "Photosynthesis is the process by which plants convert light energy into chemical energy using chloroplasts and chlorophyll to produce glucose from carbon dioxide and water.",
        """Photosynthesis is a complex biological process that occurs in plants, algae, and some bacteria. 
        It involves two main stages: the light-dependent reactions and the light-independent reactions (Calvin cycle). 
        During the light reactions, chlorophyll molecules in the thylakoids absorb photons and use this energy to 
        split water molecules, releasing oxygen and producing ATP and NADPH. The Calvin cycle then uses these 
        energy carriers along with carbon dioxide to synthesize glucose through a series of enzymatic reactions."""
    ]
    
    for i, desc in enumerate(test_descriptions, 1):
        print(f"\n=== Test {i} ===")
        print(f"Description: {desc[:100]}{'...' if len(desc) > 100 else ''}")
        analysis = analyze_description_complexity(desc)
        print(get_complexity_summary(analysis))
        
        # Test with different educational levels
        for level in ["elementary", "high school", "graduate"]:
            adjusted = adjust_complexity_for_educational_level(analysis["complexity"].copy(), level)
            print(f"  {level}: {adjusted['target_concepts']} concepts ({adjusted['detail_level']})")