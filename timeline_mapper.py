"""
Timeline Mapper Module
======================
Creates a timeline data structure mapping concepts to sentences for dynamic reveal.

Makes a SINGLE LLM API call with the full description to extract all concepts,
then uses simple heuristics to map concepts to sentences based on keyword occurrence.
"""

import re
import json
import logging
from typing import Dict, List, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles:
    - Sentences with no space after period (e.g., "Sentence1.Sentence2")
    - Titles like Mr., Mrs., Dr., etc.
    - Abbreviations like U.S., Ph.D., etc.
    
    Args:
        text: Input description text
        
    Returns:
        List of sentence strings
    """
    # First, protect common abbreviations and titles by temporarily replacing them
    # Store original positions for restoration
    protected_patterns = [
        (r'\bMr\.', 'MR_PLACEHOLDER'),
        (r'\bMrs\.', 'MRS_PLACEHOLDER'),
        (r'\bMs\.', 'MS_PLACEHOLDER'),
        (r'\bDr\.', 'DR_PLACEHOLDER'),
        (r'\bProf\.', 'PROF_PLACEHOLDER'),
        (r'\bSr\.', 'SR_PLACEHOLDER'),
        (r'\bJr\.', 'JR_PLACEHOLDER'),
        (r'\bU\.S\.', 'US_PLACEHOLDER'),
        (r'\bPh\.D\.', 'PHD_PLACEHOLDER'),
        (r'\bM\.D\.', 'MD_PLACEHOLDER'),
        (r'\bB\.A\.', 'BA_PLACEHOLDER'),
        (r'\bM\.A\.', 'MA_PLACEHOLDER'),
        (r'\bB\.Sc\.', 'BSC_PLACEHOLDER'),
        (r'\bM\.Sc\.', 'MSC_PLACEHOLDER'),
        (r'\betc\.', 'ETC_PLACEHOLDER'),
        (r'\bi\.e\.', 'IE_PLACEHOLDER'),
        (r'\be\.g\.', 'EG_PLACEHOLDER'),
    ]
    
    # Protect abbreviations and titles
    protected_text = text
    for pattern, placeholder in protected_patterns:
        protected_text = re.sub(pattern, placeholder, protected_text, flags=re.IGNORECASE)
    
    # Now split on sentence boundaries:
    # 1. Period/exclamation/question followed by space(s)
    # 2. Period/exclamation/question followed by capital letter (no space case)
    # 3. Period/exclamation/question at end of string
    sentences = re.split(r'([.!?])(?:\s+|(?=[A-Z])|$)', protected_text.strip())
    
    # Reconstruct sentences by pairing text with punctuation
    reconstructed = []
    i = 0
    while i < len(sentences):
        if i + 1 < len(sentences) and sentences[i + 1] in '.!?':
            # Pair text with its punctuation
            reconstructed.append(sentences[i] + sentences[i + 1])
            i += 2
        elif sentences[i].strip() and sentences[i] not in '.!?':
            # Text without punctuation (last sentence might not have punctuation)
            reconstructed.append(sentences[i])
            i += 1
        else:
            i += 1
    
    # Restore protected patterns
    final_sentences = []
    for sentence in reconstructed:
        restored = sentence
        for pattern, placeholder in protected_patterns:
            # Restore original text with proper casing
            if placeholder == 'US_PLACEHOLDER':
                original = 'U.S.'
            elif placeholder == 'PHD_PLACEHOLDER':
                original = 'Ph.D.'
            elif placeholder == 'MD_PLACEHOLDER':
                original = 'M.D.'
            elif placeholder == 'BA_PLACEHOLDER':
                original = 'B.A.'
            elif placeholder == 'MA_PLACEHOLDER':
                original = 'M.A.'
            elif placeholder == 'BSC_PLACEHOLDER':
                original = 'B.Sc.'
            elif placeholder == 'MSC_PLACEHOLDER':
                original = 'M.Sc.'
            elif placeholder == 'IE_PLACEHOLDER':
                original = 'i.e.'
            elif placeholder == 'EG_PLACEHOLDER':
                original = 'e.g.'
            elif placeholder == 'ETC_PLACEHOLDER':
                original = 'etc.'
            else:
                # For titles (Mr., Mrs., etc.), capitalize first letter
                original = placeholder.replace('_PLACEHOLDER', '').replace('_', '')
                original = original.capitalize() + '.'
            
            restored = restored.replace(placeholder, original)
        
        # Clean up and add if not empty
        restored = restored.strip()
        if restored:
            final_sentences.append(restored)
    
    return final_sentences


def estimate_tts_duration(sentence: str) -> float:
    """
    Estimate TTS duration based on word count.
    Assumes speaking rate of ~150 words per minute (0.4 seconds per word).
    
    Args:
        sentence: Input sentence text
        
    Returns:
        Estimated duration in seconds
    """
    word_count = len(sentence.split())
    # 150 wpm = 2.5 words per second = 0.4 seconds per word
    duration = word_count * 0.4
    # Add minimum duration of 1 second
    return max(duration, 1.0)


def map_concepts_to_sentences(
    concepts: List[Dict],
    relationships: List[Dict],
    sentences: List[str]
) -> Dict[int, Dict]:
    """
    Map concepts and relationships to sentences using simple heuristic:
    A concept belongs to the first sentence where its name appears.
    
    Args:
        concepts: List of concept dicts with 'name' keys
        relationships: List of relationship dicts
        sentences: List of sentence strings
        
    Returns:
        Dict mapping sentence index to {concepts, relationships}
    """
    sentence_map = {i: {"concepts": [], "relationships": []} for i in range(len(sentences))}
    
    # Map concepts to sentences
    for concept in concepts:
        concept_name = concept.get('name', '').lower()
        assigned = False
        
        # Find first sentence containing this concept name
        for idx, sentence in enumerate(sentences):
            if concept_name in sentence.lower():
                sentence_map[idx]["concepts"].append(concept)
                assigned = True
                break
        
        # If not found in any sentence, assign to first sentence
        if not assigned and sentences:
            sentence_map[0]["concepts"].append(concept)
    
    # Map relationships to sentences
    # Strategy: Assign relationship to the sentence where its "to" concept appears
    for relationship in relationships:
        to_concept = relationship.get('to', '').lower()
        assigned = False
        
        for idx, sentence in enumerate(sentences):
            if to_concept in sentence.lower():
                sentence_map[idx]["relationships"].append(relationship)
                assigned = True
                break
        
        # If not found, assign to first sentence
        if not assigned and sentences:
            sentence_map[0]["relationships"].append(relationship)
    
    return sentence_map


def extract_concepts_from_full_description(
    description: str,
    educational_level: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Make SINGLE LLM API call to extract all concepts and relationships
    from the full description at once.
    
    Args:
        description: Full description text
        educational_level: Educational level for context
        
    Returns:
        Tuple of (concepts_list, relationships_list)
    """
    logger.info("🔥 Making SINGLE API call to extract all concepts from full description...")
    
    # Use the optimized gemini-2.5-flash-lite model
    model = genai.GenerativeModel(
        'gemini-2.5-flash-lite',
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    # Compressed prompt for efficient extraction
    prompt = f"""Extract concepts and relationships from this description for {educational_level} level.

Description: {description}

Return ONLY valid JSON (no markdown, no explanation):
{{
  "concepts": [
    {{"name": "ConceptName", "type": "category", "importance": "high/medium/low", "definition": "brief definition"}}
  ],
  "relationships": [
    {{"from": "Concept1", "to": "Concept2", "relationship": "verb phrase"}}
  ]
}}

Rules:
- Extract 3-8 key concepts max
- Use clear, concise names
- Focus on core ideas only
- Ensure all relationship concepts exist in concepts list"""

    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean markdown code blocks if present
        if response_text.startswith('```'):
            response_text = re.sub(r'^```(?:json)?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        data = json.loads(response_text)
        concepts = data.get('concepts', [])
        relationships = data.get('relationships', [])
        
        logger.info(f"✅ API call complete: Extracted {len(concepts)} concepts, {len(relationships)} relationships")
        
        return concepts, relationships
        
    except Exception as e:
        logger.error(f"❌ Error extracting concepts: {e}")
        # Return minimal fallback data
        return [], []


def create_timeline(
    description: str,
    educational_level: str,
    topic_name: str
) -> Dict:
    """
    Create timeline data structure for dynamic concept map generation.
    
    This is the main entry point that:
    1. Makes SINGLE LLM API call with full description
    2. Splits description into sentences
    3. Maps concepts to sentences using heuristics
    4. Estimates TTS duration for each sentence
    
    Args:
        description: Full description text
        educational_level: Educational level (e.g., "High School")
        topic_name: Topic name for the concept map
        
    Returns:
        Timeline dict with structure:
        {
            "metadata": {
                "topic_name": str,
                "educational_level": str,
                "total_sentences": int,
                "total_concepts": int
            },
            "sentences": [
                {
                    "index": int,
                    "text": str,
                    "concepts": List[Dict],
                    "relationships": List[Dict],
                    "estimated_tts_duration": float
                }
            ]
        }
    """
    logger.info(f"🔄 Creating timeline for topic: {topic_name}")
    
    # Step 1: Split into sentences
    sentences = split_into_sentences(description)
    logger.info(f"📝 Split description into {len(sentences)} sentences")
    
    # Step 2: Extract ALL concepts with SINGLE API call
    concepts, relationships = extract_concepts_from_full_description(
        description, educational_level
    )
    
    # Step 3: Map concepts to sentences using heuristics
    sentence_map = map_concepts_to_sentences(concepts, relationships, sentences)
    
    # Step 4: Build timeline structure
    timeline_sentences = []
    for idx, sentence in enumerate(sentences):
        timeline_sentences.append({
            "index": idx,
            "text": sentence,
            "concepts": sentence_map[idx]["concepts"],
            "relationships": sentence_map[idx]["relationships"],
            "estimated_tts_duration": estimate_tts_duration(sentence)
        })
    
    timeline = {
        "metadata": {
            "topic_name": topic_name,
            "educational_level": educational_level,
            "total_sentences": len(sentences),
            "total_concepts": len(concepts)
        },
        "sentences": timeline_sentences
    }
    
    logger.info(f"✅ Timeline created! {len(sentences)} sentences, {len(concepts)} concepts")
    
    return timeline


def print_timeline_summary(timeline: Dict):
    """
    Print a human-readable summary of the timeline for debugging.
    
    Args:
        timeline: Timeline dict from create_timeline()
    """
    metadata = timeline["metadata"]
    print(f"\n{'='*60}")
    print(f"Timeline Summary: {metadata['topic_name']}")
    print(f"{'='*60}")
    print(f"Educational Level: {metadata['educational_level']}")
    print(f"Total Sentences: {metadata['total_sentences']}")
    print(f"Total Concepts: {metadata['total_concepts']}")
    print(f"{'='*60}\n")
    
    for sentence_data in timeline["sentences"]:
        print(f"Sentence {sentence_data['index']}: \"{sentence_data['text']}\"")
        print(f"  Concepts: {[c['name'] for c in sentence_data['concepts']]}")
        print(f"  Relationships: {len(sentence_data['relationships'])}")
        print(f"  Est. Duration: {sentence_data['estimated_tts_duration']:.1f}s")
        print()
