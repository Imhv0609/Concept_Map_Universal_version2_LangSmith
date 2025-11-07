"""
Timeline Mapper Module
======================
Creates a timeline data structure mapping concepts to sentences for dynamic reveal.

Makes a SINGLE LLM API call with the full description to extract all concepts,
then uses simple heuristics to map concepts to sentences based on keyword occurrence.
"""

import os
import re
import json
import logging
from typing import Dict, List, Tuple
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from description_analyzer import (
    analyze_description_complexity,
    adjust_complexity_for_educational_level
)

logger = logging.getLogger(__name__)

# Configure Google Generative AI with API key from environment
api_key = os.getenv('GOOGLE_API_KEY')
if api_key:
    genai.configure(api_key=api_key)
    logger.info("✅ Google Generative AI configured with API key")
else:
    logger.warning("⚠️ GOOGLE_API_KEY not found in environment variables")


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


def estimate_tts_duration(sentence: str, words_per_second: float = 2.86) -> float:
    """
    Estimate TTS duration based on word count.
    Default: ~172 words per minute (0.35 seconds per word).
    
    Args:
        sentence: Input sentence text
        words_per_second: Speaking rate (default ~2.86 words/sec = 0.35s per word)
        
    Returns:
        Estimated duration in seconds
    """
    word_count = len(sentence.split())
    duration = word_count / words_per_second
    # Add minimum duration of 0.5 seconds
    return max(duration, 0.5)


def calculate_word_timings(text: str, speaking_rate: float = 0.35) -> List[Dict]:
    """
    Calculate timestamp for each word in text.
    
    Args:
        text: Full text (can be single sentence or multiple sentences merged)
        speaking_rate: Seconds per word (default 0.35s = ~172 wpm)
        
    Returns:
        List of dicts: [{"word": str, "start_time": float, "end_time": float}, ...]
    """
    words = text.split()
    word_timings = []
    
    current_time = 0.0
    for word in words:
        word_duration = speaking_rate
        
        # Adjust duration for punctuation (add slight pause for gTTS)
        # gTTS adds natural pauses, so we account for them here
        if word.endswith(('.', '!', '?')):
            word_duration += 0.4  # 400ms pause after sentence (gTTS adds natural pause)
        elif word.endswith((',', ';', ':')):
            word_duration += 0.2  # 200ms pause after clause
        
        word_timings.append({
            "word": word,
            "start_time": current_time,
            "end_time": current_time + word_duration
        })
        
        current_time += word_duration
    
    return word_timings


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


def assign_concept_reveal_times(
    concepts: List[Dict],
    word_timings: List[Dict],
    full_text: str
) -> List[Dict]:
    """
    Assign reveal_time to each concept based on when its last word is spoken.
    
    Args:
        concepts: List of concept dicts with 'name' keys
        word_timings: List of word timing dicts from calculate_word_timings()
        full_text: Full merged text to search for concepts
        
    Returns:
        List of concepts with added 'reveal_time' field
    """
    full_text_lower = full_text.lower()
    
    for concept in concepts:
        concept_name = concept.get('name', '')
        if not concept_name:
            concept['reveal_time'] = 0.0
            continue
        
        concept_name_lower = concept_name.lower()
        
        # Find the position of the concept in the full text
        try:
            concept_position = full_text_lower.index(concept_name_lower)
        except ValueError:
            # Concept not found in text - try finding individual words
            logger.warning(f"Concept '{concept_name}' not found exactly in text, trying word-by-word match")
            
            # Try to find any word from the concept (or word stems)
            concept_words = concept_name_lower.split()
            last_word_found_index = -1
            
            for word in concept_words:
                # Clean the word (remove punctuation)
                clean_word = re.sub(r'[^\w\s]', '', word)
                if not clean_word or len(clean_word) < 3:  # Skip very short words
                    continue
                
                # Try exact match first
                if clean_word in full_text_lower:
                    word_position = full_text_lower.index(clean_word)
                    words_before = full_text[:word_position].split()
                    word_index = len(words_before)
                    last_word_found_index = max(last_word_found_index, word_index)
                    continue
                
                # Try finding words that start with this stem (e.g., "evapor" matches "evaporates")
                # Use first 5 characters as stem
                word_stem = clean_word[:min(5, len(clean_word))]
                text_words = full_text_lower.split()
                
                for i, text_word in enumerate(text_words):
                    clean_text_word = re.sub(r'[^\w\s]', '', text_word)
                    if clean_text_word.startswith(word_stem):
                        last_word_found_index = max(last_word_found_index, i)
                        logger.debug(f"     → Matched '{clean_word}' to '{clean_text_word}' at word index {i}")
                        break
            
            if last_word_found_index >= 0 and last_word_found_index < len(word_timings):
                concept['reveal_time'] = word_timings[last_word_found_index]['end_time']
                logger.info(f"Concept '{concept_name}' matched at word index {last_word_found_index}, reveal_time: {concept['reveal_time']:.2f}s")
            else:
                # Still not found, distribute evenly
                concept_index = concepts.index(concept)
                total_duration = word_timings[-1]['end_time'] if word_timings else 1.0
                concept['reveal_time'] = (concept_index / len(concepts)) * total_duration
                logger.warning(f"Concept '{concept_name}' not found in text, distributing evenly at {concept['reveal_time']:.2f}s")
            continue
        
        # Get the words that make up this concept
        concept_words = concept_name.split()
        
        # Find the last word of the concept in word_timings
        # Strategy: Count words from start of text until we reach concept position
        words_before_concept = full_text[:concept_position].split()
        word_index_of_concept_start = len(words_before_concept)
        word_index_of_concept_end = word_index_of_concept_start + len(concept_words) - 1
        
        # Get timing of last word
        if word_index_of_concept_end < len(word_timings):
            concept['reveal_time'] = word_timings[word_index_of_concept_end]['end_time']
            logger.info(f"✓ Concept '{concept_name}' found at position {concept_position}, word index {word_index_of_concept_end}, reveal_time: {concept['reveal_time']:.2f}s")
        else:
            # Fallback: use last available timing
            concept['reveal_time'] = word_timings[-1]['end_time'] if word_timings else 0.0
            logger.warning(f"Concept '{concept_name}' word index {word_index_of_concept_end} out of bounds (max {len(word_timings)}), using fallback time {concept['reveal_time']:.2f}s")
    
    return concepts


def extract_concepts_from_full_description(
    description: str,
    educational_level: str
) -> Tuple[List[Dict], List[Dict]]:
    """
    Make SINGLE LLM API call to extract all concepts and relationships
    from the full description at once.
    
    Uses description_analyzer.py to dynamically scale concept count based on word count.
    
    Args:
        description: Full description text
        educational_level: Educational level for context
        
    Returns:
        Tuple of (concepts_list, relationships_list)
    """
    logger.info("🔥 Making SINGLE API call to extract all concepts from full description...")
    
    # Analyze description complexity to determine target concept count
    description_analysis = analyze_description_complexity(description)
    base_complexity = description_analysis['complexity']
    adjusted_complexity = adjust_complexity_for_educational_level(base_complexity, educational_level)
    
    target_concepts = adjusted_complexity['target_concepts']
    detail_level = adjusted_complexity['detail_level']
    word_count = description_analysis['word_count']
    
    logger.info(f"📊 Description analysis: {word_count} words → {target_concepts} concepts ({detail_level} level)")
    
    # Use the optimized gemini-2.5-flash-lite model with deterministic output
    generation_config = genai.GenerationConfig(
        temperature=0.0,  # Deterministic output for consistent results
        top_p=0.95,
        top_k=40,
        max_output_tokens=2048,
    )
    
    model = genai.GenerativeModel(
        'gemini-2.5-flash-lite',
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    
    # Dynamic prompt based on description analysis (matching nodes.py approach)
    prompt = f"""Extract concepts and relationships from this description for {educational_level} level.

Description: {description}

EXTRACTION PARAMETERS:
- Target Concepts: {target_concepts} (based on {word_count} words)
- Detail Level: {detail_level}
- Educational Level: {educational_level}

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
- Extract exactly {target_concepts} key concepts
- Create meaningful relationships between concepts
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
        
        # Log detailed information
        if concepts:
            logger.info(f"   Concepts: {[c.get('name', 'N/A') for c in concepts]}")
        else:
            logger.warning("   ⚠️ No concepts extracted!")
            
        if relationships:
            logger.info(f"   Relationships: {[(r.get('from', '?'), r.get('relationship', '?'), r.get('to', '?')) for r in relationships]}")
        else:
            logger.warning("   ⚠️ No relationships extracted! Graph will have no edges.")
        
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
    2. Merges all sentences into continuous text
    3. Calculates word-level timings
    4. Assigns reveal_time to each concept based on last word timing
    
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
                "total_duration": float,
                "total_concepts": int,
                "speaking_rate": float
            },
            "full_text": str,
            "word_timings": List[Dict],
            "concepts": List[Dict] (with reveal_time),
            "relationships": List[Dict]
        }
    """
    logger.info(f"🔄 Creating continuous timeline for topic: {topic_name}")
    
    # Step 1: Split into sentences (for grammatical correctness check)
    sentences = split_into_sentences(description)
    logger.info(f"📝 Split description into {len(sentences)} sentences")
    
    # Step 2: Merge sentences back into continuous text (spaces preserved)
    # TTS engines handle sentence punctuation naturally
    full_text = " ".join(sentences)
    logger.info(f"📝 Merged into continuous text ({len(full_text)} chars)")
    
    # Step 3: Extract ALL concepts with SINGLE API call
    concepts, relationships = extract_concepts_from_full_description(
        description, educational_level
    )
    
    # Step 4: Calculate word-level timings
    # gTTS speaking rate: ~150-160 WPM = 0.375-0.40s per word
    # Using 0.40s per word for better sync with gTTS (slightly slower than Edge-TTS's 0.35s)
    speaking_rate = 0.40  # seconds per word - tuned for gTTS
    word_timings = calculate_word_timings(full_text, speaking_rate)
    total_duration = word_timings[-1]['end_time'] if word_timings else 0.0
    logger.info(f"⏱️ Calculated timings for {len(word_timings)} words (total: {total_duration:.1f}s)")
    
    # Step 5: Assign reveal_time to each concept
    concepts = assign_concept_reveal_times(concepts, word_timings, full_text)
    logger.info(f"✅ Assigned reveal times to {len(concepts)} concepts")
    
    # CRITICAL: Force the first concept to appear immediately at time 0
    # This ensures the visualization starts right away and doesn't have a delay
    if concepts and len(concepts) > 0:
        # Find the concept with earliest reveal time and set it to 0
        earliest_concept = min(concepts, key=lambda c: c.get('reveal_time', 0.0))
        original_time = earliest_concept.get('reveal_time', 0.0)
        earliest_concept['reveal_time'] = 0.0
        logger.info(f"⚡ Forced first concept '{earliest_concept.get('name')}' to appear at 0.0s (was {original_time:.2f}s)")
    
    timeline = {
        "metadata": {
            "topic_name": topic_name,
            "educational_level": educational_level,
            "total_duration": total_duration,
            "total_concepts": len(concepts),
            "speaking_rate": speaking_rate,
            "word_count": len(word_timings)
        },
        "full_text": full_text,
        "word_timings": word_timings,
        "concepts": concepts,
        "relationships": relationships,
        # Keep legacy sentence structure for backward compatibility
        "sentences": [{
            "index": 0,
            "text": full_text,
            "concepts": concepts,
            "relationships": relationships,
            "estimated_tts_duration": total_duration
        }]
    }
    
    logger.info(f"✅ Continuous timeline created! {total_duration:.1f}s duration, {len(concepts)} concepts")
    
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
