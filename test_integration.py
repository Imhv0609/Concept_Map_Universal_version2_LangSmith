"""
Test the integrated description_analyzer logic in timeline_mapper
"""
import os
from dotenv import load_dotenv
from timeline_mapper import extract_concepts_from_full_description
from description_analyzer import analyze_description_complexity, adjust_complexity_for_educational_level

# Load environment variables
load_dotenv()

# Make sure API key is set
if not os.getenv('GOOGLE_API_KEY'):
    print("❌ GOOGLE_API_KEY not found in environment!")
    print("Please run: export GOOGLE_API_KEY='your_key_here'")
    exit(1)

# Test description
description = """The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow in clouds, and falls again to the surface as precipitation. The cycling of water in and out of the atmosphere is a significant aspect of the weather patterns on Earth."""

educational_level = "High School"

print("="*70)
print("TESTING INTEGRATED LOGIC")
print("="*70)

# Step 1: Check description analysis
print("\n📊 STEP 1: Description Analysis")
print("-"*70)
analysis = analyze_description_complexity(description)
base_complexity = analysis['complexity']
adjusted = adjust_complexity_for_educational_level(base_complexity, educational_level)

print(f"Word Count: {analysis['word_count']}")
print(f"Unique Words: {analysis['unique_words']}")
print(f"Sentences: {analysis['sentence_count']}")
print(f"Detail Level: {adjusted['detail_level']}")
print(f"Target Concepts: {adjusted['target_concepts']} ← This should be used in prompt")
print(f"Connection Density: {adjusted['connection_density']}")

# Step 2: Test concept extraction with integrated logic
print("\n🔥 STEP 2: Extracting Concepts with Integrated Logic")
print("-"*70)
concepts, relationships = extract_concepts_from_full_description(description, educational_level)

# Step 3: Display results
print(f"\n✅ EXTRACTION COMPLETE")
print("="*70)

print(f"\n📝 CONCEPTS ({len(concepts)}):")
print("-"*70)
for i, concept in enumerate(concepts, 1):
    name = concept.get('name', 'N/A')
    concept_type = concept.get('type', 'N/A')
    importance = concept.get('importance', 'N/A')
    definition = concept.get('definition', 'N/A')
    print(f"\n{i}. {name}")
    print(f"   Type: {concept_type}")
    print(f"   Importance: {importance}")
    print(f"   Definition: {definition[:80]}...")

print(f"\n🔗 RELATIONSHIPS ({len(relationships)}):")
print("-"*70)
for i, rel in enumerate(relationships, 1):
    from_node = rel.get('from', 'N/A')
    to_node = rel.get('to', 'N/A')
    rel_type = rel.get('relationship', 'N/A')
    print(f"{i}. {from_node} --[{rel_type}]--> {to_node}")

# Step 4: Validation
print(f"\n🔍 VALIDATION:")
print("-"*70)
print(f"Expected concepts: {adjusted['target_concepts']}")
print(f"Actual concepts: {len(concepts)}")
print(f"Match: {'✅' if len(concepts) == adjusted['target_concepts'] else '❌'}")
print(f"\nRelationships: {len(relationships)}")
print(f"Density: {'✅ Good' if len(relationships) >= len(concepts) - 2 else '⚠️ Low'}")

# Step 5: Check graph connectivity
print(f"\n📊 GRAPH STRUCTURE:")
print("-"*70)
concept_names = set([c.get('name') for c in concepts])
valid_relationships = 0
invalid_relationships = 0

for rel in relationships:
    from_node = rel.get('from')
    to_node = rel.get('to')
    if from_node in concept_names and to_node in concept_names:
        valid_relationships += 1
    else:
        invalid_relationships += 1
        print(f"⚠️ Invalid relationship: {from_node} → {to_node} (concept not in list)")

print(f"Valid relationships: {valid_relationships}/{len(relationships)}")
print(f"Invalid relationships: {invalid_relationships}/{len(relationships)}")
print(f"Connectivity: {'✅ All valid' if invalid_relationships == 0 else '❌ Has invalid edges'}")

# Step 6: Summary
print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
if len(concepts) == adjusted['target_concepts'] and invalid_relationships == 0 and len(relationships) > 0:
    print("✅ Integration working correctly!")
    print(f"   - Concept count matches target ({adjusted['target_concepts']})")
    print(f"   - All relationships are valid")
    print(f"   - Graph has {len(relationships)} edges")
else:
    print("⚠️ Issues detected:")
    if len(concepts) != adjusted['target_concepts']:
        print(f"   - Concept mismatch: expected {adjusted['target_concepts']}, got {len(concepts)}")
    if invalid_relationships > 0:
        print(f"   - Invalid relationships: {invalid_relationships}")
    if len(relationships) == 0:
        print(f"   - No relationships extracted!")
