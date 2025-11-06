"""
Test concept and relationship extraction
"""
import os
from timeline_mapper import extract_concepts_from_full_description

# Set up API key
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "")

# Test description
description = """The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow in clouds, and falls again to the surface as precipitation. The cycling of water in and out of the atmosphere is a significant aspect of the weather patterns on Earth."""

print("Testing concept extraction...\n")
concepts, relationships = extract_concepts_from_full_description(description, "High School")

print(f"\n{'='*60}")
print(f"CONCEPTS ({len(concepts)}):")
print(f"{'='*60}")
for i, concept in enumerate(concepts, 1):
    print(f"{i}. {concept.get('name', 'N/A')}")
    print(f"   Type: {concept.get('type', 'N/A')}")
    print(f"   Definition: {concept.get('definition', 'N/A')[:80]}...")
    print()

print(f"\n{'='*60}")
print(f"RELATIONSHIPS ({len(relationships)}):")
print(f"{'='*60}")
for i, rel in enumerate(relationships, 1):
    print(f"{i}. {rel.get('from', 'N/A')} --[{rel.get('relationship', 'N/A')}]--> {rel.get('to', 'N/A')}")
