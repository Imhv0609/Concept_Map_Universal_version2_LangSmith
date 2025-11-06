"""
Test with proper env loading order
"""
# CRITICAL: Load .env BEFORE importing timeline_mapper!
from dotenv import load_dotenv
load_dotenv()

import os
print(f"API Key loaded: {'✅ Yes' if os.getenv('GOOGLE_API_KEY') else '❌ No'}")

# NOW import timeline_mapper (after env is loaded)
from timeline_mapper import extract_concepts_from_full_description

# Test
description = "The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow in clouds, and falls again to the surface as precipitation."

print("\n🔥 Testing concept extraction...")
concepts, relationships = extract_concepts_from_full_description(description, "High School")

print(f"\n✅ Results:")
print(f"   Concepts: {len(concepts)}")
print(f"   Relationships: {len(relationships)}")

if len(concepts) > 0:
    print(f"\n📝 Concept names:")
    for c in concepts:
        print(f"   - {c.get('name')}")
        
if len(relationships) > 0:
    print(f"\n🔗 Relationships:")
    for r in relationships:
        print(f"   - {r.get('from')} → {r.get('to')}")
else:
    print("\n⚠️ No relationships extracted!")
