"""
Quick test to check reveal_time assignments
"""
from timeline_mapper import calculate_word_timings, assign_concept_reveal_times

# Test text
text = """The water cycle describes how water evaporates from the surface of the earth, rises into the atmosphere, cools and condenses into rain or snow in clouds, and falls again to the surface as precipitation. The cycling of water in and out of the atmosphere is a significant aspect of the weather patterns on Earth."""

# Test concepts
concepts = [
    {"name": "Water Cycle"},
    {"name": "Evaporation"},
    {"name": "Atmosphere"},
    {"name": "Condensation"},
    {"name": "Precipitation"},
    {"name": "Surface Runoff"},
    {"name": "Groundwater"},
    {"name": "Transpiration"},
    {"name": "Infiltration"},
    {"name": "Solar Energy"}
]

# Calculate word timings
word_timings = calculate_word_timings(text, speaking_rate=0.35)
print(f"Total duration: {word_timings[-1]['end_time']:.2f}s")
print(f"Total words: {len(word_timings)}")
print()

# Assign reveal times
concepts_with_times = assign_concept_reveal_times(concepts, word_timings, text)

# Print results
print("Concept Reveal Times:")
print("=" * 60)
for concept in concepts_with_times:
    print(f"{concept['name']:20s} → {concept['reveal_time']:6.2f}s")
