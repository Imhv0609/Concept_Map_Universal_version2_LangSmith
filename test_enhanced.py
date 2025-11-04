"""Quick test script for enhanced dynamic concept map"""
import os
from dotenv import load_dotenv
from dynamic_orchestrator import run_dynamic_mode

# Load environment variables
load_dotenv()

description = """Photosynthesis is the process by which plants convert light energy into chemical energy. 
Chlorophyll molecules in chloroplasts absorb sunlight. 
During the light-dependent reactions, water molecules are split to release oxygen. 
The Calvin cycle uses carbon dioxide to produce glucose."""

educational_level = "high school"
topic_name = "Photosynthesis"

print("🧪 Testing Enhanced Dynamic Concept Map System")
print("=" * 70)
print(f"Topic: {topic_name}")
print(f"Level: {educational_level}")
print("=" * 70)
print()

success = run_dynamic_mode(description, educational_level, topic_name)

if success:
    print("\n✅ Test completed successfully!")
else:
    print("\n❌ Test failed!")
