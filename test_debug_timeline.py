"""
Quick Debug Test - Check Timeline Structure
"""
import os
import sys
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from timeline_mapper import create_timeline

description = "Photosynthesis converts light energy into chemical energy. Chlorophyll molecules absorb sunlight in plant cells. Water molecules split to release oxygen."

print("🔍 Creating timeline...")
timeline = create_timeline(description, "high school", "Test")

print(f"\n✅ Timeline Created:")
print(f"   Concepts: {len(timeline.get('concepts', []))}")
print(f"   Full text: {timeline.get('full_text', 'N/A')[:100]}...")
print(f"   Audio file: {timeline.get('audio_file', 'Not generated yet')}")
print(f"   Total duration: {timeline.get('metadata', {}).get('total_duration', 0):.2f}s")

print(f"\n💡 Concepts with reveal times:")
for c in timeline.get('concepts', []):
    print(f"   - {c.get('name')}: {c.get('reveal_time', 0):.2f}s")

print(f"\n🎯 This timeline structure should work with the app!")
