"""
Test Keyword-Timed Timeline Generation
=======================================
Verify that the new continuous timeline structure works correctly.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

from timeline_mapper import create_timeline

def test_continuous_timeline():
    """Test the continuous timeline with keyword timing."""
    
    # Test description
    description = """Photosynthesis converts light energy into chemical energy.
Chlorophyll molecules absorb sunlight in plant cells.
Water molecules split to release oxygen.
The Calvin cycle uses carbon dioxide.
Glucose is produced as the final product."""
    
    print("=" * 70)
    print("Testing Continuous Timeline with Keyword Timing")
    print("=" * 70)
    
    # Create timeline
    timeline = create_timeline(
        description=description,
        educational_level="high school",
        topic_name="Photosynthesis"
    )
    
    # Validate structure
    print("\n✅ Timeline Structure:")
    print(f"  - Metadata: {timeline['metadata']}")
    print(f"  - Full text length: {len(timeline['full_text'])} chars")
    print(f"  - Word timings: {len(timeline['word_timings'])} words")
    print(f"  - Concepts: {len(timeline['concepts'])}")
    print(f"  - Relationships: {len(timeline['relationships'])}")
    
    # Show full text
    print(f"\n📝 Full Text:")
    print(f"  {timeline['full_text'][:200]}...")
    
    # Show first 5 word timings
    print(f"\n⏱️ First 5 Word Timings:")
    for i, wt in enumerate(timeline['word_timings'][:5]):
        print(f"  {i+1}. '{wt['word']}' @ {wt['start_time']:.2f}s - {wt['end_time']:.2f}s")
    
    # Show concepts with reveal times
    print(f"\n💡 Concepts with Reveal Times:")
    for concept in timeline['concepts']:
        name = concept.get('name', 'Unknown')
        reveal_time = concept.get('reveal_time', 0.0)
        print(f"  - '{name}' reveals at {reveal_time:.2f}s")
    
    # Show total duration
    total_duration = timeline['metadata'].get('total_duration', 0.0)
    print(f"\n🎵 Total Duration: {total_duration:.2f}s")
    print(f"   Speaking rate: {timeline['metadata'].get('speaking_rate', 0.35):.2f}s per word")
    
    print("\n" + "=" * 70)
    print("✅ Test Complete!")
    print("=" * 70)

if __name__ == "__main__":
    test_continuous_timeline()
