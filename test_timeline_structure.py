"""
Quick test to verify timeline structure
"""
from timeline_mapper import create_timeline
import json

# Test with solar system description
description = """The solar system contains the Sun, eight planets, dwarf planets, moons, asteroids, and comets. Planets orbit the Sun due to gravitational force. Inner rocky planets include Mercury, Venus, Earth, and Mars. Outer gas giants are Jupiter, Saturn, Uranus, and Neptune. The asteroid belt separates inner and outer planets."""

print("🧪 Testing timeline structure...")
print("=" * 70)

timeline = create_timeline(description, "High School", "Solar System")

print(f"\n📊 Timeline Keys: {list(timeline.keys())}")
print(f"\n📈 Metadata: {json.dumps(timeline['metadata'], indent=2)}")
print(f"\n🔢 Number of concepts: {len(timeline.get('concepts', []))}")
print(f"🔢 Number of relationships: {len(timeline.get('relationships', []))}")

# Check if concepts is at top level
if 'concepts' in timeline:
    print(f"\n✅ 'concepts' found at top level")
    print(f"📝 First 3 concepts:")
    for i, concept in enumerate(timeline['concepts'][:3]):
        print(f"   {i+1}. {concept}")
else:
    print(f"\n❌ 'concepts' NOT found at top level")
    print(f"   Available keys: {list(timeline.keys())}")

# Check sentence structure
if 'sentences' in timeline:
    print(f"\n📄 Number of sentences: {len(timeline['sentences'])}")
    if len(timeline['sentences']) > 0:
        first_sent = timeline['sentences'][0]
        print(f"   First sentence has {len(first_sent.get('concepts', []))} concepts")

print("\n" + "=" * 70)
print("✅ Test complete!")
