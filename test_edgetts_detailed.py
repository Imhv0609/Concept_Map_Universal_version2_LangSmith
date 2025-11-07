"""
Test different Edge-TTS approaches to get word boundaries
"""
import asyncio
import edge_tts
from edge_tts import SubMaker

async def test_with_submaker():
    text = "The water cycle involves evaporation."
    voice = "en-US-AriaNeural"
    
    print("🧪 Testing Edge-TTS with SubMaker approach...")
    print("=" * 70)
    
    # Create SubMaker instance
    submaker = SubMaker()
    
    communicate = edge_tts.Communicate(text, voice)
    
    # Method 1: Use SubMaker's create_sub method
    print("\n📡 Streaming with SubMaker...")
    async for chunk in communicate.stream():
        if chunk["type"] == "WordBoundary":
            # SubMaker expects (offset, text) tuple
            submaker.create_sub((chunk["offset"], chunk["text"]))
            print(f"Word: {chunk['text']}, Offset: {chunk['offset']}")
    
    # Try to generate subtitles
    print("\n🎬 Generating subtitles...")
    try:
        # SubMaker has different methods, let's try them
        print(f"SubMaker methods: {[m for m in dir(submaker) if not m.startswith('_')]}")
        
        # Try generate_subs if it exists
        if hasattr(submaker, 'generate_subs'):
            subs = submaker.generate_subs()
            print(f"✅ generate_subs() returned: {subs}")
        
        # Try subs attribute
        if hasattr(submaker, 'subs'):
            print(f"✅ subs attribute: {submaker.subs}")
            
        # Try offset attribute
        if hasattr(submaker, 'offset'):
            print(f"✅ offset attribute: {submaker.offset}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("=" * 70)

async def test_raw_word_boundaries():
    text = "The water cycle involves evaporation."
    voice = "en-US-AriaNeural"
    
    print("\n🧪 Testing raw WordBoundary extraction...")
    print("=" * 70)
    
    communicate = edge_tts.Communicate(text, voice)
    
    word_timings = []
    
    async for chunk in communicate.stream():
        if chunk["type"] == "WordBoundary":
            print(f"\n📝 WordBoundary chunk:")
            print(f"   Keys: {list(chunk.keys())}")
            print(f"   Data: {chunk}")
            
            # Try different offset interpretations
            offset = chunk.get("offset", 0)
            duration = chunk.get("duration", 0)
            text_word = chunk.get("text", "")
            
            # Edge-TTS uses 100-nanosecond units (ticks)
            # 1 tick = 100 nanoseconds = 0.0000001 seconds
            # So divide by 10,000,000 to get seconds
            
            start_sec = offset / 10_000_000.0
            
            if duration > 0:
                end_sec = (offset + duration) / 10_000_000.0
            else:
                # Estimate based on word length
                end_sec = start_sec + (len(text_word) * 0.05 + 0.2)
            
            word_timings.append({
                "word": text_word,
                "start": start_sec,
                "end": end_sec
            })
            
            print(f"   Parsed: '{text_word}' at {start_sec:.3f}s - {end_sec:.3f}s")
    
    print("\n" + "=" * 70)
    print(f"📊 Total word timings: {len(word_timings)}")
    for wt in word_timings:
        print(f"   '{wt['word']:15s}' → {wt['start']:6.3f}s - {wt['end']:6.3f}s")
    
    if word_timings:
        total = word_timings[-1]['end']
        print(f"\n⏱️ Total duration: {total:.3f}s")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_with_submaker())
    asyncio.run(test_raw_word_boundaries())
