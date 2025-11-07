"""
Test Edge-TTS word boundary extraction
"""
import asyncio
import edge_tts

async def test_word_boundaries():
    text = "The water cycle involves evaporation, condensation, precipitation, and collection."
    voice = "en-US-AriaNeural"
    
    print("🧪 Testing Edge-TTS word boundaries...")
    print("=" * 70)
    print(f"Text: {text}")
    print("=" * 70)
    
    communicate = edge_tts.Communicate(text, voice)
    
    word_boundaries = []
    audio_chunks = []
    
    print("\n📡 Streaming chunks...")
    async for chunk in communicate.stream():
        print(f"Chunk type: {chunk['type']}")
        
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            print(f"\n📝 Word Boundary Event:")
            print(f"   Raw chunk keys: {chunk.keys()}")
            print(f"   Full chunk: {chunk}")
            
            # Extract timing info
            offset = chunk.get("offset", 0)
            duration = chunk.get("duration", 0)
            text_word = chunk.get("text", "")
            
            # Convert from 100-nanosecond units to seconds
            start_time = offset / 10000000.0
            end_time = (offset + duration) / 10000000.0 if duration else start_time + 0.4
            
            word_boundaries.append({
                "word": text_word,
                "start_time": start_time,
                "end_time": end_time,
                "offset_raw": offset,
                "duration_raw": duration
            })
            
            print(f"   Word: '{text_word}'")
            print(f"   Time: {start_time:.2f}s - {end_time:.2f}s")
    
    print(f"\n📦 Received {len(audio_chunks)} audio chunks")
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY:")
    print(f"Total words detected: {len(word_boundaries)}")
    print("\n🎯 Word Timings:")
    for i, wb in enumerate(word_boundaries, 1):
        print(f"{i:2d}. '{wb['word']:20s}' → {wb['start_time']:6.2f}s - {wb['end_time']:6.2f}s")
    
    if word_boundaries:
        total_duration = word_boundaries[-1]["end_time"]
        print(f"\n⏱️ Total duration: {total_duration:.2f}s")
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_word_boundaries())
