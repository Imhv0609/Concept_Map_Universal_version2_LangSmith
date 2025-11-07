"""
Test to measure actual gTTS speaking rate
"""
from gtts import gTTS
import os
import tempfile

# Test text with known word count
test_texts = [
    ("The water cycle involves evaporation condensation precipitation and collection.", 9),
    ("Photosynthesis converts sunlight into chemical energy stored in glucose molecules through complex biochemical processes.", 14),
    ("The solar system contains the Sun eight planets dwarf planets moons asteroids and comets orbiting due to gravity.", 18)
]

print("🧪 Testing gTTS speaking rate...")
print("=" * 70)

total_words = 0
total_duration = 0

for text, word_count in test_texts:
    # Generate audio with gTTS
    tts = gTTS(text=text, lang='en', slow=False)
    
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
        tmp_path = tmp.name
    
    tts.save(tmp_path)
    
    # Get actual duration
    try:
        # Try pydub first (more reliable)
        from pydub import AudioSegment
        audio = AudioSegment.from_mp3(tmp_path)
        duration = len(audio) / 1000.0  # Convert ms to seconds
        
        wpm = (word_count / duration) * 60
        seconds_per_word = duration / word_count
        
        print(f"\n📝 Text: \"{text[:50]}...\"")
        print(f"   Words: {word_count}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Rate: {wpm:.0f} WPM ({seconds_per_word:.3f}s per word)")
        
        total_words += word_count
        total_duration += duration
        
    except ImportError:
        print("   ⚠️ pydub not installed, trying mutagen...")
        try:
            from mutagen.mp3 import MP3
            audio = MP3(tmp_path)
            duration = audio.info.length
            
            wpm = (word_count / duration) * 60
            seconds_per_word = duration / word_count
            
            print(f"\n📝 Text: \"{text[:50]}...\"")
            print(f"   Words: {word_count}")
            print(f"   Duration: {duration:.2f}s")
            print(f"   Rate: {wpm:.0f} WPM ({seconds_per_word:.3f}s per word)")
            
            total_words += word_count
            total_duration += duration
        except ImportError:
            print("   ⚠️ mutagen also not installed, using file size estimation")
            # Estimate: gTTS typically produces ~1KB per 0.1 second
            file_size = os.path.getsize(tmp_path)
            estimated_duration = file_size / 10000  # Very rough estimate
            print(f"   Estimated duration: {estimated_duration:.2f}s")
    except Exception as e:
        print(f"   ⚠️ Error reading audio: {e}")
    
    # Clean up
    os.unlink(tmp_path)

print("\n" + "=" * 70)
if total_words > 0 and total_duration > 0:
    avg_wpm = (total_words / total_duration) * 60
    avg_seconds_per_word = total_duration / total_words
    print(f"📊 AVERAGE gTTS RATE:")
    print(f"   {avg_wpm:.0f} WPM")
    print(f"   {avg_seconds_per_word:.3f} seconds per word")
    print(f"\n💡 RECOMMENDATION:")
    print(f"   Use speaking_rate = {avg_seconds_per_word:.2f} in timeline_mapper.py")
else:
    print("❌ Could not calculate rate (mutagen not installed)")

print("=" * 70)
