"""
Test gTTS (Google Text-to-Speech) Fallback
===========================================
This tests if gTTS works as a fallback for edge-tts.
"""

import os
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_gtts():
    """Test gTTS generation"""
    try:
        from gtts import gTTS
        
        # Test text
        test_text = "Photosynthesis is the process by which green plants convert light energy into chemical energy."
        
        # Create temp file
        temp_file = os.path.join(tempfile.gettempdir(), "gtts_test.mp3")
        
        logger.info(f"🎤 Testing gTTS...")
        logger.info(f"📝 Text: {test_text[:50]}...")
        logger.info(f"📁 Output: {temp_file}")
        
        # Generate audio
        tts = gTTS(text=test_text, lang='en', slow=False)
        tts.save(temp_file)
        
        # Check if file exists
        if os.path.exists(temp_file):
            file_size = os.path.getsize(temp_file)
            logger.info(f"✅ SUCCESS! Audio file created: {file_size} bytes")
            logger.info(f"📍 File location: {temp_file}")
            
            # Clean up
            os.remove(temp_file)
            return True
        else:
            logger.error("❌ FAILED! No audio file was created")
            return False
            
    except ImportError as e:
        logger.error(f"❌ gTTS not installed: {e}")
        logger.info("💡 Install with: pip install gTTS")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🧪 gTTS DIAGNOSTIC TEST")
    logger.info("=" * 60)
    
    # Check internet connectivity
    logger.info("\n📡 Checking internet connectivity...")
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        logger.info("✅ Internet connection OK")
    except:
        logger.error("❌ No internet connection!")
        logger.error("   gTTS requires internet to access Google's TTS service")
        exit(1)
    
    # Test gTTS
    logger.info("\n🎤 Testing gTTS...")
    success = test_gtts()
    
    if success:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("✅ gTTS is working correctly")
        logger.info("✅ This will work as fallback if edge-tts fails")
    else:
        logger.error("\n❌ TESTS FAILED!")
        logger.error("   Please check the errors above")
