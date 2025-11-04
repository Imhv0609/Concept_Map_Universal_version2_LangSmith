"""
Test Edge-TTS Audio Generation
================================
This script tests if edge-tts can generate audio files.
"""

import asyncio
import os
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_edge_tts():
    """Test edge-tts generation"""
    try:
        import edge_tts
        
        # Test text
        test_text = "Hello, this is a test of the edge TTS system."
        
        # Create temp file
        temp_file = os.path.join(tempfile.gettempdir(), "edge_tts_test.mp3")
        
        logger.info(f"🎤 Testing edge-tts...")
        logger.info(f"📝 Text: {test_text}")
        logger.info(f"🎵 Voice: en-US-AriaNeural")
        logger.info(f"📁 Output: {temp_file}")
        
        # Try to generate
        communicate = edge_tts.Communicate(test_text, "en-US-AriaNeural")
        await communicate.save(temp_file)
        
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
        logger.error(f"❌ edge-tts not installed: {e}")
        logger.info("💡 Install with: pip install edge-tts")
        return False
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error(f"   Error type: {type(e).__name__}")
        return False

async def test_with_nest_asyncio():
    """Test with nest_asyncio (Streamlit compatibility)"""
    try:
        import nest_asyncio
        nest_asyncio.apply()
        logger.info("✅ nest_asyncio applied")
    except ImportError:
        logger.warning("⚠️ nest_asyncio not installed (may cause issues in Streamlit)")
    
    return await test_edge_tts()

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🧪 EDGE-TTS DIAGNOSTIC TEST")
    logger.info("=" * 60)
    
    # Check internet connectivity
    logger.info("\n📡 Checking internet connectivity...")
    try:
        import urllib.request
        urllib.request.urlopen('https://www.google.com', timeout=5)
        logger.info("✅ Internet connection OK")
    except:
        logger.error("❌ No internet connection!")
        logger.error("   Edge-TTS requires internet to access Microsoft's TTS service")
        exit(1)
    
    # Test edge-tts
    logger.info("\n🎤 Testing Edge-TTS...")
    try:
        success = asyncio.run(test_with_nest_asyncio())
        if success:
            logger.info("\n🎉 ALL TESTS PASSED!")
            logger.info("✅ Edge-TTS is working correctly")
        else:
            logger.error("\n❌ TESTS FAILED!")
            logger.error("   Please check the errors above")
    except Exception as e:
        logger.error(f"\n❌ Test failed with exception: {e}")
        logger.error(f"   Type: {type(e).__name__}")
