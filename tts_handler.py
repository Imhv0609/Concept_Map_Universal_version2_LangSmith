"""
Text-to-Speech Handler Module

This module provides text-to-speech functionality for dynamic concept map generation.
It reads descriptions sentence-by-sentence with pauses between sentences.
"""

import pyttsx3
import re
import time
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TTSHandler:
    """Handles text-to-speech operations for concept map narration"""
    
    def __init__(self, rate: int = 200, volume: float = 0.9):
        """
        Initialize the TTS engine
        
        Args:
            rate: Speech rate (words per minute). Default: 200 (faster speed)
            volume: Volume level (0.0 to 1.0). Default: 0.9
        """
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            logger.info(f"✅ TTS engine initialized (rate: {rate} wpm, volume: {volume})")
        except Exception as e:
            logger.error(f"❌ Failed to initialize TTS engine: {e}")
            self.engine = None
    
    def split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences using regex
        
        Args:
            text: Input text to split
            
        Returns:
            List of sentences
        """
        # Split on period, exclamation, or question mark followed by space or end of string
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def speak_sentence(self, sentence: str) -> None:
        """
        Speak a single sentence and block until complete
        
        Args:
            sentence: The sentence to speak
        """
        if not self.engine:
            logger.warning("TTS engine not available, skipping speech")
            return
        
        try:
            logger.info(f"🎤 Speaking: {sentence[:50]}..." if len(sentence) > 50 else f"🎤 Speaking: {sentence}")
            
            # Clear any pending speech
            self.engine.stop()
            
            # Queue the sentence
            self.engine.say(sentence)
            
            # Block until speech is complete
            # runAndWait() should block, but we'll add extra insurance
            self.engine.runAndWait()
            
            # Small buffer to ensure audio playback is fully complete
            import time
            time.sleep(0.2)
            
        except Exception as e:
            logger.error(f"❌ Error speaking sentence: {e}")
    
    def speak_text_sentence_by_sentence(self, text: str, pause_duration: float = 1.0) -> None:
        """
        Read text sentence-by-sentence with pauses between sentences
        
        Args:
            text: The text to read
            pause_duration: Pause duration in seconds between sentences. Default: 1.0
        """
        if not self.engine:
            logger.warning("TTS engine not available, skipping speech")
            return
        
        sentences = self.split_into_sentences(text)
        
        logger.info(f"📢 Starting TTS narration ({len(sentences)} sentences)")
        logger.info("="*60)
        
        for i, sentence in enumerate(sentences, 1):
            logger.info(f"📝 Sentence {i}/{len(sentences)}")
            self.speak_sentence(sentence)
            
            # Pause between sentences (except after the last one)
            if i < len(sentences):
                logger.info(f"⏸️  Pausing for {pause_duration} seconds...")
                time.sleep(pause_duration)
        
        logger.info("="*60)
        logger.info("✅ TTS narration complete")
    
    def set_voice_properties(self, rate: int = None, volume: float = None) -> None:
        """
        Update voice properties
        
        Args:
            rate: Speech rate (words per minute)
            volume: Volume level (0.0 to 1.0)
        """
        if not self.engine:
            return
        
        if rate is not None:
            self.engine.setProperty('rate', rate)
            logger.info(f"🔧 Speech rate set to: {rate} wpm")
        
        if volume is not None:
            self.engine.setProperty('volume', volume)
            logger.info(f"🔧 Volume set to: {volume}")
    
    def test_speech(self) -> bool:
        """
        Test TTS functionality
        
        Returns:
            bool: True if test successful, False otherwise
        """
        try:
            test_message = "Text to speech is working correctly."
            logger.info("🧪 Testing TTS engine...")
            self.speak_sentence(test_message)
            logger.info("✅ TTS test successful")
            return True
        except Exception as e:
            logger.error(f"❌ TTS test failed: {e}")
            return False


def test_tts_handler():
    """Test the TTS handler with sample text"""
    print("\n" + "="*60)
    print("🧪 Testing TTS Handler")
    print("="*60)
    
    # Initialize handler
    tts = TTSHandler(rate=150, volume=0.9)
    
    # Test single sentence
    print("\n--- Test 1: Single Sentence ---")
    tts.speak_sentence("This is a test of the text to speech system.")
    
    # Test multiple sentences
    print("\n--- Test 2: Multiple Sentences ---")
    sample_text = """
    Photosynthesis is the process by which plants convert sunlight into energy.
    This process occurs in the chloroplasts of plant cells.
    It involves the absorption of carbon dioxide and water to produce glucose and oxygen.
    """
    tts.speak_text_sentence_by_sentence(sample_text, pause_duration=1.5)
    
    print("\n" + "="*60)
    print("✅ TTS Handler test complete")
    print("="*60)


if __name__ == "__main__":
    test_tts_handler()
