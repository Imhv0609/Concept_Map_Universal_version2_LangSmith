# Edge-TTS Audio Generation Troubleshooting Guide

## 🔍 The Problem

You're seeing errors like:
```
ERROR: ❌ Failed to generate audio for sentence X: 
No audio was received. Please verify that your parameters are correct.
```

## 🎯 Root Causes & Solutions

### **1. Event Loop Conflict (Streamlit + asyncio)**

**Problem:** Streamlit runs its own async event loop, and edge-tts also needs async. When `asyncio.run()` is called inside an already-running loop, it fails.

**Solution Implemented:**
```python
# Install nest-asyncio
pip install nest-asyncio

# Apply in code
import nest_asyncio
nest_asyncio.apply()

# Now nested async works
asyncio.get_event_loop().run_until_complete(async_function())
```

**Files Changed:**
- ✅ `precompute_engine.py` - Added nest_asyncio handling
- ✅ `requirements.txt` - Added nest-asyncio>=1.5.0

---

### **2. Internet Connectivity Required**

**Problem:** Edge-TTS is a **cloud service** that requires internet to connect to Microsoft's Azure TTS servers.

**Check Internet:**
```python
python test_edge_tts.py
```

If you see:
```
❌ No internet connection!
```

**Solutions:**
- ✅ Check your internet connection
- ✅ Check firewall settings (allow outbound HTTPS)
- ✅ If behind proxy, configure proxy settings
- ✅ In Streamlit Cloud: Should have internet by default

---

### **3. Voice Name Issues**

**Problem:** Invalid or unavailable voice name.

**Default voice:** `en-US-AriaNeural`

**Check available voices:**
```bash
python -c "import asyncio; import edge_tts; asyncio.run(edge_tts.list_voices())"
```

**Popular voices:**
- `en-US-AriaNeural` - Female, clear, professional ✅
- `en-US-GuyNeural` - Male, natural
- `en-GB-SoniaNeural` - British female
- `en-AU-NatashaNeural` - Australian female

---

### **4. Streamlit Cloud Specific Issues**

**Problem:** Works locally but fails on Streamlit Cloud.

**Checklist:**
```
✅ requirements.txt includes:
   - edge-tts>=7.2.0
   - nest-asyncio>=1.5.0

✅ No firewall blocking outbound requests

✅ Check Streamlit Cloud logs for detailed errors
```

---

## 🧪 Diagnostic Steps

### **Step 1: Test Edge-TTS Locally**

```bash
python test_edge_tts.py
```

**Expected output:**
```
🎉 ALL TESTS PASSED!
✅ Edge-TTS is working correctly
```

**If fails:** Check internet, install edge-tts

---

### **Step 2: Test in Streamlit**

```bash
streamlit run streamlit_app_standalone.py
```

**Watch for:**
```
INFO:precompute_engine:🎤 Generating audio 1/8...
INFO:precompute_engine:✅ Generated audio for sentence 0: sentence_0.mp3 (24048 bytes)
```

**If fails:** Check logs for specific error

---

### **Step 3: Check Generated Files**

```python
import tempfile
import os

# Check temp directory
temp_dir = tempfile.gettempdir()
print(f"Temp dir: {temp_dir}")

# List audio files
for f in os.listdir(temp_dir):
    if f.startswith("sentence_") and f.endswith(".mp3"):
        full_path = os.path.join(temp_dir, f)
        size = os.path.getsize(full_path)
        print(f"  📁 {f}: {size} bytes")
```

---

## 🔧 Fix Applied

### **Before (Broken in Streamlit):**
```python
def generate_audio_file(self, text: str, index: int) -> str:
    output_file = os.path.join(self.temp_dir, f"sentence_{index}.mp3")
    
    # ❌ Fails in Streamlit - event loop conflict
    asyncio.run(self._generate_audio_async(text, output_file))
    
    return output_file
```

### **After (Works in Streamlit):**
```python
def generate_audio_file(self, text: str, index: int) -> str:
    output_file = os.path.join(self.temp_dir, f"sentence_{index}.mp3")
    
    try:
        # ✅ Apply nest_asyncio for Streamlit compatibility
        import nest_asyncio
        nest_asyncio.apply()
        
        # ✅ Use existing event loop if running
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.run_until_complete(self._generate_audio_async(text, output_file))
        else:
            asyncio.run(self._generate_audio_async(text, output_file))
        
        # ✅ Verify file was created
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return output_file
        else:
            return None
            
    except Exception as e:
        logger.error(f"Failed: {e}")
        return None
```

---

## 📋 Installation Commands

### **Local Development:**
```bash
pip install -r requirements.txt
```

### **Test Audio Generation:**
```bash
python test_edge_tts.py
```

### **Run App:**
```bash
streamlit run streamlit_app_standalone.py
```

---

## 🎯 Expected Behavior

### **Successful Audio Generation:**
```
INFO:precompute_engine:🎵 Pre-generating all audio files...
INFO:precompute_engine:  🎤 Generating audio 1/8: "Photosynthesis is the process..."
INFO:precompute_engine:✅ Generated audio for sentence 0: sentence_0.mp3 (24048 bytes)
INFO:precompute_engine:  🎤 Generating audio 2/8: "Chloroplasts contain..."
INFO:precompute_engine:✅ Generated audio for sentence 1: sentence_1.mp3 (28932 bytes)
...
INFO:precompute_engine:✅ Generated 8 audio files
INFO:precompute_engine:  🎵 Audio files: 8
```

---

## 🚨 Common Errors & Solutions

### **Error 1: "No audio was received"**
```
❌ Cause: Internet connectivity or edge-tts service issue
✅ Solution: Check internet, verify edge-tts version
```

### **Error 2: "RuntimeError: This event loop is already running"**
```
❌ Cause: Streamlit's event loop conflict
✅ Solution: Use nest_asyncio (already implemented)
```

### **Error 3: "Audio file not created"**
```
❌ Cause: File system permissions or disk space
✅ Solution: Check temp directory permissions
```

### **Error 4: "Voice not found"**
```
❌ Cause: Invalid voice name
✅ Solution: Use "en-US-AriaNeural" or list available voices
```

---

## 🎉 Success Indicators

✅ **Logs show:**
- `✅ Generated audio for sentence X: sentence_X.mp3 (XXXX bytes)`
- `🎵 Audio files: 8` (or number of sentences)

✅ **Temp directory contains:**
- `sentence_0.mp3`
- `sentence_1.mp3`
- ... (one per sentence)

✅ **Streamlit UI shows:**
- Audio players below each graph step
- Playable audio files

---

## 📞 Still Having Issues?

1. **Check logs:** Look for the specific error message
2. **Run diagnostic:** `python test_edge_tts.py`
3. **Verify internet:** Test connectivity to Azure
4. **Check requirements:** All packages installed?
5. **Try different voice:** Change to `en-US-GuyNeural`

---

## 🔮 Future Improvements

### **Option 1: Offline TTS (No Internet Required)**
```bash
pip install pyttsx3  # Already in requirements
```

```python
def generate_audio_offline(text, output_file):
    import pyttsx3
    engine = pyttsx3.init()
    engine.save_to_file(text, output_file)
    engine.runAndWait()
```

**Pros:** Works offline  
**Cons:** Lower quality voice

### **Option 2: Cached Audio**
Pre-generate common phrases and cache them.

### **Option 3: Graceful Degradation**
If TTS fails, continue without audio.

---

## 📊 Testing Matrix

| Environment | Internet | nest_asyncio | Expected Result |
|-------------|----------|--------------|-----------------|
| Local | ✅ Yes | ✅ Yes | ✅ Works |
| Local | ❌ No | ✅ Yes | ❌ Fails (no internet) |
| Streamlit Cloud | ✅ Yes | ✅ Yes | ✅ Works |
| Streamlit Cloud | ✅ Yes | ❌ No | ❌ Fails (event loop) |

---

## 💡 Key Takeaways

1. **edge-tts requires internet** - It's a cloud service
2. **Streamlit needs nest_asyncio** - For event loop compatibility
3. **Always verify file creation** - Check file exists and size > 0
4. **Graceful error handling** - App should continue without audio if it fails
5. **Test locally first** - Use `test_edge_tts.py` before deployment

---

Your audio generation should now work! 🎉
