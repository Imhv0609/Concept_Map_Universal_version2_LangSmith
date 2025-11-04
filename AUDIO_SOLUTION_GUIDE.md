# Audio Playback Solutions for Streamlit Cloud

## 🎯 The Problem
Streamlit Cloud runs in a **headless Linux container without audio hardware**, so pygame (which needs sound cards) doesn't work for playing audio.

## ✅ Recommended Solution: Use `st.audio()`

### **Why This Works:**
- ✅ **Browser-based** - Audio plays in the **user's browser**, not on the server
- ✅ **Cloud-compatible** - Works perfectly on Streamlit Cloud
- ✅ **Native Streamlit** - Built-in component, no extra dependencies
- ✅ **User-controlled** - Users can play/pause/seek
- ✅ **No hardware needed** - Server just sends the audio file

---

## 🔄 Implementation Comparison

### **❌ Old Method (Pygame - Doesn't Work in Cloud)**
```python
pygame.mixer.init()  # ❌ Fails: No audio device
pygame.mixer.music.load(audio_file)
pygame.mixer.music.play()
```

### **✅ New Method (Streamlit - Works Everywhere)**
```python
st.audio(audio_file, format='audio/mp3')  # ✅ Works in cloud!
```

---

## 📝 Complete Solution

### **1. Updated play_audio() Function**
```python
def play_audio(audio_file):
    """Play audio file using Streamlit's audio player"""
    try:
        if os.path.exists(audio_file):
            # Display audio player in Streamlit
            st.audio(audio_file, format='audio/mp3', start_time=0)
            
            # Optional: Show audio info
            st.caption(f"🔊 Audio: {os.path.basename(audio_file)}")
            
            return True
    except Exception as e:
        logger.error(f"Error playing audio: {e}")
        return False
```

### **2. Enhanced Version with Duration**
If you need to know audio duration (e.g., for timing animations):

```python
def play_audio(audio_file):
    """Play audio with duration tracking"""
    try:
        if os.path.exists(audio_file):
            st.audio(audio_file, format='audio/mp3')
            
            # Get duration (optional)
            try:
                from mutagen.mp3 import MP3
                audio = MP3(audio_file)
                duration = audio.info.length
                st.info(f"⏱️ Duration: {duration:.1f}s")
                return duration
            except:
                return True
    except Exception as e:
        logger.error(f"Error: {e}")
        return False
```

---

## 🎨 UI Enhancements

### **Option 1: Inline Audio Player**
```python
col1, col2 = st.columns([3, 1])
with col1:
    st.pyplot(fig)  # Show graph
with col2:
    st.audio(audio_file)  # Audio player beside it
```

### **Option 2: Auto-play with Sidebar Control**
```python
# In sidebar
auto_play = st.sidebar.checkbox("🔊 Auto-play Audio", value=True)

# In main area
if auto_play:
    st.audio(audio_file, format='audio/mp3')
else:
    with st.expander("🎵 Click to play audio"):
        st.audio(audio_file)
```

### **Option 3: Sequential Audio for Each Node**
```python
for i, (timestamp, nodes, edges, audio) in enumerate(timeline):
    st.markdown(f"### Step {i+1}: {timestamp}")
    
    # Show visualization
    st.pyplot(create_visualization(nodes, edges))
    
    # Audio for this step
    if audio:
        st.audio(audio, format='audio/mp3')
        st.caption("🎧 Listen to explanation")
```

---

## 🚀 Advanced: Background Audio with JavaScript

If you need **true auto-play** (browsers limit this), use custom HTML:

```python
def autoplay_audio(audio_file):
    """Auto-play audio using HTML5 (user interaction required)"""
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)
```

⚠️ **Note:** Browsers block autoplay unless user has interacted with the page.

---

## 📦 Requirements Update

Add to `requirements.txt`:
```txt
mutagen>=1.47.0  # Optional: For audio duration
```

Pygame is now **optional** (only for local testing):
```txt
pygame>=2.5.0  # Optional: Local audio only
```

---

## 🎭 Complete Example for Your Project

```python
def run_dynamic_visualization(timeline, layout_style="hierarchical"):
    """Run visualization with audio narration"""
    
    st.markdown("### 🎬 Dynamic Concept Map with Audio")
    
    # Audio controls
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        show_audio = st.checkbox("🔊 Enable Audio", value=True)
    with col2:
        audio_speed = st.select_slider("Speed", [0.75, 1.0, 1.25, 1.5], value=1.0)
    
    # Create containers
    graph_placeholder = st.empty()
    audio_placeholder = st.empty()
    status_placeholder = st.empty()
    
    # Animate through timeline
    for i, (timestamp, nodes, edges, audio_file) in enumerate(timeline):
        # Update status
        status_placeholder.info(f"📍 Step {i+1}/{len(timeline)}: {timestamp}")
        
        # Show visualization
        with graph_placeholder:
            fig = create_graph(nodes, edges, layout_style)
            st.pyplot(fig)
            plt.close(fig)
        
        # Play audio
        if show_audio and audio_file and os.path.exists(audio_file):
            with audio_placeholder:
                st.audio(audio_file, format='audio/mp3')
        
        # Wait between steps (optional)
        time.sleep(2)
    
    status_placeholder.success("✅ Visualization Complete!")
```

---

## 🎯 Benefits of This Approach

| Feature | Pygame (❌) | st.audio() (✅) |
|---------|------------|----------------|
| Works in Streamlit Cloud | ❌ No | ✅ Yes |
| Needs audio hardware | ❌ Yes | ✅ No |
| User controls | ❌ No | ✅ Yes (play/pause/seek) |
| Mobile-friendly | ❌ No | ✅ Yes |
| Setup complexity | ❌ High | ✅ Low |
| Browser compatibility | ❌ N/A | ✅ All modern browsers |

---

## 🧪 Testing

### **Local Testing:**
```bash
streamlit run streamlit_app_standalone.py
```

### **Cloud Testing:**
1. Push changes to GitHub
2. Deploy on Streamlit Cloud
3. Audio should play in browser ✅

---

## 🔧 Troubleshooting

### **Audio not playing?**
1. ✅ Check file exists: `os.path.exists(audio_file)`
2. ✅ Check format: Use MP3 (best compatibility)
3. ✅ Check file size: Keep under 200MB for cloud
4. ✅ Check browser: Enable audio in browser settings

### **Want synchronization?**
- Use `time.sleep()` between steps
- Or let user control with "Next" button
- Or use session state for step tracking

---

## 📚 Summary

**Old approach:** pygame → ❌ Requires audio hardware → Fails in cloud

**New approach:** `st.audio()` → ✅ Browser-based → Works everywhere

**Result:** Your audio narration will work perfectly on Streamlit Cloud! 🎉
