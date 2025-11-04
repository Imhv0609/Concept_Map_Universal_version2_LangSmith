# 🧠 Standalone Dynamic Concept Map App

## Single-Page Streamlit App

A complete, self-contained web app where you can:
1. **Type or paste** your description
2. **Click a button** to generate
3. **Watch concepts appear** dynamically on the same page!

No terminal commands needed - everything happens in the browser! 🚀

---

## 🎯 Quick Start

### Method 1: Launch Script (Easiest)
```bash
./launch_standalone_app.sh
```

### Method 2: Direct Command
```bash
streamlit run streamlit_app_standalone.py
```

### Method 3: Python Command
```bash
python3 -m streamlit run streamlit_app_standalone.py
```

---

## ✨ Features

### 📝 **Input Section**
- Large text area for your description
- Supports 4-12 sentences (optimal)
- Handles missing spaces after periods automatically
- Preserves titles (Dr., Mr., Mrs., etc.)

### ⚙️ **Settings Sidebar**
- Educational level selector
- Optional topic name
- Instructions and tips
- Feature list
- Example descriptions

### 🎬 **Dynamic Visualization**
- **Progress tracking** - See which sentence is being narrated
- **Fade-in animations** - Concepts appear smoothly (0 → 100% opacity)
- **Pop-in effects** - Nodes scale from 30% → 100%
- **Gold highlighting** - New concepts glow in gold
- **Natural voice** - Edge-TTS narration (Microsoft Azure voices)
- **Hierarchical layout** - Clean top-to-bottom organization

### 📊 **Live Updates**
- Sentence counter (1/7, 2/7, etc.)
- Current sentence display
- Concepts being revealed
- Progress bar
- Completion celebration (balloons! 🎉)

---

## 📚 Built-in Examples

Click any example to auto-fill:

1. **🌿 Photosynthesis** (5 sentences)
2. **💧 Water Cycle** (7 sentences)
3. **🌍 Climate Change** (8 sentences)
4. **⚛️ Newton's Laws** (6 sentences)

---

## 🎨 How It Works

### Behind the Scenes:
1. **Timeline Creation** (3-5 seconds)
   - Single AI API call extracts all concepts
   - Splits description into sentences
   - Maps concepts to sentences

2. **Asset Pre-computation** (10-15 seconds)
   - Generates natural voice audio for each sentence
   - Calculates hierarchical graph layout
   - Prepares animation parameters

3. **Dynamic Playback** (Real-time)
   - Plays audio narration
   - Animates concept appearance
   - Updates progress indicators
   - Smooth transitions between sentences

---

## 🖥️ Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│  🧠 Dynamic Concept Map Generator                       │
│  Enter a description and watch concepts come alive!     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────┐  ┌─────────────────┐ │
│  │  Description Text Area       │  │   ⚙️ Settings   │ │
│  │  (Type or paste here)        │  │                 │ │
│  │                              │  │   Level:        │ │
│  │                              │  │   High School   │ │
│  │                              │  │                 │ │
│  │                              │  │   Topic: (opt)  │ │
│  └──────────────────────────────┘  │                 │ │
│                                     │   📖 Instructions│ │
│  [🚀 Generate Concept Map]          │   🎨 Features   │ │
│                                     └─────────────────┘ │
├─────────────────────────────────────────────────────────┤
│  🔄 Processing...                                        │
│  📋 Creating timeline... ✅                              │
│  🎨 Generating assets... ✅                              │
├─────────────────────────────────────────────────────────┤
│  🎬 Dynamic Concept Map                                  │
│  ┌──────────────────────┐  ┌────────────────────────┐  │
│  │   📊 Concept Map     │  │  📝 Narration Progress │  │
│  │                      │  │  Progress: ████░░░░ 40%│  │
│  │      ●──●──●         │  │                        │  │
│  │      │     │         │  │  🗣️ Current Sentence   │  │
│  │      ●     ●         │  │  "Plants absorb..."    │  │
│  │                      │  │                        │  │
│  │  (Animated nodes)    │  │  💡 Concepts:          │  │
│  │                      │  │  Chlorophyll, Light    │  │
│  └──────────────────────┘  └────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

---

## 💡 Tips for Best Results

### ✅ **Do:**
- Use 4-12 sentences for optimal animations
- Include key scientific/educational terms
- Use clear, descriptive language
- Let AI auto-detect the topic name

### ⚠️ **Avoid:**
- Single-word inputs (too simple)
- Very long paragraphs (20+ sentences)
- Unrelated sentences (breaks concept mapping)

---

## 🎯 Example Usage

### Step 1: Launch the app
```bash
./launch_standalone_app.sh
```

### Step 2: Enter description
```
Photosynthesis converts light energy into chemical energy.
Chlorophyll molecules absorb sunlight in plant cells.
Water molecules split to release oxygen.
The Calvin cycle uses carbon dioxide.
Glucose is produced as the final product.
```

### Step 3: Click "Generate"
Watch as:
- Timeline is created (5 seconds)
- Audio is generated (10 seconds)
- Concepts appear dynamically with animations!

### Step 4: Enjoy!
- Listen to narration
- Watch fade-in animations
- See concepts connect in real-time

---

## 🔧 Advanced Features

### Custom Voice Speed
Edit `streamlit_app_standalone.py` line 24:
```python
engine = PrecomputeEngine(voice="en-US-AriaNeural", rate="+20%")
# +20% = faster, -20% = slower
```

### Animation Timing
Edit line 180:
```python
animation_duration=0.8  # 0.8 seconds per concept reveal
```

### Layout Style
The app uses hierarchical layout by default. To change:
Edit `precompute_engine.py` line 177:
```python
LAYOUT_STYLE = "hierarchical"  # or "shell", "circular", "spring"
```

---

## 🚀 Compared to Command-Line Version

| Feature | Standalone App | CLI Version |
|---------|---------------|-------------|
| Input method | Browser text area | Terminal prompt |
| Visualization | Same page | Opens new tab |
| Examples | Click to use | Must type |
| Settings | GUI dropdowns | CLI arguments |
| Progress | Visual status | Terminal logs |
| Ease of use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

---

## 📦 Dependencies

All dependencies already installed:
- ✅ Streamlit 1.28+
- ✅ Edge-TTS 7.2.3
- ✅ pygame 2.6.1
- ✅ NetworkX
- ✅ Matplotlib
- ✅ Google Generative AI (Gemini)

---

## 🎓 Educational Use Cases

### Teachers:
- Create concept maps for lessons
- Visualize complex topics for students
- Generate study materials dynamically

### Students:
- Understand relationships between concepts
- Study for exams with visual aids
- Learn through animated explanations

### Self-learners:
- Explore new topics visually
- Break down complex subjects
- Create study notes automatically

---

## 🐛 Troubleshooting

### Port Already in Use?
```bash
streamlit run streamlit_app_standalone.py --server.port 8502
```

### Audio Not Playing?
- Check system volume
- Try refreshing the page
- Verify Edge-TTS is installed: `pip install edge-tts`

### Slow Loading?
- Reduce number of sentences (aim for 4-8)
- Check internet connection (needed for AI API)
- Close other browser tabs

---

## 🎉 What's New vs Original?

✨ **New in Standalone App:**
1. Single-page interface (no terminal needed)
2. Built-in examples (click to use)
3. Visual progress tracking
4. Settings in sidebar
5. Status updates during generation
6. Celebration effects (balloons!)
7. Example descriptions
8. Responsive layout
9. Better error messages
10. Auto-cleanup after use

---

## 📝 Notes

- **Audio files** are temporary and auto-deleted after use
- **Layout** is calculated once, then reused
- **AI API** call happens only once (fast!)
- **Browser** must stay open during playback
- **Terminal** must stay open (Streamlit runs there)

---

## 🔗 Related Files

- `streamlit_app_standalone.py` - Main standalone app
- `launch_standalone_app.sh` - Launch script
- `timeline_mapper.py` - Timeline creation (shared)
- `precompute_engine.py` - Asset generation (shared)
- `streamlit_visualizer_enhanced.py` - Original visualizer

---

## 🏆 Best Features

1. **Zero Terminal Commands** - Everything in browser ✅
2. **Built-in Examples** - Click and go ✅
3. **Real-time Progress** - Know what's happening ✅
4. **Same-Page Viewing** - No tab switching ✅
5. **Visual Feedback** - Status messages and progress bars ✅

---

**Ready to use!** 🚀

Launch with: `./launch_standalone_app.sh`

Enjoy creating dynamic concept maps! 🧠✨
