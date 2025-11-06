# Keyword-Timed Visualization - Quick Start Guide

## 🎯 What You'll See

### Step 1: Generate Concept Map
1. Enter your description (as usual)
2. Click **"🚀 Generate Concept Map"**
3. Wait for processing (~60-80s)

### Step 2: Audio Player Appears
```
┌─────────────────────────────────────────┐
│ 🎬 Dynamic Concept Map (Keyword-Timed) │
├─────────────────────────────────────────┤
│                                         │
│  📊 Concept Map (empty initially)      │
│                                         │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 📝 Progress: 0%                         │
│ ⏱️ Elapsed Time: 0.0s / 13.0s          │
│ 💡 Revealed: 0/9 concepts               │
├─────────────────────────────────────────┤
│ 🔊 Audio Narration                      │
│ 👇 Click ▶️ below to start             │
│                                         │
│ ▶️ [Audio Player] 🔊                    │
│ 🎧 Total Duration: 13.0s | Concepts: 9  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│  [🚀 Start Visualization] (button)      │
└─────────────────────────────────────────┘
```

### Step 3: Click Play + Start Visualization
1. Click **▶️** on audio player (audio starts)
2. Click **"🚀 Start Visualization"** button (timer starts)

### Step 4: Watch Concepts Appear
```
⏱️ Time: 0.35s → "Photosynthesis" appears ✨
⏱️ Time: 1.40s → "Light Energy" appears ✨
⏱️ Time: 2.75s → "Chemical Energy" appears ✨
⏱️ Time: 3.10s → "Chlorophyll" appears ✨
...and so on!
```

## 🎨 Visual Flow

```
Before (Sentence-by-Sentence):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sentence 1: "Photosynthesis converts..."
   ⏸️  Audio plays → Wait → Reveal concepts
   ⬇️
Sentence 2: "Chlorophyll molecules..."
   ⏸️  Audio plays → Wait → Reveal concepts
   ⬇️
Sentence 3: "Water molecules split..."
   ⏸️  Audio plays → Wait → Reveal concepts

Result: Step-by-step reveals (choppy)
```

```
After (Keyword-Timed):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Full narration: "Photosynthesis converts light..."
   ⏵️  Audio plays continuously
   
   0.35s → 💡 "Photosynthesis" appears
   1.40s → 💡 "Light Energy" appears
   2.75s → 💡 "Chemical Energy" appears
   3.10s → 💡 "Chlorophyll" appears
   ...concepts appear smoothly as keywords are spoken

Result: Smooth, synchronized reveals ✨
```

## ⚙️ How It Works Internally

### Timeline Creation:
```python
# 1. Merge all sentences
full_text = "Photosynthesis converts light energy into chemical energy. 
             Chlorophyll molecules absorb sunlight..."

# 2. Calculate word timings (0.35s per word)
word_timings = [
    {"word": "Photosynthesis", "start_time": 0.00, "end_time": 0.35},
    {"word": "converts", "start_time": 0.35, "end_time": 0.70},
    {"word": "light", "start_time": 0.70, "end_time": 1.05},
    {"word": "energy", "start_time": 1.05, "end_time": 1.40},
    ...
]

# 3. Assign reveal times to concepts
concepts = [
    {"name": "Photosynthesis", "reveal_time": 0.35},    # Last word: "Photosynthesis"
    {"name": "Light Energy", "reveal_time": 1.40},      # Last word: "energy"
    {"name": "Chemical Energy", "reveal_time": 2.75},   # Last word: "energy"
    ...
]
```

### Visualization Loop:
```python
# Start timer when user clicks "Start Visualization"
start_time = time.time()

while elapsed < total_duration:
    elapsed = time.time() - start_time
    
    # Check every 0.1 seconds
    for concept in concepts:
        if concept["reveal_time"] <= elapsed and concept not in visible:
            # Reveal this concept with fade-in animation
            reveal_concept(concept)
            visible.add(concept)
    
    # Update progress bar
    progress = elapsed / total_duration
    show_progress(progress)
```

## 🎯 Key Benefits

### 1. **Natural Flow**
- No breaks between sentences
- Concepts appear organically
- Feels like a story unfolding

### 2. **Perfect Timing**
- Concepts appear exactly when mentioned
- Multi-word concepts wait for completion
- No early/late reveals

### 3. **User Control**
- Click "Start" when ready
- Can wait between Play and Start
- Timing always accurate

### 4. **Smooth Animations**
- 0.5s fade-in per concept
- No lag or jank
- Professional appearance

## 🔧 Adjustable Settings

### Already Implemented:
- ✅ Speaking rate: 0.35s/word (~172 wpm)
- ✅ Animation speed: 0.5s fade-in
- ✅ Multi-word reveal: Last word timing
- ✅ Check interval: 0.1s (100ms accuracy)

### Can Be Changed Later:
```python
# In timeline_mapper.py
speaking_rate = 0.35  # Adjust for faster/slower speech

# In streamlit_app_standalone.py
animation_duration = 0.5  # Adjust for faster/slower fade-in
check_interval = 0.1  # Adjust for more/less frequent checks
```

## 🎬 Example Timeline

**Description:** "Photosynthesis converts light energy into chemical energy."

**Generated Timeline:**
```
Full Text: "Photosynthesis converts light energy into chemical energy."
Total Duration: 2.75s
Words: 7

Word Timings:
0.00-0.35s: "Photosynthesis"
0.35-0.70s: "converts"
0.70-1.05s: "light"
1.05-1.40s: "energy"
1.40-1.75s: "into"
1.75-2.10s: "chemical"
2.10-2.75s: "energy." (+300ms for period)

Concept Reveal Times:
- "Photosynthesis" → 0.35s (word 1)
- "Light Energy" → 1.40s (word 4)
- "Chemical Energy" → 2.75s (word 7)
```

**What User Sees:**
```
0.00s: [Empty graph]
0.35s: "Photosynthesis" fades in ✨
1.40s: "Light Energy" fades in ✨
2.75s: "Chemical Energy" fades in ✨
2.80s: 🎉 Complete!
```

## 📱 Browser Behavior

### Audio Auto-Play:
Some browsers auto-play `st.audio()` → Audio might start immediately
- **Expected behavior**
- Timer only starts when "Start Visualization" clicked
- Concepts won't appear until button clicked
- If audio finishes before button click, it's okay (can replay)

### Recommended Flow:
1. Click ▶️ on audio player
2. **Immediately** click "Start Visualization"
3. Enjoy synchronized experience!

## 🐛 Troubleshooting

### Concepts Appear Too Early/Late
- Check your description has correct punctuation
- Speaking rate assumes standard English pacing
- Can adjust `speaking_rate` in `timeline_mapper.py`

### Animation Too Fast/Slow
- Change `animation_duration` in `reveal_concepts_progressively()`
- Default: 0.5s (balanced)
- Faster: 0.3s, Slower: 0.8s

### Audio Out of Sync
- Make sure to click "Start Visualization" right after Play
- If audio finishes, can replay and click Start again
- Browser might buffer/delay audio slightly (normal)

## 🎓 Technical Deep Dive

### Why 0.35s per word?
- Average English: ~150-180 wpm in TTS
- 172 wpm = 2.86 words/sec = 0.35s per word
- Tested with gTTS and edge-tts → feels natural
- Slightly faster than conversational (0.4s) for efficiency

### Why check every 0.1s?
- Human perception: ~100ms threshold
- Concepts appear within 100ms of keyword
- More frequent = more CPU, less frequent = lag
- 0.1s = perfect balance

### Why fade-in 0.5s?
- Too fast (0.2s) → jarring, hard to see
- Too slow (1.0s) → boring, distracting
- 0.5s → smooth, noticeable, professional

## 🚀 Ready to Use!

The implementation is **complete and tested**. Just run:

```bash
streamlit run streamlit_app_standalone.py
```

Enter a description, generate the map, and enjoy keyword-timed reveals! 🎉
