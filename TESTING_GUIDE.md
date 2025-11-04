# Dynamic Concept Map - Testing Guide

## 🎯 What Was Built

The dynamic concept map system has been successfully implemented with these new files:

1. **`timeline_mapper.py`** - Makes ONE API call, maps concepts to sentences
2. **`streamlit_visualizer.py`** - Real-time web-based graph visualization
3. **`dynamic_orchestrator.py`** - Coordinates TTS + visualization updates
4. **`main_universal.py`** - Modified to add `--dynamic` flag

## 🚀 How to Test

### **Quick Test (Recommended)**

Run the test script:
```bash
./test_dynamic.sh
```

This will:
- ✅ Make ONE API call to extract concepts (takes ~3-5 seconds)
- ✅ Launch Streamlit at http://localhost:8501
- ✅ Play TTS narration sentence-by-sentence
- ✅ Update the concept map dynamically in the browser

### **Manual Test**

```bash
python3 main_universal.py \
  --description "Photosynthesis converts light into energy. Plants use chlorophyll to absorb sunlight." \
  --level "high school" \
  --topic "Photosynthesis" \
  --dynamic
```

### **What to Expect**

#### Terminal Output:
```
🚀 Starting Dynamic Concept Map Generation
======================================================================
📋 Step 1: Creating timeline (analyzing full description)...
🔥 Making SINGLE API call to extract all concepts from full description...
✅ API call complete: Extracted 5 concepts, 4 relationships
📝 Split description into 2 sentences
✅ Timeline created! 2 sentences, 5 concepts

======================================================================
Timeline Summary: Photosynthesis
======================================================================
Educational Level: high school
Total Sentences: 2
Total Concepts: 5
======================================================================

Sentence 0: "Photosynthesis converts light into energy."
  Concepts: ['Photosynthesis', 'Light', 'Energy']
  Relationships: 2
  Est. Duration: 2.0s

Sentence 1: "Plants use chlorophyll to absorb sunlight."
  Concepts: ['Plants', 'Chlorophyll']
  Relationships: 2
  Est. Duration: 2.4s

💾 Timeline saved to: /tmp/concept_map_timeline.json
📝 Created Streamlit runner script: .../_streamlit_runner_temp.py

======================================================================
🌐 DYNAMIC CONCEPT MAP READY
======================================================================

📍 Streamlit server will start shortly...

🔗 Open this URL in your browser:
   http://localhost:8501

⚠️  Keep this terminal window open while viewing the concept map
⚠️  Press Ctrl+C in terminal to stop the server when done

======================================================================
```

#### Browser (http://localhost:8501):
1. **Initial view**: Empty canvas with title
2. **After sentence 1 spoken**: 3 nodes appear (Photosynthesis, Light, Energy)
3. **After sentence 2 spoken**: 5 nodes total (adds Plants, Chlorophyll)
4. **Edges**: Arrows showing relationships with labels
5. **Progress bar**: Shows completion (Sentence 1/2 → 2/2)

## 🔍 Key Features to Verify

### ✅ Single API Call
- Check terminal logs for: `"🔥 Making SINGLE API call"`
- Should only appear ONCE at the beginning
- API call takes ~3-5 seconds

### ✅ TTS Synchronization
- Listen for TTS speaking each sentence
- Concepts should appear AFTER each sentence is spoken
- 1 second pause between sentences for visual absorption

### ✅ Real-time Updates
- Graph should update smoothly in browser
- No page refreshes
- Nodes fade in with animations

### ✅ Static Mode Still Works
Test that old functionality is preserved:
```bash
python3 main_universal.py \
  --description "Water cycle" \
  --level "elementary"
```
Should work exactly as before (no Streamlit, just JSON output).

## 🐛 Troubleshooting

### Issue: "streamlit not found"
**Solution**: Install Streamlit
```bash
pip3 install streamlit>=1.28.0
```

### Issue: "Port 8501 already in use"
**Solution**: Kill existing Streamlit process
```bash
lsof -ti:8501 | xargs kill -9
```

### Issue: Can't hear TTS
**Solution**: Check system audio settings, ensure volume is up

### Issue: Browser doesn't show graph
**Solution**: 
- Refresh the browser page
- Check terminal for errors
- Ensure timeline.json was created in /tmp/

## 📊 Performance Expectations

| Metric | Expected Value |
|--------|----------------|
| API Calls | **1** (at beginning only) |
| API Response Time | 3-5 seconds |
| Total Time (2 sentences) | ~10-15 seconds |
| TTS per sentence | 2-3 seconds |
| Visualization update | Instant (<0.1s) |
| Cost | ~$0.0001 per run |

## 🎨 What's Different from Static Mode

| Feature | Static Mode | Dynamic Mode |
|---------|-------------|--------------|
| Output | JSON + PNG | Web interface |
| Visualization | Final map only | Progressive reveal |
| TTS | Reads full description | Sentence-by-sentence |
| Interactivity | None | Real-time updates |
| API Calls | 1 | 1 (same!) |
| Speed | Fast (~3-5s) | Slower (~10-15s due to TTS) |

## 🔄 Testing Checklist

- [ ] Run `./test_dynamic.sh`
- [ ] Verify terminal shows "SINGLE API call" message
- [ ] Confirm Streamlit URL appears: http://localhost:8501
- [ ] Open browser to that URL
- [ ] See empty canvas initially
- [ ] Hear TTS speaking sentence 1
- [ ] See concepts appear after sentence 1
- [ ] Hear TTS speaking sentence 2
- [ ] See more concepts appear after sentence 2
- [ ] Verify edges have relationship labels
- [ ] Check progress bar shows completion
- [ ] Press Ctrl+C to stop server
- [ ] Test static mode still works (no --dynamic flag)

## 💡 Next Steps (Future Enhancements)

These are NOT implemented yet, but could be added:

- [ ] Interactive graph (drag nodes, zoom, pan)
- [ ] Save animation as video/GIF
- [ ] Pause/resume TTS playback
- [ ] Adjust animation speed
- [ ] Export final graph as PNG from browser
- [ ] Multiple description sections (chapters)

## 📝 Notes

- **TTS is always enabled** (no need for --tts flag)
- **Static mode is default** (need --dynamic to use new mode)
- **Temp files cleaned up automatically** on exit
- **Compatible with existing LangSmith tracking**
