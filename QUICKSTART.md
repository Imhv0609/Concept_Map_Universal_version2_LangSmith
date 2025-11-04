# Quick Start - Dynamic Concept Maps

## 🎯 Run Dynamic Mode (DEFAULT)

```bash
# Option 1: Interactive mode (asks for input, defaults to dynamic)
python3 main_universal.py

# Option 2: Use the test script
./test_dynamic.sh

# Option 3: Run manually with CLI arguments (no --dynamic flag needed, it's default!)
python3 main_universal.py \
  --description "Your multi-sentence description here. Second sentence here." \
  --level "high school" \
  --topic "Your Topic"
```

## 📖 What Happens

1. **Single API call** extracts all concepts (~3-5 seconds)
2. **Streamlit launches** at http://localhost:8501
3. **Open browser** to that URL
4. **TTS narrates** each sentence
5. **Concepts appear** dynamically after each sentence
6. **Press Ctrl+C** in terminal to stop

## ✅ Success Indicators

- Terminal shows: "🔥 Making SINGLE API call" (only once!)
- Terminal shows: "Open http://localhost:8501 in your browser"
- Browser shows: Empty graph → concepts appear progressively
- You hear: TTS reading each sentence

## 🔙 Static Mode (Original JSON + PNG)

```bash
# Use --static flag to get JSON + PNG output (no Streamlit)
./test_static.sh

# Or manually:
python3 main_universal.py \
  --description "Your description" \
  --level "high school" \
  --static
```

**Static mode outputs:**
- JSON file in `output/` directory
- PNG visualization in `output/` directory

## 🛑 Stop the Server

Press `Ctrl+C` in the terminal where Streamlit is running.

## 📚 More Details

See `TESTING_GUIDE.md` for comprehensive testing instructions.
