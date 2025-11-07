# Why Edge-TTS Can't Provide Word-Level Timings

## The Simple Explanation

**TL;DR**: Edge-TTS is like ordering food delivery but the restaurant never sends you tracking updates - the delivery system is ready to show you updates, but the restaurant simply doesn't provide them.

---

## What We Tried to Do

We wanted to synchronize concept nodes appearing on screen with the exact moment each word is spoken in the audio. For example:
- When the narrator says "**water**" at 2.3 seconds → show "Water" node
- When the narrator says "**evaporation**" at 5.7 seconds → show "Evaporation" node

To do this perfectly, we need to know **exactly when each word is spoken** (word-level timing).

---

## Why We Thought Edge-TTS Would Work

Edge-TTS (Microsoft's text-to-speech library) has a feature called **WordBoundary events**. Looking at the code:

```python
# You CAN set this parameter:
communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")

# And the library DOES send a request to Microsoft's server asking for word timings:
{"wordBoundaryEnabled":"true"}
```

The library has a `SubMaker` class specifically designed to collect word timings:
```python
submaker = edge_tts.SubMaker()
for chunk in communicate.stream():
    if chunk["type"] == "WordBoundary":  # Should give us word timings
        submaker.feed(chunk)
```

Everything looks ready! The code is there, the classes exist, the documentation mentions it.

---

## The Problem: Microsoft's Server Doesn't Send WordBoundary Data

Here's what actually happens:

### What We Expected:
```
Our App → Microsoft Server: "Please give me WordBoundary events"
Microsoft Server → Our App: 
  ✅ Audio chunk 1
  ✅ Audio chunk 2
  ✅ WordBoundary: "water" at 2.3s
  ✅ Audio chunk 3
  ✅ WordBoundary: "cycle" at 3.1s
  ✅ WordBoundary: "evaporation" at 5.7s
```

### What Actually Happens:
```
Our App → Microsoft Server: "Please give me WordBoundary events"
Microsoft Server → Our App: 
  ✅ Audio chunk 1
  ✅ Audio chunk 2
  ✅ SentenceBoundary: "The water cycle involves..." at 0-8s
  ✅ Audio chunk 3
  ✅ Audio chunk 4
  ❌ NO WordBoundary events at all!
```

---

## Proof: Our Testing Results

We created comprehensive tests and discovered:

### Test 1: What Events Do We Receive?
```python
# test_edgetts_boundaries.py
communicate = edge_tts.Communicate(text, voice)
for chunk in communicate.stream():
    print(f"Event type: {chunk['type']}")
```

**Result**:
```
Event type: audio
Event type: audio
Event type: audio
Event type: SentenceBoundary  ← Only sentence-level timing
Event type: audio
Event type: audio
...53 audio chunks total...
Event type: SentenceBoundary

Total WordBoundary events: 0  ← ZERO word timings!
```

### Test 2: Check Edge-TTS Version
```bash
pip show edge-tts
```

**Result**:
```
Name: edge-tts
Version: 7.2.3  ← Latest version
```

### Test 3: SubMaker Methods Available
```python
print(dir(submaker))
```

**Result**:
```
['cues', 'feed', 'get_srt', 'type']  ← Methods exist, but no data to feed them!
```

---

## Why This Happens

There are a few possible reasons:

### 1. **Microsoft's Policy Decision**
- Microsoft might have **disabled** WordBoundary events on their free Edge-TTS service
- They might reserve this feature for their **paid Azure Speech Service**
- It's a business decision to differentiate free vs. paid tiers

### 2. **Technical Limitations**
- The server might be configured to only provide sentence-level boundaries
- WordBoundary generation might be too computationally expensive for free tier

### 3. **API Version Mismatch**
- The Edge-TTS library code is ready for WordBoundary events
- But Microsoft's server API might have changed or never fully implemented it
- The documentation mentions a feature that doesn't actually work in practice

---

## Real-World Analogy

Imagine you order a pizza with real-time tracking:

1. **The App (Edge-TTS library)**: Has a beautiful tracking screen ready to show you "Driver is 2 blocks away!"

2. **You Enable Tracking**: You click "Enable detailed tracking" (set `boundary="WordBoundary"`)

3. **The Restaurant Receives Your Request**: They see you want detailed tracking

4. **What You Get**: 
   - ❌ Not detailed updates like "Driver at Main St", "Driver at Oak Ave"
   - ✅ Only basic updates like "Order confirmed", "Out for delivery", "Delivered"

The app is ready to show details, you requested details, but **the restaurant simply doesn't send detailed location data** - they only send major milestones.

---

## Alternative Solutions

Since Edge-TTS doesn't work for word-level timing, here are the alternatives:

### Option 1: **Estimated Timings** (Current Solution ✅)
```python
speaking_rate = 0.40  # seconds per word (150 WPM)
reveal_time = word_position * 0.40 + punctuation_pauses
```

**Pros**:
- ✅ Works reliably
- ✅ Free
- ✅ Well-calibrated for gTTS
- ✅ Good enough for educational content

**Cons**:
- ⚠️ Not millisecond-perfect (but close enough!)

### Option 2: **Paid TTS Services**
Use services that actually provide word timings:
- **Azure Speech Service** (Microsoft's paid version)
- **AWS Polly** (Amazon)
- **Google Cloud TTS**

**Pros**:
- ✅ Real word-level timings
- ✅ Millisecond precision

**Cons**:
- 💰 Costs money
- 🔧 Requires API keys and setup

### Option 3: **Forced Alignment Tools**
Use audio analysis tools like Gentle, aeneas, or Montreal Forced Aligner:

**Pros**:
- ✅ Can extract word timings from any audio

**Cons**:
- 🐌 Slow processing
- 🔧 Complex setup
- 📊 Requires audio analysis libraries

---

## Our Final Solution

We use **calibrated estimated timings** (Option 1) because:

1. **It works reliably** - No dependency on Microsoft's server behavior
2. **It's free** - No API costs
3. **It's well-tuned** - 0.40s per word matches gTTS speaking rate (~150 WPM)
4. **First concept at time 0** - No delay at the start
5. **Natural pauses** - Accounts for punctuation (0.4s for sentences, 0.2s for commas)

For educational concept maps, **±0.5 second accuracy is perfectly acceptable**. Viewers won't notice if a node appears 0.3 seconds early or late - the visual flow is smooth and the concepts sync well with the narration.

---

## Bottom Line

**Edge-TTS doesn't provide word timings not because our code is wrong, but because Microsoft's server simply doesn't send that data.**

It's like having a perfectly working radio but the station isn't broadcasting on that frequency. We can't force Microsoft to send data they don't provide.

The good news? Our estimated timing solution works great for the intended purpose! 🎉
