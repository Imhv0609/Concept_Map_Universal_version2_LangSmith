# 🚀 Quick LangSmith Setup - TL;DR

## What You Need to Do (5 minutes)

### 1. Install Package
```bash
pip install langsmith
```

### 2. Sign Up & Get API Key
- Go to: **https://smith.langchain.com**
- Sign up (free)
- Create project: `Concept-Map-Generator`
- Get API key from Settings → API Keys

### 3. Update .env File
```bash
# Add these lines to your .env file:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_key_here_xxxxxxxxxx
LANGCHAIN_PROJECT=Concept-Map-Generator
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 4. Test It
```bash
python verify_langsmith.py
```

Should show: ✅ LangSmith is fully configured!

### 5. Run Your Project
```bash
python main_universal.py
```

Look for: 🔍 LangSmith tracing ENABLED

### 6. View Performance Metrics
- Go to: **https://smith.langchain.com**
- Click your project
- See all runs with timing & token data!

---

## What You'll Get

### For Each Run, You'll See:
- ⏱️ **Total execution time** (e.g., 10.5 seconds)
- 🎯 **Per-node timing**:
  - Extract concepts: 2.3s
  - Analyze relationships: 3.1s
  - Build hierarchy: 2.8s
  - Educational enrichment: 2.3s
- 🪙 **Token usage per node**
- 💰 **Estimated cost per run**
- 📊 **Visual trace tree**
- 🐛 **Error traces** (if any)

### Dashboard Features:
- Filter runs by success/failure
- Compare different inputs
- Export data to CSV
- Search by tags
- View full input/output of each node

---

## Optimization Tips

Once you have data:

1. **Find the slowest node** → Optimize its prompt
2. **Check token usage** → Reduce verbose prompts
3. **Compare runs** → Find patterns in slow runs
4. **Track improvements** → Measure optimization impact

---

## Need Help?

- Full guide: [LANGSMITH_SETUP.md](LANGSMITH_SETUP.md)
- LangSmith docs: https://docs.smith.langchain.com
- Verification: `python verify_langsmith.py`

---

**That's it! 🎉 You're now tracking performance metrics automatically.**
