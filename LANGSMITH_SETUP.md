# 🔍 LangSmith Integration Guide

## What is LangSmith?

**LangSmith** is a developer platform for monitoring, debugging, and optimizing LLM applications. It automatically tracks:

- ⏱️ **Execution time** per node/function
- 🎯 **Token usage** per LLM call
- 💰 **Cost tracking** (estimated API costs)
- 📊 **Input/Output** of each step
- 🐛 **Error traces** with full context
- 🔄 **Workflow visualization** (tree view)

## Why Use LangSmith for This Project?

Your project has 4 LangGraph nodes that call Gemini AI:
1. `extract_concepts_from_description` 
2. `analyze_concept_relationships`
3. `build_concept_hierarchy`
4. `enrich_with_educational_metadata`

LangSmith will show you:
- Which node is the slowest (bottleneck identification)
- How many tokens each node uses
- Whether nodes can be parallelized
- Where errors occur in the workflow
- Performance comparison across different inputs

---

## 🚀 Setup Instructions

### Step 1: Sign Up for LangSmith

1. Go to **https://smith.langchain.com**
2. Sign up with your email (free tier available)
3. Create a new project:
   - Click "New Project"
   - Name it: `Concept-Map-Generator` (or any name)
4. Get your API key:
   - Go to Settings → API Keys
   - Click "Create API Key"
   - Copy the key (starts with `lsv2_...`)

### Step 2: Install LangSmith

```bash
pip install -r requirements.txt
```

This will install `langsmith>=0.1.0` which is now in requirements.txt.

### Step 3: Configure Environment Variables

Edit your `.env` file (create it from `.env.example` if needed):

```bash
# Copy from .env.example
cp .env.example .env

# Edit .env and add:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=lsv2_pt_your_actual_api_key_here
LANGCHAIN_PROJECT=Concept-Map-Generator
```

**Important**: Make sure `LANGCHAIN_TRACING_V2=true` (not false)

### Step 4: Run Your Project

```bash
python main_universal.py
```

You should see in the logs:
```
🔍 LangSmith tracing ENABLED - Performance metrics will be tracked
📊 LangSmith Project: Concept-Map-Generator
🌐 View traces at: https://smith.langchain.com
```

### Step 5: View Traces

1. Go to **https://smith.langchain.com**
2. Click on your project (`Concept-Map-Generator`)
3. You'll see all your runs with timing and token data

---

## 📊 What You'll See in LangSmith

### Trace View

Each run shows:
- **Run name**: Timestamp or custom name
- **Status**: ✅ Success or ❌ Error
- **Duration**: Total execution time
- **Tokens**: Total tokens used
- **Cost**: Estimated API cost

### Detailed Trace Tree

Click on any run to see:

```
📦 ConceptMapWorkflow (10.5s total)
  ├─ 🔍 extract_concepts_from_description (2.3s, 450 tokens)
  │   └─ Gemini API Call (2.1s)
  ├─ 🔗 analyze_concept_relationships (3.1s, 620 tokens)
  │   └─ Gemini API Call (2.9s)
  ├─ 🏗️ build_concept_hierarchy (2.8s, 580 tokens)
  │   └─ Gemini API Call (2.6s)
  └─ 🎓 enrich_with_educational_metadata (2.3s, 890 tokens)
      └─ Gemini API Call (2.1s)
```

### Performance Insights

You can:
- **Compare runs**: See which inputs are faster/slower
- **Identify bottlenecks**: Find the slowest node
- **Track token usage**: Optimize prompts to reduce tokens
- **Monitor errors**: See exactly where failures occur
- **Export data**: Download CSV for analysis

---

## 🎯 Optimization Tips Based on LangSmith Data

### 1. **If Node 4 (Enrichment) is Slowest**
- Consider splitting into smaller enrichment tasks
- Cache common enrichment patterns
- Reduce prompt verbosity

### 2. **If Token Usage is High**
- Shorten prompt instructions
- Use more concise examples
- Consider using a smaller model for simpler nodes

### 3. **If Sequential Execution is Slow**
- Nodes 1-4 run sequentially by design
- Could parallelize some enrichment subtasks
- Consider caching results for similar inputs

### 4. **Cost Optimization**
- LangSmith shows cost per run
- Identify high-cost patterns
- Adjust complexity_config to reduce concepts for simple inputs

---

## 🔧 Advanced Configuration

### Custom Run Names

Add to your `run_description_based_concept_mapping()` function:

```python
from langsmith import traceable

@traceable(name="ConceptMapping", run_type="chain")
def run_description_based_concept_mapping(description, educational_level, topic_name):
    # ... existing code ...
```

### Metadata Tags

Add tags to filter runs:

```python
import os
os.environ["LANGCHAIN_TAGS"] = f"level:{educational_level},words:{len(description.split())}"
```

### Custom Metrics

Track custom metrics in LangSmith:

```python
from langsmith import Client
client = Client()

# Log custom metric
client.create_feedback(
    run_id=run_id,
    key="concept_count",
    score=len(extracted_concepts)
)
```

---

## 📈 Example Performance Analysis Workflow

1. **Run 10 tests** with different description lengths:
   - 1 word, 10 words, 50 words, 200 words, 1000 words

2. **Go to LangSmith dashboard**
   - Filter by project: `Concept-Map-Generator`
   - Sort by duration or token count

3. **Identify patterns**:
   - Does time scale linearly with description length?
   - Which node uses the most tokens?
   - Are there any failures?

4. **Optimize**:
   - Adjust prompts in `nodes.py`
   - Modify complexity_config in `description_analyzer.py`
   - Re-run and compare metrics

---

## 🚫 Troubleshooting

### Issue: "LangSmith tracing disabled"

**Solution**: Check your `.env` file:
```bash
LANGCHAIN_TRACING_V2=true  # Must be 'true', not 'false'
```

### Issue: No traces appearing in LangSmith

**Solutions**:
1. Verify API key is correct (starts with `lsv2_`)
2. Check internet connection
3. Ensure project name matches
4. Look for errors in terminal logs

### Issue: "Invalid API key"

**Solution**: 
1. Go to https://smith.langchain.com/settings
2. Generate a new API key
3. Update `.env` file

### Issue: Traces are slow to appear

**Normal**: Traces may take 5-10 seconds to appear in dashboard after run completes.

---

## 📚 Additional Resources

- **LangSmith Docs**: https://docs.smith.langchain.com
- **LangGraph Docs**: https://langchain-ai.github.io/langgraph/
- **Support**: support@langchain.dev

---

## 💡 Quick Start Checklist

- [ ] Sign up at https://smith.langchain.com
- [ ] Create project: `Concept-Map-Generator`
- [ ] Get API key from Settings → API Keys
- [ ] Install: `pip install -r requirements.txt`
- [ ] Update `.env`: Set `LANGCHAIN_TRACING_V2=true`
- [ ] Add `LANGCHAIN_API_KEY=lsv2_...` to `.env`
- [ ] Run: `python main_universal.py`
- [ ] Check logs for "LangSmith tracing ENABLED"
- [ ] View traces at https://smith.langchain.com

Happy optimizing! 🚀
