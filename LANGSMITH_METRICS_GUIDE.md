# 📊 LangSmith Metrics Guide - Checking Your Optimization

## 🎯 Quick Access

**URL**: https://smith.langchain.com
**Project**: universal_concept_maps
**Latest Run**: Water Cycle (Oct 27, 2025, ~9:42 PM)

---

## ✅ Step-by-Step Guide

### 1. Open LangSmith
- Go to: https://smith.langchain.com
- Log in with your credentials

### 2. Navigate to Your Project
- Click on: **"universal_concept_maps"**
- You'll see a list of all your runs

### 3. Find Your Optimization Run
Look for:
- **Name**: Water Cycle
- **Date**: Oct 27, 2025, ~9:42 PM
- **Duration**: ~141 seconds
- This is your OPTIMIZED run (3 nodes)

### 4. Compare with Previous Runs
Look for older "Water Cycle" runs:
- **Date**: Oct 25, 2025 (or earlier)
- **Duration**: ~170 seconds
- This is your OLD run (4 nodes)

---

## 📊 What to Check

### A. Overall Performance (Top of Page)

```
Duration: 141.59s vs 169.90s = 28s faster! ✅
Status: Success
Type: Chain
```

### B. Node-by-Node Breakdown (Left Panel - Trace Tree)

**Click to expand the trace tree:**

```
📦 LangGraph
   ├─ extract_concepts        87s   (Was: ~18s)
   ├─ analyze_relationships   25s   (Was: ~20s)
   ├─ build_hierarchy         29s   (Was: ~26s)
   └─ (educational_enrichment REMOVED - saved ~77s!)
```

### C. Input/Output Data (Right Panel)

**Expand these sections to see:**
- Description: "Water cycle"
- Educational Level: elementary
- Extracted Concepts: 3 concepts
- Relationships: 3 relationships
- Hierarchy: 2 levels

---

## 🎯 Key Metrics to Compare

### Before Optimization (4 nodes):
- ⏱️ Total Time: ~170 seconds
- 🪙 Total Tokens: 4,432 (from terminal)
- 💰 Cost: $0.0007 (from terminal)
- 📦 Nodes: 4

### After Optimization (3 nodes):
- ⏱️ Total Time: ~141 seconds ✅ -17% faster
- 🪙 Total Tokens: 1,760 (from terminal) ✅ -60% reduction
- 💰 Cost: $0.0003 (from terminal) ✅ -57% cheaper
- 📦 Nodes: 3 ✅ 1 fewer node

---

## 🔍 What Each Metric Means

### 1. Latency (Total Duration)
- **What**: Time from start to finish
- **Where**: Top right panel
- **Goal**: Lower is better
- **Your Result**: 141s (was 170s)

### 2. Per-Node Timing
- **What**: Time each node takes
- **Where**: Trace tree on left
- **Goal**: Identify slowest nodes
- **Your Result**: extract_concepts is now slowest (87s)

### 3. Status
- **What**: Did it succeed or fail?
- **Where**: Right panel
- **Your Result**: ✅ Success

---

## 🎨 Using the Compare Feature

### To Compare Multiple Runs:

1. In the runs list, click checkboxes next to 2+ runs
2. Click "Compare" button at top
3. See side-by-side comparison:
   ```
   Run 1 (Old)          Run 2 (New)
   Duration: 170s       Duration: 141s  ✅
   Nodes: 4             Nodes: 3        ✅
   ```

---

## ⚠️ Important Notes

### What You WON'T See:
- ❌ **Token counts** - Gemini doesn't report to LangSmith
- ❌ **Cost data** - Not auto-calculated
- ❌ **Token breakdown** - API limitation

### Where to Find Missing Data:
- ✅ **Terminal output** - Full token summary
- ✅ **Log files** - Detailed token tracking
- ✅ **Token tracker** - Our custom implementation

---

## 📈 Analyzing Trends

### View Multiple Runs Over Time:

1. Stay in the runs list view
2. Sort by date (newest first)
3. Look for patterns:
   - Are runs getting faster?
   - Any failures?
   - Consistent timing?

### Example Analysis:
```
Oct 27, 9:42 PM  |  141s  |  3 nodes  ← Current (optimized)
Oct 25, 7:45 PM  |  170s  |  4 nodes  ← Previous (with enrichment)
Oct 25, 7:28 PM  |  110s  |  4 nodes  ← Shorter input (Gravity)
```

**Insight**: Removing enrichment saved ~29 seconds!

---

## 🎯 Quick Checklist

To verify your optimization worked:

- [ ] Open https://smith.langchain.com
- [ ] Navigate to "universal_concept_maps" project
- [ ] Find Oct 27 "Water Cycle" run (~141s)
- [ ] Check trace tree shows 3 nodes (not 4)
- [ ] Compare with older run (~170s)
- [ ] Verify 28-30 second improvement
- [ ] Check for "Success" status
- [ ] Expand nodes to see timing breakdown

---

## 💡 Pro Tips

1. **Use Filters**: Filter by status (success/error) or time range
2. **Search**: Use search bar to find specific topics
3. **Tags**: Add tags to runs for easier tracking
4. **Bookmarks**: Bookmark important runs
5. **Export**: Export data to CSV for deeper analysis

---

## 🚀 Next Steps

After checking LangSmith:

1. ✅ Confirm optimization worked (29s faster)
2. 🎯 Identify next bottleneck (extract_concepts: 87s)
3. 📊 Compare multiple runs to establish patterns
4. 🔄 Run more tests with different inputs
5. 📈 Track improvements over time

---

**Your optimization is visible in LangSmith! Check it out now!** 🎉
