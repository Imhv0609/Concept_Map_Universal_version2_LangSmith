# 🚀 Universal LLM-Powered Concept Map Teaching Agent

## 📋 Overview

This is a **universal concept mapping system** that can analyze **ANY topic** and create comprehensive educational concept maps with subtopics, hierarchies, and detailed educational metadata. Unlike the original system that was limited to predefined NCERT JSON files, this system works with any subject matter.

## ✨ Key Features

### 🎯 **Universal Topic Analysis**
- Works with **any topic** - not limited to predefined files
- Supports multiple educational levels (elementary to graduate)
- Generates subtopic-based concept maps

### 🧠 **5-Node AI Workflow**
1. **Topic Analysis** - Extracts 4-8 key subtopics from any topic
2. **Concept Generation** - Creates 6-12 specific concepts per subtopic
3. **Key Concept Identification** - Identifies 3-6 most critical concepts per subtopic
4. **Hierarchy Building** - Creates learning relationships within and across subtopics
5. **Educational Enrichment** - Adds teaching metadata and strategies

### 📚 **Educational Metadata**
- Difficulty levels and learning time estimates
- Learning objectives and prerequisites
- Common misconceptions and teaching strategies
- Assessment methods and real-world applications
- Memory aids and extension activities

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
1. Copy `.env.example` to `.env`
2. Add your Google Gemini API key to `.env`
3. **(Optional)** Enable LangSmith for performance monitoring:
   - Sign up at [smith.langchain.com](https://smith.langchain.com)
   - Add your LangSmith API key to `.env`
   - Set `LANGCHAIN_TRACING_V2=true`
   - See [LANGSMITH_SETUP.md](LANGSMITH_SETUP.md) for details

### Verify Setup (Optional)
```bash
python verify_langsmith.py  # Check if LangSmith is configured
```

### Usage

#### Interactive Mode
```bash
python main_universal.py
```

#### Command Line
```bash
# Basic usage
python main_universal.py --topic "Photosynthesis"

# With educational level
python main_universal.py --topic "Machine Learning" --level "undergraduate"

# With description for context
python main_universal.py --topic "Climate Change" --level "high school" --description "Focus on causes and effects"
```

## 📊 Example Topics

This system works with **any topic**, for example:
- **Science**: Photosynthesis, Quantum Physics, Climate Change
- **Technology**: Machine Learning, Cybersecurity, Blockchain
- **History**: World War II, Renaissance, Industrial Revolution  
- **Mathematics**: Calculus, Statistics, Linear Algebra
- **Literature**: Shakespeare, Poetry Analysis, Narrative Structure
- **Arts**: Color Theory, Musical Composition, Film Analysis

## 🏗️ Architecture

### Files Structure
```
Universal_Concept_Map/
├── states.py              # Workflow state definitions
├── nodes.py               # 5 core AI processing functions
├── graph.py               # LangGraph workflow orchestration
├── main_universal.py      # Main application with CLI/interactive modes
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
└── output/               # Generated concept map JSON files
```

### Workflow Design
```
Topic Input → [1] Analyze Topic → [2] Generate Concepts → [3] Identify Key Concepts 
                     ↓
[5] Enrich Subtopics ← [4] Build Hierarchies ← ↑
                     ↓
              Final Output
```

## 📈 Output Format

The system generates comprehensive JSON files containing:
- **Subtopics**: Main areas of the topic
- **Concept Hierarchies**: Learning progressions within each subtopic
- **Cross-Links**: Connections between different subtopics
- **Educational Metadata**: Teaching strategies, assessments, and learning objectives

## 🎓 Educational Levels Supported

- Elementary
- Middle School
- High School
- Undergraduate
- Graduate
- Professional
- General Audience

## 🔄 Comparison with Original System

| Feature | Original System | Universal System |
|---------|----------------|------------------|
| **Input Source** | Predefined NCERT JSON files | Any topic/subject |
| **Scope** | Single chapter analysis | Multi-subtopic analysis |
| **Flexibility** | Limited to curriculum | Universal application |
| **Output Structure** | Flat concept relationships | Hierarchical subtopic organization |
| **Educational Focus** | Chapter-specific | Comprehensive topic coverage |

## 🎯 Use Cases

- **Teachers**: Create concept maps for any lesson topic
- **Curriculum Designers**: Analyze subject matter structure
- **Students**: Understand complex topics through visual relationships
- **Educational Researchers**: Study concept relationships across domains
- **Content Creators**: Structure educational materials

## 🚀 Future Enhancements

- ✅ Visual concept map generation (PNG images)
- ✅ LangSmith performance monitoring integration
- Interactive web-based concept maps
- Learning path optimization
- Assessment question generation
- Multi-language support

## 📊 Performance Monitoring

This project integrates with **LangSmith** for performance analysis:
- Track execution time per workflow node
- Monitor token usage and API costs
- Identify bottlenecks for optimization
- Debug errors with full context traces

**See [LANGSMITH_SETUP.md](LANGSMITH_SETUP.md) for setup instructions.**

---

**Ready to map any concept? Start with `python main_universal.py` and explore the power of AI-driven educational analysis!** 🎓✨
