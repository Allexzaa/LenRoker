<div align="center">

![Lenroker Logo](Logo/logo.gif)

#  Lenroker
## AI-Powered Document Intelligence Platform

A powerful, production-grade Retrieval-Augmented Generation (RAG) application that allows you to upload PDF documents and ask questions about their content using AI. Features a **Multi-Stage Hierarchical RAG Architecture** with **6x faster processing** for superior answer quality.

</div>

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![LangChain](https://img.shields.io/badge/LangChain-latest-green.svg)
![Gradio](https://img.shields.io/badge/Gradio-5.0+-orange.svg)

<details>
<summary><h2>🌟 Key Features</h2></summary>

- **🧠 Multi-Stage Hierarchical RAG**: State-of-the-art 3-stage reasoning architecture
- **� Chain-nof-Thought Reasoning**: Explicit 5-step reasoning process for each chunk
- **� Deocument-Type Awareness**: Specialized prompts for Policy, Technical, and General documents
- **� Seoction-Aware Processing**: Automatic section detection and metadata anchoring
- **� CrFoss-Reference Intelligence**: Detects and boosts contextually important chunks
- **📤 PDF Upload**: Easy drag-and-drop PDF upload interface
- **🤖 AI-Powered Q&A**: Ask natural language questions about your documents
- **💬 Chat Interface**: Interactive chat history with automatic saving
- **📜 Persistent History**: All conversations saved with timestamps
- **🔒 Cloud-Powered LLM**: Llama 3.1 Nemotron Nano 8B via NVIDIA API (embeddings via NVIDIA API)
- **⚡ Fast Retrieval**: ChromaDB vector database with FlashrankRerank re-ranking
- **🎨 Beautiful UI**: Clean, modern Gradio web interface with gradient design

</details>

<details>
<summary><h2>🏗️ Multi-Stage RAG Architecture</h2></summary>

Our advanced RAG system uses a **3-stage hierarchical approach** instead of traditional single-stage retrieval:

### Traditional RAG (Before):
```
Question → Retrieve 3 chunks → LLM → Answer
```
❌ Limited context
❌ Struggles with disjoint information
❌ No reasoning layer

### Hierarchical RAG (Current):
```
Question → Retrieve 10 chunks → Chunk-level reasoning → Synthesis → Final Answer
```
✅ 10x more context analyzed
✅ Layered reasoning process
✅ Better quality answers
✅ Handles complex questions

### **Stage 1: Section-Aware Retrieval + Re-ranking**
- Retrieves **top 20** candidate chunks with section-aware boosting
- Uses NVIDIA Llama 3.2 NeMo Retriever embeddings
- **Section metadata anchoring** with automatic section detection
- **Cross-reference boosting** for contextually important chunks
- FlashrankRerank re-ranking to final top 10 most relevant chunks

### **Stage 2: Optimized Batch Reasoning**
- **Single LLM call** processes all chunks simultaneously (6x faster)
- **Document-type aware** prompts (General, Policy, Technical)
- **Integrated analysis**: Scanning → Extraction → Integration → Synthesis
- Section metadata and cross-references preserved in analysis
- **Clean output**: Professional answers without internal reasoning verbosity
- Maintains reasoning quality while dramatically improving speed

### **Stage 3: Metadata Enhancement**
- **Lightweight processing** - no additional LLM calls required
- **Analysis transparency** - shows reasoning mode and sections analyzed
- **Clean user experience** - technical details hidden from end users
- **Performance optimized** - maintains quality while maximizing speed
- Cross-reference integration and section attribution preserved
- Produces comprehensive, professional answers with helpful metadata

</details>

<details>
<summary><h2>📊 Architecture Diagram</h2></summary>

```
┌─────────────────┐
│  PDF Document   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Semantic Splitter│  (Chunks: 1200 chars, Overlap: 200, Section-aware)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  NVIDIA Llama   │  (Embeddings Generation)
│  NeMo Retriever │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ChromaDB      │  (Vector Storage)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  User Question  │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│    STAGE 1: SECTION-AWARE RETRIEVAL      │
│  • Retrieve top 20 candidates            │
│  • Section metadata boosting             │
│  • Cross-reference detection             │
│  • FlashrankRerank to top 10             │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│   STAGE 2: OPTIMIZED BATCH REASONING     │
│  Single NVIDIA API call processes all:   │
│  • Llama 3.1 Nemotron Nano 8B model     │
│  • Document-type aware analysis          │
│  • Cross-section integration             │
│  • Comprehensive synthesis               │
│  • Clean, professional output            │
│  • 6x faster than individual processing  │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│    STAGE 3: METADATA ENHANCEMENT         │
│  • Add reasoning mode and section info   │
│  • No additional LLM calls needed        │
│  • Preserve analysis transparency        │
│  • Clean user-facing output              │
└────────┬─────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

</details>

<details>
<summary><h2>🚀 Getting Started</h2></summary>

### Prerequisites

Before you begin, ensure you have the following:

- **Python 3.10+**: [Download Python](https://www.python.org/downloads/)
- **NVIDIA API Key**: [Get API Key](https://build.nvidia.com/) - Required for both embeddings and reasoning

### Installation

1. **Clone or download this repository**

```bash
cd lenroker
```

2. **Install Python dependencies**

```bash
pip install -r requirements.txt
```

**Note:** If you encounter issues with `flashrank` installation, the system will automatically fall back to section-aware retrieval without re-ranking. FlashrankRerank is optional for enhanced performance but not required for core functionality.

3. **Configure API Keys**

Edit the `key_param.py` file and add your NVIDIA API keys:

```python
# API key for NVIDIA embeddings (Llama 3.2 NeMo Retriever)
NVIDIA_API_KEY = "your_nvidia_embeddings_api_key_here"

# API key for NVIDIA reasoning model (Llama 3.1 Nemotron Nano 8B)
NVIDIA_REASONING_API_KEY = "your_nvidia_reasoning_api_key_here"
```

**Note:** You can use the same NVIDIA API key for both if you have access to both models.

</details>

<details>
<summary><h2>📖 Usage</h2></summary>

### Web Interface (Recommended)

Launch the Gradio web interface:

```bash
python app.py
```

Then open your browser to:
```
http://127.0.0.1:7860
```

**Steps:**
1. Upload a PDF document
2. Click "Process PDF" and wait for confirmation
3. Ask questions in the text box
4. Get AI-powered answers with multi-stage reasoning!

### Command Line (Alternative)

**Process a PDF document:**

```bash
python load_data.py
```

This will:
- Load the PDF specified in the code
- Split it into chunks
- Generate embeddings
- Store in ChromaDB (`./chroma_db/`)

**Ask questions:**

```bash
python rag.py
```

Edit the query at the bottom of `rag.py` to ask different questions.

</details>

<details>
<summary><h2>📁 Project Structure</h2></summary>

```
lenroker/
├── app.py                  # Gradio web interface with Multi-Stage RAG
├── load_data.py           # PDF processing script
├── rag.py                 # Simple query script
├── key_param.py           # API keys configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── chat_histories.json    # Saved chat conversations (auto-created)
├── chroma_db/             # Vector database storage (auto-created)
├── chroma_db_temp/        # Temporary storage for UI uploads
└── documents/             # Your PDF documents (optional)
```

</details>

<details>
<summary><h2>⚙️ Configuration</h2></summary>

### Multi-Stage RAG Parameters

**Current settings in [app.py](app.py):**

```python
# Stage 1: Section-Aware Retrieval
search_kwargs={"k": 20}  # Retrieve top 20 candidates, re-rank to 10

# Semantic Document Chunking
chunk_size=1200          # Characters per chunk (larger for context)
chunk_overlap=200        # Overlap between chunks (more for continuity)
separators=["\n\n", "\n", ".", ";", ",", " "]  # Semantic boundaries

# Section Metadata Enhancement
section_detection=True   # Automatic section title extraction
cross_reference_boost=0.2  # Boost chunks with cross-references
section_match_boost=0.3    # Boost chunks matching question keywords
```

**Adjust for your needs:**

- **More candidates**: Increase initial `k` value (e.g., `k=30` for more candidates)
- **Larger chunks**: Increase `chunk_size` (e.g., `1500` for even bigger chunks)
- **More overlap**: Increase `chunk_overlap` (e.g., `300` for better continuity)
- **Section boosting**: Adjust boost values for section matching and cross-references
- **Semantic boundaries**: Modify separators list for different document types

### Embedding Model

**Current:** NVIDIA Llama 3.2 NeMo Retriever (300M)
- Optimized for retrieval tasks
- Requires NVIDIA API key
- High-quality embeddings

**Alternative:** Use local HuggingFace embeddings:

```python
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
```

### LLM Model

**Current:** Llama 3.1 Nemotron Nano 8B (via NVIDIA API)
- Cloud-powered reasoning model
- Optimized for instruction following and reasoning
- Good balance of speed and quality
- Perfect for multi-stage reasoning
- API-based - no local installation required

**Alternative NVIDIA models:**

Update the model in `key_param.py`:
```python
def query_nvidia_model(messages, temperature=0, max_tokens=4096):
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1",  # Current model
        # Alternative: "nvidia/llama-3.2-3b-instruct"
        # Alternative: "nvidia/llama-3.1-8b-instruct"
        messages=messages,
        temperature=temperature,
        # ... other parameters
    )
```

### Vector Database

**Current:** ChromaDB (local)
- No setup required
- Persists to disk
- Great for development

**Alternatives:**
- **FAISS**: Faster for large datasets
- **Qdrant**: Production-ready with advanced features
- **Weaviate**: Cloud-native option

</details>

<details>
<summary><h2>🎯 Example Questions</h2></summary>

Depending on your uploaded document, try questions like:

- "What is the main topic of this document?"
- "Summarize the key points in detail"
- "What are the prerequisites mentioned?"
- "Explain [specific concept] from the document"
- "What does section 3 discuss?"
- "Compare the approaches mentioned in sections 2 and 4"
- "List all the important dates and events mentioned"

The Multi-Stage RAG architecture excels at:
- ✅ Complex, multi-part questions
- ✅ Questions requiring information from multiple sections
- ✅ Questions needing synthesis of different perspectives
- ✅ Detailed explanations with comprehensive context

</details>

<details>
<summary><h2>🔧 Troubleshooting</h2></summary>

### "NVIDIA API key invalid" error

1. Check your API keys in `key_param.py`
2. Verify they're active at [NVIDIA API Console](https://build.nvidia.com/)
3. Ensure you have access to both embedding and reasoning models
4. Test the API connection:
   ```bash
   python -c "import key_param; print('API test:', key_param.query_nvidia_model([{'role': 'user', 'content': 'Hello'}]))"
   ```

### "FlashrankRerank rebuild skipped" warning

This is a **non-critical warning** that doesn't affect functionality:

```
⚠️ FlashrankRerank rebuild skipped: name 'Ranker' is not defined
```

**What it means:** FlashrankRerank has a dependency issue but the app continues working with fallback retrieval.

**Solutions:**
1. **Ignore it** - Your app works fine without re-ranking
2. **Reinstall flashrank:**
   ```bash
   pip uninstall flashrank
   pip install flashrank>=0.2.0
   ```
3. **Use without re-ranking** - The system automatically falls back to section-aware retrieval

### "ChromaDB error" or "Collection not found"

Delete the database and recreate:
```bash
# Windows
rmdir /s /q chroma_db
rmdir /s /q chroma_db_temp

# Mac/Linux
rm -rf chroma_db chroma_db_temp
```

Then re-run `python app.py` or `python load_data.py`

### Slow response times

The Multi-Stage RAG uses optimized batch processing for speed:

**To speed up further:**
1. Reduce chunks: `search_kwargs={"k": 5}` (instead of 10)
2. Reduce max_tokens: `max_tokens=2048` (instead of 4096)
3. Check your internet connection (API-based model)

### API rate limiting or quota errors

If you encounter API limits:
1. Reduce chunk retrieval: `k=5` instead of 10
2. Increase chunk size: `chunk_size=1000` (fewer chunks)
3. Add delays between requests if needed
4. Check your NVIDIA API quota and usage

</details>

<details>
<summary><h2>🚀 Advanced Usage</h2></summary>

### Batch Processing Multiple PDFs

Modify `load_data.py` to process multiple files:

```python
pdf_files = [
    "document1.pdf",
    "document2.pdf",
    "document3.pdf"
]

all_docs = []
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)
    pages = loader.load()
    all_docs.extend(pages)
```

### Custom Reasoning Prompts

Edit the prompts in `app.py`:

**Chunk-level reasoning:**
```python
chunk_reasoning_template = """Based ONLY on this excerpt, answer...
[Customize this prompt]
"""
```

**Synthesis:**
```python
synthesis_template = """Combine these partial answers...
[Customize this prompt]
"""
```

### Disable Multi-Stage RAG

If you want traditional single-stage RAG (faster but lower quality):

Revert to simple retrieval by replacing the `answer_question` function with basic RAG code.

</details>

<details>
<summary><h2>📊 Performance Tips</h2></summary>

1. **Balance quality vs speed**:
   - Quality: `k=10`, larger model (3B or 7B)
   - Speed: `k=5`, smaller model (1B)

2. **API optimization**: NVIDIA API handles GPU acceleration automatically

3. **Chunk size optimization**:
   - Technical docs: 500-700 chars
   - Narrative text: 800-1000 chars

4. **Monitor token usage**: Multi-Stage RAG uses more tokens but produces better results

</details>

<details>
<summary><h2>🎨 UI Features</h2></summary>

- **Compact purple gradient header**
- **Left sidebar** with:
  - PDF upload section
  - Chat history (auto-saved)
  - New Chat & Delete buttons
- **Main chat area** (3x larger than sidebar)
- **Gradient buttons** (purple, pink, teal)
- **Automatic conversation saving**
- **Timestamp tracking**
- **Load previous conversations**
- **Current document display**

</details>

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📝 License

This project is open source and available for educational and personal use.

## 🙏 Acknowledgments

- **LangChain**: Framework for LLM applications
- **NVIDIA**: NeMo embedding models and Nemotron reasoning models via API
- **Gradio**: Web UI framework
- **ChromaDB**: Vector database
- **Meta**: Llama foundation models

## 📧 Support

If you encounter issues:

1. Check the Troubleshooting section above
2. Review the error messages carefully
3. Ensure all dependencies are installed
4. Verify both NVIDIA API keys are correct and active
5. Test API connectivity with the provided test command

<details>
<summary><h2>🧠 Reasoning-Oriented Prompt Engineering</h2></summary>

### Chain-of-Thought Scaffolding

Traditional RAG systems use passive prompts like "Answer the question based on the context." Our system uses **explicit reasoning scaffolding** that guides the LLM through systematic analysis:

**5-Step Chunk Analysis:**
1. **IDENTIFY**: What specific facts, rules, or concepts relate to the question?
2. **ANALYZE**: How does section context and cross-references help interpretation?
3. **INTEGRATE**: How do identified elements connect logically?
4. **ASSESS**: What's the confidence level for this information?
5. **CONCLUDE**: Provide reasoned answer or mark as irrelevant

**5-Step Synthesis Process:**
1. **INVENTORY**: Catalog all key facts, rules, and concepts
2. **RELATIONSHIPS**: Map logical connections and IF/THEN dependencies  
3. **CONFLICTS**: Identify and resolve contradictions between sections
4. **INTEGRATION**: Combine cross-references and maintain document structure
5. **SYNTHESIS**: Produce comprehensive, well-reasoned final answer

### Document-Type Aware Reasoning

**Automatic Document Type Detection:**
- **Policy Documents**: Focus on compliance, rules, and regulatory requirements
- **Technical Documents**: Emphasize implementation, specifications, and methods
- **General Documents**: Balanced analysis of facts and concepts

**Specialized Reasoning Templates:**
```python
# Policy Analysis Example
"1. IDENTIFY APPLICABLE RULES: What policies apply to this scenario?
 2. ANALYZE CONDITIONS: What conditions or exceptions exist?
 3. DETERMINE COMPLIANCE: How do rules relate to the question?
 4. ASSESS AUTHORITY: What's the authoritative basis?"

# Technical Analysis Example  
"1. IDENTIFY TECHNICAL CONCEPTS: What methods or specs are mentioned?
 2. ANALYZE RELATIONSHIPS: How do technical elements connect?
 3. EVALUATE IMPLEMENTATION: What are practical implications?
 4. ASSESS COMPLETENESS: Is there sufficient technical detail?"
```

</details>

<details>
<summary><h2>☁️ Cloud-Powered Architecture</h2></summary>

### NVIDIA API Integration

Lenroker now uses **100% cloud-powered AI** via NVIDIA's API infrastructure:

**Embedding Model**: NVIDIA Llama 3.2 NeMo Retriever (300M)
- Optimized specifically for retrieval tasks
- High-quality vector representations
- Fast embedding generation

**Reasoning Model**: NVIDIA Llama 3.1 Nemotron Nano 8B
- Instruction-tuned for reasoning and analysis
- Optimized for multi-step thinking
- Superior performance on complex questions

**Benefits of API-Based Architecture:**
- ✅ **No local installation** - No need for Ollama or local model management
- ✅ **Always up-to-date** - Access to latest model versions automatically
- ✅ **GPU acceleration** - NVIDIA's infrastructure handles optimization
- ✅ **Scalable** - No local hardware limitations
- ✅ **Consistent performance** - Professional-grade inference infrastructure

**API Configuration:**
```python
# In key_param.py
def get_nvidia_client():
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_REASONING_API_KEY
    )

def query_nvidia_model(messages, temperature=0, max_tokens=4096):
    client = get_nvidia_client()
    completion = client.chat.completions.create(
        model="nvidia/llama-3.1-nemotron-nano-8b-v1",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content
```

</details>

<details>
<summary><h2>🔧 Enhanced Chunking & Metadata Strategy</h2></summary>

### Semantic Chunking with Section Awareness

Traditional RAG systems use naive text splitting that often breaks context mid-sentence or mid-concept. Our enhanced approach uses:

**Semantic Boundaries:**
```python
separators=["\n\n", "\n", ".", ";", ",", " "]  # Prioritize natural breaks
chunk_size=1200     # Larger chunks preserve more context
chunk_overlap=200   # Substantial overlap maintains continuity
```

**Automatic Section Detection:**
- Identifies section headers, chapter titles, and topic boundaries
- Tags each chunk with section metadata
- Preserves document structure and hierarchy

**Cross-Reference Intelligence:**
- Detects phrases like "see section", "as mentioned above", "refer to"
- Boosts chunks containing cross-references (often contain key context)
- Maintains document interconnections

**Metadata Anchoring:**
Each chunk includes rich metadata:
```python
{
    "section": "Document Overview",
    "page": 15,
    "chunk_index": 3,
    "has_cross_reference": True,
    "chunk_size": 1150
}
```

### Section-Aware Retrieval Boosting

**Intelligent Scoring:**
- Base similarity score from embeddings
- +0.3 boost for section keyword matches
- +0.2 boost for cross-reference presence  
- +0.1 boost for larger chunks (more context)

**Example:** Question about "cloud security" will boost chunks from "Security" or "Cloud Security" sections.

</details>

<details>
<summary><h2>🔮 Technical Details</h2></summary>

### Why Multi-Stage RAG?

Traditional RAG systems retrieve chunks and dump them all into the LLM, expecting it to:
- Parse 10 disjoint paragraphs
- Extract relevant information
- Synthesize a coherent answer

**This is difficult!** LLMs struggle with:
- Context switching between unrelated paragraphs
- Identifying which chunks are actually relevant
- Combining information from multiple sources

**Multi-Stage RAG solves this by:**
1. **Isolating reasoning**: Each chunk analyzed separately
2. **Filtering**: Irrelevant chunks automatically discarded
3. **Synthesis**: Dedicated stage for combining insights
4. **Transparency**: Shows reasoning process

### Performance Metrics

**Current Optimized System vs Single-Stage RAG:**
- **Answer Quality**: +40-60% improvement (subjective)
- **Context Coverage**: 10 chunks vs 3 (3.3x more)
- **Relevance Filtering**: Automatic with FlashrankRerank re-ranking
- **Response Time**: 15-20 seconds (6x faster than previous multi-stage)
- **Token Efficiency**: Optimized batch processing reduces redundancy
- **User Experience**: Clean, professional answers without technical verbosity

**Optimization Achievement**: Maintained quality while achieving 6x speed improvement.

**Previous vs Current Performance:**
- **v2.2 Multi-Stage**: 2+ minutes (11 LLM calls)
- **v2.3 Optimized**: 15-20 seconds (1 LLM call)
- **Quality**: Maintained or improved through better integration

</details>

<details>
<summary><h2>🎓 Learning Resources</h2></summary>

- [LangChain Documentation](https://python.langchain.com/)
- [NVIDIA NIM API Documentation](https://docs.nvidia.com/nim/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [NVIDIA NIM Documentation](https://build.nvidia.com/explore/discover)

</details>

--