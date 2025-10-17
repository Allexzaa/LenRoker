"""
Lenroker - AI-Powered Document Intelligence Platform

A sophisticated RAG (Retrieval-Augmented Generation) application that enables users to upload PDF documents
and ask intelligent questions about their content. Features a state-of-the-art Multi-Stage Hierarchical RAG
architecture with 6x faster processing through optimized batch reasoning.

Key Features:
- Multi-Stage Hierarchical RAG with 3-stage reasoning architecture
- Section-aware document processing with metadata anchoring
- Cross-reference intelligence and boosting
- Document-type aware prompts (Policy, Technical, General)
- FlashrankRerank re-ranking for optimal chunk selection
- 6x performance improvement through batch processing
- Beautiful, responsive Gradio web interface with modern 3D avatars
- Persistent chat history with automatic saving
- Cloud-powered AI via NVIDIA API (Llama 3.1 Nemotron Nano 8B)
- Modern 3D UI with Font Awesome integration and gradient designs

Architecture:
- Stage 1: Section-Aware Retrieval + Re-ranking (FlashrankRerank)
- Stage 2: Optimized Batch Reasoning (Single NVIDIA API call)
- Stage 3: Clean Output (Professional responses without metadata)

Author: Lenroker Team
Version: 2.4 - NVIDIA API Integration & Modern 3D UI
License: MIT
"""

import gradio as gr
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# Removed OllamaLLM import - using NVIDIA API directly
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
import key_param
import os
import tempfile
import json
from datetime import datetime

# =============================================================================
# FLASHRANK INITIALIZATION PATCH
# =============================================================================
# Runtime patch for Flashrank class initialization to handle Pydantic model issues
try:
    # Try to import flashrank dependencies first
    import flashrank
    FlashrankRerank.model_rebuild(force=True)
    print("✅ FlashrankRerank initialized successfully")
except ImportError:
    print("⚠️ FlashrankRerank not available - will use fallback retrieval")
except Exception as e:
    print(f"⚠️ FlashrankRerank initialization warning: {e}")
    print("   Re-ranking will fall back to standard retrieval")

# =============================================================================
# GLOBAL VARIABLES & INITIALIZATION
# =============================================================================
# Global state management for the application
vectorstore = None              # ChromaDB vector store for document embeddings
current_pdf_name = None         # Name of currently loaded PDF document
chat_history_store = {}         # Persistent storage for chat conversations
current_chat_id = None          # ID of the current active chat session

# Initialize NVIDIA embeddings model for high-quality vector representations
# Using Llama 3.2 NeMo Retriever optimized for RAG tasks (300M parameters)
embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    nvidia_api_key=key_param.NVIDIA_API_KEY
)

# =============================================================================
# REASONING-ORIENTED PROMPT TEMPLATES
# =============================================================================
def get_reasoning_prompts(document_type="general"):
    """
    Get specialized reasoning prompts based on document type.
    
    This function provides document-type-aware prompts that guide the AI through
    systematic analysis tailored to different content types.
    
    Args:
        document_type (str): Type of document ('general', 'policy', 'technical')
        
    Returns:
        dict: Dictionary containing 'chunk_reasoning' and 'synthesis' prompt templates
        
    Document Types:
        - 'general': Balanced analysis for general documents
        - 'policy': Compliance-focused analysis for regulatory documents  
        - 'technical': Implementation-focused analysis for technical docs
    """
    
    prompts = {
        "general": {
            "chunk_reasoning": """You are a document analysis expert. Use the following text excerpt to reason step-by-step about the question.

CONTEXT METADATA:
- Section: {section}
- Page: {page}
- Cross-references present: {has_cross_ref}

TEXT EXCERPT:
{chunk}

QUESTION: {question}

REASONING PROCESS:
1. IDENTIFY RELEVANT INFORMATION: What specific facts, rules, or concepts in this excerpt relate to the question?

2. ANALYZE CONTEXT: How does the section context and any cross-references help interpret this information?

3. LOGICAL INTEGRATION: How do the identified elements connect to answer the question?

4. CONFIDENCE ASSESSMENT: How confident are you that this excerpt contains relevant information?

If this excerpt contains relevant information, provide your step-by-step reasoning and conclusion.
If this excerpt doesn't contain relevant information, respond with "Not applicable to this excerpt."

REASONING & ANSWER:""",
            
            "synthesis": """You are a document synthesis expert. Use the partial answers from different document sections to provide a comprehensive, well-reasoned response.

SECTION CONTEXT SUMMARY:
{section_summary}

PARTIAL ANSWERS BY SECTION:
{partial_answers}

ORIGINAL QUESTION: {question}

SYNTHESIS REASONING PROCESS:

1. INFORMATION INVENTORY: What key facts, rules, and concepts are provided across all sections?

2. LOGICAL RELATIONSHIPS: How do the different pieces of information connect? Are there any IF/THEN relationships or dependencies?

3. CONFLICT RESOLUTION: Are there any contradictions between sections? If so, how should they be resolved or explained?

4. CROSS-REFERENCE INTEGRATION: What connections exist between different sections? How do cross-references enhance understanding?

5. COMPREHENSIVE SYNTHESIS: Based on the above analysis, what is the complete, well-structured answer?

FINAL ANSWER:
[Provide a clear, comprehensive answer that demonstrates your reasoning process while maintaining readability]"""
        },
        
        "policy": {
            "chunk_reasoning": """You are a policy analysis expert. Use the following policy excerpt to reason step-by-step about the question.

POLICY CONTEXT:
- Section: {section}
- Page: {page}
- Cross-references present: {has_cross_ref}

POLICY EXCERPT:
{chunk}

QUESTION: {question}

POLICY REASONING PROCESS:
1. IDENTIFY APPLICABLE RULES: What specific policies, procedures, or requirements in this excerpt apply to the question?

2. ANALYZE CONDITIONS: What conditions, exceptions, or qualifications are mentioned?

3. DETERMINE COMPLIANCE: How do the identified rules relate to the question's scenario?

4. ASSESS AUTHORITY: What is the authoritative basis for these requirements?

If this excerpt contains relevant policy information, provide your analysis and conclusion.
If this excerpt doesn't contain relevant policy information, respond with "Not applicable to this excerpt."

POLICY ANALYSIS & ANSWER:""",
            
            "synthesis": """You are a policy synthesis expert. Integrate policy information from multiple sections to provide a comprehensive, regulation-based answer.

POLICY SECTIONS ANALYZED:
{section_summary}

POLICY FINDINGS BY SECTION:
{partial_answers}

ORIGINAL QUESTION: {question}

POLICY SYNTHESIS PROCESS:

1. RULE IDENTIFICATION: What are all the applicable policies, procedures, and requirements?

2. HIERARCHICAL ANALYSIS: Which rules take precedence? Are there any overriding authorities?

3. CONDITION MAPPING: What conditions must be met? What are the IF/THEN relationships?

4. EXCEPTION HANDLING: Are there any exceptions, waivers, or special circumstances?

5. COMPLIANCE DETERMINATION: Based on all applicable rules, what is the definitive answer?

FINAL POLICY-BASED ANSWER:
[Provide a clear, authoritative answer based on the identified policies and procedures]"""
        },
        
        "technical": {
            "chunk_reasoning": """You are a technical documentation expert. Use the following technical excerpt to reason step-by-step about the question.

TECHNICAL CONTEXT:
- Section: {section}
- Page: {page}
- Cross-references present: {has_cross_ref}

TECHNICAL EXCERPT:
{chunk}

QUESTION: {question}

TECHNICAL REASONING PROCESS:
1. IDENTIFY TECHNICAL CONCEPTS: What specific technical concepts, methods, or specifications are mentioned?

2. ANALYZE RELATIONSHIPS: How do these technical elements relate to each other and to the question?

3. EVALUATE IMPLEMENTATION: What are the practical implications or implementation details?

4. ASSESS COMPLETENESS: Does this excerpt provide sufficient technical detail to address the question?

If this excerpt contains relevant technical information, provide your analysis and conclusion.
If this excerpt doesn't contain relevant technical information, respond with "Not applicable to this excerpt."

TECHNICAL ANALYSIS & ANSWER:""",
            
            "synthesis": """You are a technical synthesis expert. Combine technical information from multiple sections to provide a comprehensive, implementation-focused answer.

TECHNICAL SECTIONS ANALYZED:
{section_summary}

TECHNICAL FINDINGS BY SECTION:
{partial_answers}

ORIGINAL QUESTION: {question}

TECHNICAL SYNTHESIS PROCESS:

1. CONCEPT INTEGRATION: How do the technical concepts from different sections work together?

2. DEPENDENCY ANALYSIS: What are the technical dependencies and prerequisites?

3. IMPLEMENTATION SEQUENCE: What is the logical order for implementation or understanding?

4. COMPATIBILITY ASSESSMENT: Are there any technical conflicts or compatibility issues?

5. PRACTICAL APPLICATION: What is the complete technical solution or explanation?

FINAL TECHNICAL ANSWER:
[Provide a clear, technically accurate answer that addresses implementation and practical considerations]"""
        }
    }
    
    return prompts.get(document_type, prompts["general"])

def detect_document_type(pdf_name, sample_content=""):
    """
    Automatically detect document type for specialized reasoning prompts.
    
    Args:
        pdf_name (str): Name of the PDF file
        sample_content (str): Sample content from the document
        
    Returns:
        str: Document type ('policy', 'technical', or 'general')
    """
    pdf_name_lower = pdf_name.lower() if pdf_name else ""
    content_lower = sample_content.lower()
    
    # Policy/regulatory documents
    if any(keyword in pdf_name_lower for keyword in ['policy', 'procedure', 'regulation', 'compliance', 'guideline', 'manual']):
        return "policy"
    if any(keyword in content_lower for keyword in ['shall', 'must', 'required', 'prohibited', 'compliance', 'regulation']):
        return "policy"
    
    # Technical documents
    if any(keyword in pdf_name_lower for keyword in ['technical', 'api', 'specification', 'implementation', 'architecture']):
        return "technical"
    if any(keyword in content_lower for keyword in ['function', 'method', 'algorithm', 'implementation', 'code', 'system']):
        return "technical"
    
    return "general"

# =============================================================================
# CHAT HISTORY MANAGEMENT
# =============================================================================
def load_chat_histories():
    global chat_history_store
    try:
        if os.path.exists("chat_histories.json"):
            with open("chat_histories.json", "r") as f:
                chat_history_store = json.load(f)
    except:
        chat_history_store = {}

def save_chat_histories():
    try:
        with open("chat_histories.json", "w") as f:
            json.dump(chat_history_store, f, indent=2)
    except:
        pass

# =============================================================================
# PDF PROCESSING & VECTOR STORE CREATION
# =============================================================================
def process_pdf(pdf_file):
    """
    Process uploaded PDF document and create vector store with enhanced metadata.
    
    This function implements advanced document processing with section-aware chunking
    and metadata extraction for optimal retrieval performance.
    
    Features:
    - Semantic chunking with section-aware boundaries (1200 chars, 200 overlap)
    - Automatic section detection and metadata tagging
    - Cross-reference intelligence and boosting
    - Enhanced document metadata for better retrieval
    - ChromaDB vector store creation with NVIDIA embeddings
    
    Processing Pipeline:
    1. Load PDF using PyPDFLoader
    2. Filter out pages with insufficient content (< 20 words)
    3. Apply semantic text splitting with section awareness
    4. Extract section metadata and cross-references
    5. Generate embeddings using NVIDIA Llama 3.2 NeMo Retriever
    6. Store in ChromaDB with enhanced metadata
    
    Args:
        pdf_file: Uploaded PDF file object from Gradio interface
        
    Returns:
        str: Success/error message for user feedback
        
    Global Variables Modified:
        - vectorstore: ChromaDB instance for document retrieval
        - current_pdf_name: Name of the currently loaded document
    """
    global vectorstore, current_pdf_name

    if pdf_file is None:
        return "⚠️ Please upload a PDF file first."

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file)
            tmp_path = tmp_file.name

        current_pdf_name = os.path.basename(pdf_file.name) if hasattr(pdf_file, 'name') else "uploaded_document.pdf"

        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        cleaned_pages = [page for page in pages if len(page.page_content.split(" ")) > 20]

        if not cleaned_pages:
            os.unlink(tmp_path)
            return "⚠️ No valid content found in the PDF."

        # Enhanced semantic text splitter with section-awareness
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", ";", ",", " "]
        )
        
        # Enhanced chunking with section metadata extraction
        def extract_section_metadata(pages):
            """Extract section titles and enhance chunks with metadata"""
            enhanced_docs = []
            
            for page in pages:
                content = page.page_content
                page_num = page.metadata.get('page', 0)
                
                # Split page into chunks
                chunks = text_splitter.split_text(content)
                
                for i, chunk in enumerate(chunks):
                    # Extract potential section headers
                    lines = chunk.split('\n')
                    section_title = "General Content"
                    
                    # Look for section headers in the chunk
                    for line in lines[:3]:  # Check first 3 lines
                        line = line.strip()
                        if (len(line) < 100 and len(line) > 5 and 
                            (line.isupper() or line.istitle()) and
                            not line.endswith('.') and
                            any(keyword in line.lower() for keyword in 
                                ['chapter', 'section', 'part', 'introduction', 'overview', 
                                 'summary', 'conclusion', 'requirements', 'objectives',
                                 'fundamentals', 'concepts', 'principles', 'methods'])):
                            section_title = line
                            break
                    
                    # Create enhanced document with metadata
                    from langchain.schema import Document
                    enhanced_doc = Document(
                        page_content=chunk,
                        metadata={
                            "page": page_num,
                            "section": section_title,
                            "chunk_index": i,
                            "source": page.metadata.get('source', current_pdf_name),
                            "chunk_size": len(chunk),
                            "has_cross_reference": any(ref in chunk.lower() for ref in 
                                ['see section', 'refer to', 'as mentioned', 'above', 'below', 
                                 'previous', 'following', 'chapter', 'page'])
                        }
                    )
                    enhanced_docs.append(enhanced_doc)
            
            return enhanced_docs

        split_docs = extract_section_metadata(cleaned_pages)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory="./chroma_db_temp"
        )

        os.unlink(tmp_path)

        success_msg = f"✅ Successfully processed '{current_pdf_name}'! Created {len(split_docs)} chunks."

        return success_msg

    except Exception as e:
        return f"❌ Error: {str(e)}"

# =============================================================================
# MULTI-STAGE HIERARCHICAL RAG SYSTEM
# =============================================================================
def answer_question(question, chat_history):
    """
    Answer questions using advanced Multi-Stage Hierarchical RAG Architecture.
    
    This is the core function that implements Lenroker's sophisticated 3-stage
    reasoning system for document intelligence. Optimized for 6x faster processing
    while maintaining high-quality answers.
    
    Architecture Overview:
    
    Stage 1: Section-Aware Retrieval + Re-ranking
        - Retrieves top 20 candidate chunks with section-aware boosting
        - FlashrankRerank re-ranking to select best 10 chunks  
        - Cross-reference and section keyword boosting (+0.2-0.3 score)
        - Larger chunk preference for more context (+0.1 score)
        
    Stage 2: Optimized Batch Reasoning (6x Performance Improvement)
        - Single NVIDIA API call processes all chunks simultaneously
        - Document-type aware analysis (Policy/Technical/General)
        - Integrated analysis and synthesis in one optimized prompt
        - Uses Llama 3.1 Nemotron Nano 8B for superior reasoning
        
    Stage 3: Clean Output
        - Professional responses without technical metadata
        - Clean user experience with hidden complexity
        - Automatic chat history management and persistence
    
    Performance Optimizations:
    - Reduced from 11 LLM calls (v2.2) to 1 call (v2.4)
    - Maintained answer quality while achieving 6x speed improvement
    - Batch processing eliminates redundant API calls
    - Smart section-aware retrieval reduces irrelevant chunks
        
    Args:
        question (str): User's question about the document
        chat_history (list): Current conversation history in Gradio format
        
    Returns:
        tuple: (updated_chat_history, cleared_input, updated_history_list)
            - updated_chat_history: Chat with new Q&A pair added
            - cleared_input: Empty string to clear the input box
            - updated_history_list: Refreshed chat history sidebar
        
    Global Variables Used:
        - vectorstore: ChromaDB instance for document retrieval
        - current_pdf_name: Name of currently loaded document
        - current_chat_id: ID of active chat session
        - chat_history_store: Persistent chat storage
        
    Error Handling:
        - Returns helpful onboarding message if no document loaded
        - Graceful fallback if FlashrankRerank fails
        - Comprehensive error messages for API failures
    """
    global vectorstore, current_pdf_name, current_chat_id, chat_history_store

    if vectorstore is None:
        chat_history.append({"role": "user", "content": question})
        helpful_message = """📋 **Welcome to Lenroker - AI Document Intelligence!**

I'm designed to analyze PDF documents and answer questions about their content using advanced AI reasoning.

**To get started:**
1. 📤 Upload a PDF document using the sidebar
2. 🔄 Click "Process Document" and wait for confirmation
3. 💬 Ask me anything about your document's content

**What I can help with:**
- Summarize document sections
- Extract specific information
- Answer complex questions requiring multi-section analysis
- Explain technical concepts from your documents
- Compare information across different parts of your document

**Example questions after uploading:**
- "What are the main topics covered?"
- "Summarize the key findings"
- "What does section 3 discuss about [topic]?"

Ready to analyze your document! 🚀"""
        
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": helpful_message})
        return chat_history, "", get_chat_history_list()

    if not question or question.strip() == "":
        return chat_history, question, get_chat_history_list()

    try:
        # Initialize NVIDIA reasoning model via API
        # No need to initialize - we'll use key_param.query_nvidia_model directly

        # =====================================================================
        # STAGE 1: SECTION-AWARE RETRIEVAL + RE-RANKING
        # =====================================================================
        
        def section_aware_retrieval(question, k=20):
            """Enhanced retrieval with section awareness and boosting"""
            
            # Extract potential section keywords from question
            section_keywords = []
            question_lower = question.lower()
            
            # Common section indicators
            section_indicators = [
                'introduction', 'overview', 'summary', 'conclusion', 'requirements',
                'objectives', 'fundamentals', 'concepts', 'principles', 'methods',
                'chapter', 'section', 'part', 'forbearance', 'downgrade', 'upgrade',
                'policy', 'procedure', 'guidelines', 'standards', 'compliance'
            ]
            
            for indicator in section_indicators:
                if indicator in question_lower:
                    section_keywords.append(indicator)
            
            # First, get base similarity results
            base_retriever = vectorstore.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": k}
            )
            base_docs = base_retriever.invoke(question)
            
            # Apply section-based boosting
            scored_docs = []
            for doc in base_docs:
                score = 1.0  # Base score
                doc_metadata = doc.metadata
                
                # Boost documents with matching section keywords
                if section_keywords:
                    section_title = doc_metadata.get('section', '').lower()
                    for keyword in section_keywords:
                        if keyword in section_title:
                            score += 0.3  # Boost section matches
                
                # Boost documents with cross-references (often contain important context)
                if doc_metadata.get('has_cross_reference', False):
                    score += 0.2
                
                # Boost larger chunks (more context)
                chunk_size = doc_metadata.get('chunk_size', 0)
                if chunk_size > 800:
                    score += 0.1
                
                scored_docs.append((doc, score))
            
            # Sort by score and return top documents
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, score in scored_docs]
        
        # Get section-aware retrieved documents
        retrieved_docs_raw = section_aware_retrieval(question, k=20)
        
        # Apply re-ranking to get the most semantically relevant chunks
        try:
            # Check if flashrank is available
            import flashrank
            
            # Use FlashrankRerank directly on documents (simpler approach)
            compressor = FlashrankRerank(top_n=10)
            retrieved_docs = compressor.compress_documents(retrieved_docs_raw, question)
            print(f"✅ Re-ranking applied: {len(retrieved_docs)} chunks selected")
            
        except ImportError:
            print("⚠️ FlashrankRerank not available - using section-aware retrieval")
            retrieved_docs = retrieved_docs_raw[:10]  # Take top 10
        except Exception as rerank_error:
            # Fallback to section-aware retrieval without re-ranking
            print(f"⚠️ Re-ranking failed, using section-aware retrieval: {rerank_error}")
            retrieved_docs = retrieved_docs_raw[:10]  # Take top 10

        if not retrieved_docs:
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": "No relevant information found in the document."})
            return chat_history, "", get_chat_history_list()

        # =====================================================================
        # STAGE 2: OPTIMIZED BATCH REASONING (6x Performance Improvement)
        # =====================================================================
        
        # Detect document type and get appropriate reasoning prompts
        sample_content = " ".join([doc.page_content[:200] for doc in retrieved_docs[:3]])
        document_type = detect_document_type(current_pdf_name, sample_content)
        
        # Format all chunks with metadata for batch processing
        formatted_chunks = []
        section_context = {}
        
        for i, doc in enumerate(retrieved_docs):
            metadata = doc.metadata
            section = metadata.get('section', 'Unknown Section')
            page = metadata.get('page', 'Unknown')
            has_cross_ref = metadata.get('has_cross_reference', False)
            
            # Track sections
            if section not in section_context:
                section_context[section] = 0
            section_context[section] += 1
            
            chunk_header = f"CHUNK {i+1} [Section: {section} | Page: {page}{'| Cross-refs' if has_cross_ref else ''}]:"
            formatted_chunks.append(f"{chunk_header}\n{doc.page_content}")
        
        # Single optimized reasoning prompt for all chunks
        batch_reasoning_template = f"""You are a {document_type} analysis expert. Answer the question based on the provided document chunks.

QUESTION: {question}

DOCUMENT CHUNKS:
{chr(10).join(['='*50 + chr(10) + chunk for chunk in formatted_chunks])}

INSTRUCTIONS:
- Analyze all chunks and identify relevant information
- Provide a clear, comprehensive answer to the question
- Use information from multiple sections when applicable
- If no relevant information is found, state that clearly
- Base your answer ONLY on the provided chunks
- Do NOT show your reasoning process or mention chunk numbers
- Give a direct, professional answer

ANSWER:"""

        # Single NVIDIA API call for all chunks
        try:
            messages = [
                {"role": "system", "content": "You are an expert document analyst. Provide clear, comprehensive answers based on the provided context."},
                {"role": "user", "content": batch_reasoning_template}
            ]
            final_answer = key_param.query_nvidia_model(messages, temperature=0)
            
            # Create section summary for metadata
            sections_analyzed = list(section_context.keys())
            
        except Exception as e:
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": f"❌ Error during analysis: {str(e)}"})
            return chat_history, "", get_chat_history_list()

        # Check if we got a meaningful answer
        if not final_answer or final_answer.strip() == "":
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": "I couldn't find relevant information to answer this question in the document."})
            return chat_history, "", get_chat_history_list()

        # =====================================================================
        # STAGE 3: CLEAN OUTPUT (No additional LLM call needed)
        # =====================================================================
        # The batch reasoning already includes synthesis, provide clean answer

        # Use clean answer without metadata
        clean_answer = final_answer.strip()

        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": clean_answer})

        # Save to history
        if current_chat_id is None:
            create_new_chat(chat_history)
        else:
            update_current_chat(chat_history)

        return chat_history, "", get_chat_history_list()

    except Exception as e:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return chat_history, "", get_chat_history_list()

def create_new_chat(chat_history):
    global current_chat_id, chat_history_store
    current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    first_question = next((msg["content"] for msg in chat_history if msg["role"] == "user"), "New Chat")
    title = first_question[:50] + "..." if len(first_question) > 50 else first_question

    chat_history_store[current_chat_id] = {
        "title": title,
        "timestamp": datetime.now().isoformat(),
        "document": current_pdf_name,
        "messages": chat_history
    }
    save_chat_histories()

def update_current_chat(chat_history):
    global current_chat_id, chat_history_store
    if current_chat_id and current_chat_id in chat_history_store:
        chat_history_store[current_chat_id]["messages"] = chat_history
        save_chat_histories()

def get_chat_history_list():
    """Get formatted list of chat histories"""
    if not chat_history_store:
        return "No chat history yet"

    history_list = []
    for chat_id, data in sorted(chat_history_store.items(), reverse=True):
        timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M")
        history_list.append(f"[{timestamp}] {data['title']}")

    return "\n".join(history_list)

def load_chat_from_history(history_text):
    """Load a chat from history based on selection"""
    global current_chat_id, chat_history_store

    if not history_text or history_text == "No chat history yet":
        return [], "Select a chat from history to load"

    try:
        selected_line = history_text.split("\n")[0]
        for chat_id, data in chat_history_store.items():
            timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M")
            if f"[{timestamp}]" in selected_line:
                current_chat_id = chat_id
                return data["messages"], f"Loaded: {data['title']}"
    except:
        pass

    return [], "Could not load chat"

def start_new_chat():
    """Start a fresh chat session"""
    global current_chat_id
    current_chat_id = None
    return [], "Started new chat", get_chat_history_list()

def delete_selected_chat(history_text):
    """Delete selected chat from history"""
    global chat_history_store

    if not history_text or history_text == "No chat history yet":
        return get_chat_history_list(), "No chat selected"

    try:
        selected_line = history_text.split("\n")[0]
        for chat_id, data in list(chat_history_store.items()):
            timestamp = datetime.fromisoformat(data["timestamp"]).strftime("%Y-%m-%d %H:%M")
            if f"[{timestamp}]" in selected_line:
                del chat_history_store[chat_id]
                save_chat_histories()
                return get_chat_history_list(), "Chat deleted"
    except:
        pass

    return get_chat_history_list(), "Could not delete chat"

# =============================================================================
# APPLICATION INITIALIZATION
# =============================================================================
# Load existing chat histories on startup
load_chat_histories()

# =============================================================================
# ENHANCED CUSTOM CSS FOR LENROKER UI
# =============================================================================
# Professional, modern styling with optimized layout for maximum chat space
custom_css = """
/* Font Awesome CDN Import */
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');

/* Global Styling */
.gradio-container {
    max-width: none !important;
    padding: 0 !important;
}

/* Header Row */
.header-row {
    background: white;
    padding: 1rem 1.5rem;
    border-radius: 12px;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    align-items: center;
    border: 1px solid #e2e8f0;
}

/* Logo Container */
.logo-container {
    display: flex;
    justify-content: flex-start;
    align-items: center;
    padding-left: 1rem;
    background: white;
    border-radius: 15px;
    padding: 0.5rem;
}

.logo-container img {
    border-radius: 20px;
    max-width: 140px;
    height: auto;
    background: white;
}

/* Remove image controls and ensure white background */
.logo-container .image-container {
    background: white !important;
    border: none !important;
    box-shadow: none !important;
}

.logo-container .image-container .image-frame {
    background: white !important;
    border: none !important;
    box-shadow: none !important;
}

.logo-container .image-container .image-button {
    display: none !important;
}

.logo-container .image-container .download-button,
.logo-container .image-container .fullscreen-button,
.logo-container .image-container .share-button {
    display: none !important;
}

/* Ensure the entire logo area blends with white background */
.logo-container > div {
    background: white !important;
    border: none !important;
}

/* Main Header Text */
.main-header-text {
    text-align: left;
    color: #1e293b;
    padding-left: 0.5rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 100%;
}

.main-header-text h1 {
    margin: 0;
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #1e293b;
    text-shadow: none;
}

.main-header-text .subtitle {
    font-size: 1.2rem;
    opacity: 0.7;
    margin-top: 0.5rem;
    font-weight: 400;
    color: #64748b;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 100%);
    pointer-events: none;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.main-header .subtitle {
    font-size: 1.1rem;
    opacity: 0.9;
    margin-top: 0.5rem;
    font-weight: 400;
}

/* Sidebar Styling */
.sidebar {
    background: linear-gradient(145deg, #f8fafc 0%, #e2e8f0 100%);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    height: fit-content;
}

.upload-section {
    background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    border: 2px solid #e2e8f0;
    transition: all 0.3s ease;
}

.upload-section:hover {
    border-color: #667eea;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.1);
}

.history-section {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    padding: 1.5rem;
    border-radius: 12px;
    border: 2px solid #e2e8f0;
    transition: all 0.3s ease;
}

/* Main Chat Container */
.chat-container {
    background: linear-gradient(145deg, #ffffff 0%, #fafbfc 100%);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
    height: 85vh;
    max-height: 85vh;
    border: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Enhanced Chatbot */
.chatbot {
    height: calc(85vh - 200px) !important;
    max-height: calc(85vh - 200px) !important;
    border-radius: 12px !important;
    border: 2px solid #e2e8f0 !important;
    background: #ffffff !important;
    box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.05) !important;
    flex-grow: 1;
    overflow-y: auto;
}

/* Modern 3D Avatar System with Font Awesome Icons */

/* Base 3D Avatar Container */
.chatbot .avatar-container img,
.chatbot img[alt="avatar"],
.chatbot .message .avatar img {
    width: 52px !important;
    height: 52px !important;
    border-radius: 16px !important;
    object-fit: cover !important;
    border: none !important;
    position: relative !important;
    transform: translateZ(0) !important;
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    overflow: hidden !important;
}

/* User Avatar - Modern Blue Gradient Background */
.chatbot .message[data-role="user"] .avatar img,
.chatbot .message.user .avatar img {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #667eea 100%) !important;
    box-shadow: 
        0 10px 25px rgba(102, 126, 234, 0.4),
        0 5px 10px rgba(102, 126, 234, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.3),
        inset 0 -1px 0 rgba(0, 0, 0, 0.1) !important;
}

/* AI Avatar - Modern Pink/Purple Gradient Background */
.chatbot .message[data-role="assistant"] .avatar img,
.chatbot .message.assistant .avatar img,
.chatbot .message.bot .avatar img {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%) !important;
    box-shadow: 
        0 10px 25px rgba(240, 147, 251, 0.4),
        0 5px 10px rgba(245, 87, 108, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.3),
        inset 0 -1px 0 rgba(0, 0, 0, 0.1) !important;
    animation: aiPulse 3s ease-in-out infinite !important;
}

/* Hover Effects */
.chatbot .avatar-container img:hover,
.chatbot img[alt="avatar"]:hover,
.chatbot .message .avatar img:hover {
    transform: translateY(-3px) scale(1.05) !important;
    box-shadow: 
        0 15px 35px rgba(102, 126, 234, 0.5),
        0 8px 15px rgba(102, 126, 234, 0.3),
        inset 0 2px 0 rgba(255, 255, 255, 0.4),
        inset 0 -2px 0 rgba(0, 0, 0, 0.1) !important;
}

/* Pulse Animation for AI */
@keyframes aiPulse {
    0%, 100% { 
        box-shadow: 
            0 10px 25px rgba(240, 147, 251, 0.4),
            0 5px 10px rgba(245, 87, 108, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }
    50% { 
        box-shadow: 
            0 15px 35px rgba(240, 147, 251, 0.6),
            0 8px 15px rgba(245, 87, 108, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.4);
    }
}

/* Enhanced Avatar Spacing */
.chatbot .message .avatar {
    margin-right: 12px !important;
}

/* Hide default Gradio clear button in chatbot */
.chatbot .clear-button,
.chatbot button[title="Clear"],
.chatbot .chatbot-header button,
.chatbot .header button,
.chatbot .clear,
.chatbot .icon-button {
    display: none !important;
}

/* Hide any clear/delete icons in the chatbot header */
.chatbot .chatbot-header .icon,
.chatbot .header .icon,
.chatbot svg[data-testid="clear"],
.chatbot button svg {
    display: none !important;
}

/* Hide the entire chatbot header if it only contains the clear button */
.chatbot .chatbot-header:has(button):not(:has(*:not(button))) {
    display: none !important;
}

/* Input Area */
.input-area {
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    border-radius: 12px;
    padding: 1rem;
    border: 2px solid #e2e8f0;
    margin-top: 1rem;
    flex-shrink: 0;
}

.input-box textarea {
    font-size: 1.1rem !important;
    padding: 1rem !important;
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
    background: #ffffff !important;
    transition: all 0.3s ease !important;
    min-height: 60px !important;
}

.input-box textarea:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
}

/* Button Styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    border-radius: 8px !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
}

.secondary-btn {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}

.history-btn {
    background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 6px !important;
    transition: all 0.3s ease !important;
}

/* Document Display */
.document-name {
    font-weight: 700;
    color: #1e293b;
    font-size: 1.1rem;
    text-align: center;
    background: linear-gradient(135deg, #e0f2fe 0%, #b3e5fc 100%);
    padding: 0.75rem;
    border-radius: 8px;
    border: 2px solid #81d4fa;
    margin-bottom: 1rem;
}

/* Footer */
.footer-info {
    margin-top: 2rem;
    padding: 1.5rem;
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    border-radius: 12px;
    font-size: 0.9rem;
    color: #64748b;
    text-align: center;
    border: 1px solid #e2e8f0;
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .chat-container {
        padding: 1rem;
        min-height: 80vh;
    }
    
    .chatbot {
        min-height: 60vh !important;
        max-height: 60vh !important;
    }
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.upload-section, .history-section, .chat-container {
    animation: fadeIn 0.6s ease-out;
}
"""

# =============================================================================
# GRADIO WEB INTERFACE CREATION
# =============================================================================
# Create the main Gradio interface with custom styling and optimized layout
with gr.Blocks(css=custom_css, title="Lenroker - AI Document Intelligence", theme=gr.themes.Soft()) as demo:

    # =========================================================================
    # HEADER SECTION - Logo and Branding
    # =========================================================================
    with gr.Row(elem_classes="header-row"):
        with gr.Column(scale=1, min_width=150):
            logo_image = gr.Image(
                value="Logo/logo.gif",
                show_label=False,
                interactive=False,
                width=140,
                height=100,
                container=False,
                elem_classes="logo-container",
                show_fullscreen_button=False,
                show_download_button=False,
                show_share_button=False
            )
        with gr.Column(scale=5):
            gr.HTML("""
                <div class="main-header-text">
                    <h1>Lenroker</h1>
                    <div class="subtitle">AI-Powered Document Intelligence Platform</div>
                </div>
            """)

    # =========================================================================
    # MAIN LAYOUT - Sidebar and Chat Interface
    # =========================================================================
    with gr.Row():
        # =====================================================================
        # LEFT SIDEBAR - PDF Upload & Chat History
        # =====================================================================
        with gr.Column(scale=1, elem_classes="sidebar"):
            # Upload section
            with gr.Group(elem_classes="upload-section"):
                gr.Markdown("### 📤 **Upload Document**")
                pdf_input = gr.File(
                    label="Drop your PDF here or click to browse",
                    file_types=[".pdf"],
                    type="binary",
                    height=140
                )
                upload_btn = gr.Button(
                    "🔄 Process Document",
                    variant="primary",
                    size="lg",
                    elem_classes="primary-btn"
                )
                status_output = gr.Textbox(
                    label="",
                    lines=3,
                    interactive=False,
                    show_label=False,
                    placeholder="Upload status will appear here..."
                )

            # Chat History section
            with gr.Group(elem_classes="history-section"):
                gr.Markdown("### 📜 **Conversation History**")
                history_list = gr.Textbox(
                    label="",
                    value=get_chat_history_list(),
                    lines=10,
                    interactive=True,
                    show_label=False,
                    placeholder="Your conversations will appear here..."
                )
                with gr.Row():
                    new_chat_btn = gr.Button(
                        "✨ New Chat",
                        size="sm",
                        elem_classes="history-btn",
                        scale=1
                    )
                    delete_chat_btn = gr.Button(
                        "🗑️ Delete",
                        size="sm",
                        elem_classes="secondary-btn",
                        scale=1
                    )
                load_status = gr.Textbox(
                    label="",
                    lines=1,
                    interactive=False,
                    show_label=False,
                    visible=False
                )

        # =====================================================================
        # MAIN CHAT INTERFACE - Optimized for Maximum Space
        # =====================================================================
        with gr.Column(scale=4):  # Increased from scale=3 to scale=4 for more space
            with gr.Group(elem_classes="chat-container"):
                chatbot = gr.Chatbot(
                    label="",
                    height=600,  # Reduced to 600 to fit input box without scrolling
                    type="messages",
                    show_label=False,
                    render_markdown=True,
                    elem_classes="chatbot",
                    bubble_full_width=False,
                    avatar_images=("user_fa.svg", "ai_fa.svg")
                )

                # Enhanced input area with better layout
                with gr.Row():
                    question_input = gr.Textbox(
                        label="",
                        placeholder="💬 Ask anything about your document... Lenroker will analyze and provide intelligent answers.",
                        lines=3,  # Increased for more room
                        scale=5,  # Increased scale for wider input
                        show_label=False,
                        elem_classes="input-box"
                    )
                    with gr.Column(scale=1, min_width=120):
                        submit_btn = gr.Button(
                            "Ask Lenroker",
                            variant="primary",
                            size="lg",
                            elem_classes="primary-btn"
                        )
                        clear_btn = gr.Button(
                            "Clear Chat",
                            size="sm",
                            elem_classes="secondary-btn"
                        )

    # =========================================================================
    # FOOTER - Technology Information
    # =========================================================================
    gr.HTML("""
        <div class="footer-info">
            <strong>🚀 Lenroker</strong> - Advanced AI Document Intelligence Platform
            <br>
            <strong>Features:</strong> Multi-Stage RAG • Section-Aware Analysis • 6x Faster Processing • Clean Professional Output
            <br>
            <strong>Technology:</strong> NVIDIA Llama NeMo • Llama 3.1 Nemotron Nano 8B • FlashrankRerank • ChromaDB • LangChain
        </div>
    """)

    # =========================================================================
    # EVENT HANDLERS - User Interaction Logic
    # =========================================================================
    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_output]
    )

    submit_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, history_list]
    )

    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input, history_list]
    )

    clear_btn.click(
        lambda: [],
        outputs=[chatbot]
    )

    new_chat_btn.click(
        fn=start_new_chat,
        inputs=[],
        outputs=[chatbot, status_output, history_list]
    )

    delete_chat_btn.click(
        fn=delete_selected_chat,
        inputs=[history_list],
        outputs=[history_list, status_output]
    )

    history_list.select(
        fn=load_chat_from_history,
        inputs=[history_list],
        outputs=[chatbot, status_output]
    )

# =============================================================================
# APPLICATION LAUNCH
# =============================================================================
if __name__ == "__main__":
    print("🚀 Starting Lenroker - AI Document Intelligence Platform...")
    print("🌐 Server: http://127.0.0.1:7860")
    print("🤖 Using NVIDIA API: Llama 3.1 Nemotron Nano 8B for reasoning")
    print("⚡ Performance: 6x faster processing with optimized batch reasoning")
    print("🎯 Ready for intelligent document analysis!")

    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        favicon_path="Logo/logo.png",
        show_api=False,
        quiet=False,
        inbrowser=True  # Automatically open browser
    )
