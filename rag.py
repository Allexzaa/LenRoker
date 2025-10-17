# =============================================================================
# RAG.PY - Simple RAG Query Interface for Lenroker
# =============================================================================
# This file provides a simplified command-line interface for testing the RAG
# system independently of the main Gradio web application.
#
# Key Components:
# - ChromaDB vector store loading from disk
# - NVIDIA embeddings for document retrieval  
# - NVIDIA API for reasoning (Llama 3.1 Nemotron Nano 8B)
# - Section-aware document formatting with metadata
# - 5-step reasoning process for comprehensive analysis
#
# Usage: python rag.py
# Note: Requires existing ChromaDB database from load_data.py
# =============================================================================

# Import ChromaDB and NVIDIA embeddings for vector search
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import key_param
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# Removed OllamaLLM import - now using NVIDIA API directly via key_param



# Initialize the same embeddings model used in load_data.py
embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    nvidia_api_key=key_param.NVIDIA_API_KEY
)

# Load the existing ChromaDB vector store from disk
vectorStore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
def query_data(query):
    """
    Query the document database using RAG (Retrieval-Augmented Generation).
    
    This function implements a simplified version of Lenroker's RAG system
    for command-line testing and development purposes.
    
    Process:
    1. Retrieve top 5 most similar document chunks using NVIDIA embeddings
    2. Format chunks with section metadata for better context
    3. Apply 5-step reasoning template for systematic analysis
    4. Generate response using NVIDIA Llama 3.1 Nemotron Nano 8B
    
    Args:
        query (str): Question to ask about the loaded document
        
    Returns:
        str: AI-generated answer based on retrieved document chunks
        
    Note: Requires ChromaDB database to exist (run load_data.py first)
    """
    # Create section-aware retriever for similarity search
    # k=5: Return top 5 most similar documents (optimized for context vs speed)
    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 5,
        },
    )

    template = """You are a document analysis expert. Use the provided context to reason step-by-step about the question.

CONTEXT WITH METADATA:
{context}

QUESTION: {question}

REASONING PROCESS:
1. IDENTIFY RELEVANT INFORMATION: What specific facts, rules, or concepts in the context relate to the question?

2. ANALYZE SECTION CONTEXT: How do the different sections and their metadata help interpret the information?

3. INTEGRATE CROSS-REFERENCES: Are there any cross-references that provide additional context or connections?

4. LOGICAL SYNTHESIS: How do all the identified elements connect to form a complete answer?

5. CONFIDENCE ASSESSMENT: How confident are you in this answer based on the available context?

If you don't have sufficient information to answer confidently, say so clearly.
Base your answer ONLY on the provided context from the document.

REASONING & FINAL ANSWER:
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs_with_metadata(docs):
        """Format documents with their metadata for better context"""
        formatted = []
        for i, doc in enumerate(docs):
            metadata = doc.metadata
            section = metadata.get('section', 'Unknown Section')
            page = metadata.get('page', 'Unknown')
            has_cross_ref = metadata.get('has_cross_reference', False)
            
            header = f"[Section: {section} | Page: {page}"
            if has_cross_ref:
                header += " | Contains cross-references"
            header += "]"
            
            formatted.append(f"{header}\n{doc.page_content}")
        
        return "\n\n" + "="*50 + "\n\n".join(formatted)

    retrieve = {
        "context": retriever | format_docs_with_metadata, 
        "question": RunnablePassthrough()
        }

    # Use NVIDIA API instead of Ollama
    # Get context documents
    docs = retriever.invoke(query)
    context = format_docs_with_metadata(docs)
    
    # Format the prompt
    prompt_text = custom_rag_prompt.format(context=context, question=query)
    
    messages = [
        {"role": "system", "content": "You are an expert document analyst. Provide clear, comprehensive answers based on the provided context."},
        {"role": "user", "content": prompt_text}
    ]
    
    answer = key_param.query_nvidia_model(messages, temperature=0)
    

    return answer

print(query_data("What are the main topics covered in this document?"))