# Import required libraries for ChromaDB, LangChain, and document processing
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Import API keys from configuration file
import key_param

# Load PDF document
# PyPDFLoader will extract text from each page of the PDF
loader = PyPDFLoader(r"C:\Users\aniss\Desktop\Website Files\RAG-MongoDB\Cloud Fundamentals 2 Syllabus.pdf")
pages = loader.load()
cleaned_pages = []

# Filter out pages with minimal content (less than 20 words)
# This helps remove empty or irrelevant pages from processing
for page in pages:
    if len(page.page_content.split(" ")) > 20:
        cleaned_pages.append(page)

# Initialize semantic text splitter with section-awareness
# chunk_size=1200: Larger chunks to preserve more context for complex questions
# chunk_overlap=200: More overlap to maintain cross-reference continuity
# separators: Prioritize semantic boundaries (paragraphs, sentences, clauses)
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
            # Extract potential section headers (lines that are short and title-like)
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
                    "source": page.metadata.get('source', 'unknown'),
                    "chunk_size": len(chunk),
                    "has_cross_reference": any(ref in chunk.lower() for ref in 
                        ['see section', 'refer to', 'as mentioned', 'above', 'below', 
                         'previous', 'following', 'chapter', 'page'])
                }
            )
            enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs

# Apply enhanced chunking with metadata
split_docs = extract_section_metadata(cleaned_pages)

# Initialize NVIDIA NeMo embeddings model
# This will convert text chunks into vector embeddings for semantic search
# Using NVIDIA's Llama 3.2 NeMo Retriever model optimized for RAG tasks
embeddings = NVIDIAEmbeddings(
    model="nvidia/llama-3.2-nemoretriever-300m-embed-v1",
    nvidia_api_key=key_param.NVIDIA_API_KEY
)

# Create vector store in ChromaDB (local storage)
# This stores the document chunks along with their vector embeddings
# Data is persisted locally in the ./chroma_db directory
# Enables semantic search and retrieval-augmented generation (RAG)
vectorStore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory="./chroma_db"  # Local folder for vector storage
)

print(f"Successfully created vector store with {len(split_docs)} document chunks!")