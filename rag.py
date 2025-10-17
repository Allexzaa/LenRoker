# Import ChromaDB and NVIDIA embeddings for local vector search
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import key_param
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM



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
    # Create section-aware retriever for similarity search
    # k=5: Return top 5 most similar documents (increased for better context)
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

    llm = OllamaLLM(model="llama3.2:3b", temperature=0)

    response_parser = StrOutputParser()

    rag_chain = (
        retrieve
        | custom_rag_prompt
        | llm
        | response_parser
    )

    answer = rag_chain.invoke(query)
    

    return answer

print(query_data("When did MongoDB begin supporting multi-document transactions?"))