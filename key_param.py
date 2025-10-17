# =============================================================================
# KEY_PARAM.PY - API Configuration for Lenroker NVIDIA Integration
# =============================================================================
# This file contains API client configuration for NVIDIA's AI services.
# API keys are now loaded from environment variables for security.
# Lenroker uses two NVIDIA models for optimal performance:
#
# 1. Embeddings: NVIDIA Llama 3.2 NeMo Retriever (300M parameters)
#    - Optimized for document retrieval and semantic search
#    - Converts text chunks into high-quality vector representations
#    - Used by ChromaDB for similarity-based document retrieval
#
# 2. Reasoning: NVIDIA Llama 3.1 Nemotron Nano 8B
#    - Instruction-tuned for complex reasoning and analysis
#    - Powers Lenroker's Multi-Stage Hierarchical RAG system
#    - Optimized for document intelligence and question answering
#
# Security: API keys are loaded from .env file using python-dotenv
# =============================================================================

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API key for NVIDIA embeddings (Llama 3.2 NeMo Retriever)
# Used for: Document chunking, vector embeddings, semantic search
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")

# API key for NVIDIA reasoning model (Llama 3.1 Nemotron Nano 8B)  
# Used for: Question answering, document analysis, multi-stage reasoning
NVIDIA_REASONING_API_KEY = os.getenv("NVIDIA_REASONING_API_KEY")

# Validate that API keys are loaded
if not NVIDIA_API_KEY:
    raise ValueError("NVIDIA_API_KEY not found in environment variables. Please check your .env file.")

if not NVIDIA_REASONING_API_KEY:
    raise ValueError("NVIDIA_REASONING_API_KEY not found in environment variables. Please check your .env file.")

from openai import OpenAI

def get_nvidia_client():
    """
    Get configured OpenAI client for NVIDIA API.
    
    Returns:
        OpenAI: Configured client pointing to NVIDIA's API endpoint
        
    Note: Uses OpenAI-compatible interface for NVIDIA models
    """
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_REASONING_API_KEY
    )

def query_nvidia_model(messages, temperature=0, max_tokens=4096):
    """
    Query NVIDIA reasoning model with optimized configuration and error handling.
    
    This function provides a clean interface to NVIDIA's Llama 3.1 Nemotron Nano 8B
    model for document intelligence and reasoning tasks.
    
    Args:
        messages (list): List of message dictionaries in OpenAI format
            Example: [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        temperature (float): Randomness in response (0.0 = deterministic, 1.0 = creative)
        max_tokens (int): Maximum response length in tokens
        
    Returns:
        str: Generated response from the NVIDIA model or error message
        
    Model Configuration:
        - model: nvidia/llama-3.1-nemotron-nano-8b-v1 (8B parameter instruction-tuned)
        - temperature: 0 (deterministic for consistent document analysis)
        - top_p: 0.95 (nucleus sampling for quality)
        - frequency_penalty: 0 (no repetition penalty)
        - presence_penalty: 0 (no topic penalty)
        - stream: False (synchronous response for easier handling)
        
    Error Handling:
        - Handles API errors and timeouts gracefully
        - Returns descriptive error messages for debugging
        - Rate limiting handled automatically
    """
    try:
        client = get_nvidia_client()
        
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-8b-v1",
            messages=messages,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            frequency_penalty=0,
            presence_penalty=0,
            stream=False  # Set to False for easier handling
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        print(f"NVIDIA API call failed: {e}")
        return "MODEL_ERROR: Unable to process request. Please check your API connection and try again."

