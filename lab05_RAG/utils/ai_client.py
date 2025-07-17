"""
Azure OpenAI client utilities for Lab05 RAG system
Provides centralized Azure OpenAI client initialization and embedding services
"""

import os
from typing import List, Optional, Tuple
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv()

def get_azure_openai_client() -> AzureOpenAI:
    """
    Get Azure OpenAI client for chat completions
    
    Returns:
        AzureOpenAI: Initialized Azure OpenAI client for chat
    """
    api_key = os.getenv("AOAI_KEY")
    api_url = os.getenv("AOAI_URL")
    
    if not api_key or not api_url:
        raise ValueError("Azure OpenAI API key or URL not configured")
    
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=api_url,
        api_version="2024-02-15-preview"
    )


def get_embedding_client() -> AzureOpenAI:
    """
    Get Azure OpenAI client for embeddings
    
    Returns:
        AzureOpenAI: Initialized Azure OpenAI client for embeddings
    """
    api_key = os.getenv("EMBEDDING_API_KEY")
    api_base = os.getenv("EMBEDDING_URL")
    
    if not api_key or not api_base:
        raise ValueError("Azure OpenAI Embedding API key or URL not configured")
    
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=api_base,
    )


def get_embedding_for_content(content: str, max_retries: int = 2) -> List[float]:
    """
    Get embedding vector for given content using Azure OpenAI service
    
    Args:
        content (str): Text content to generate embedding for
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        List[float]: Embedding vector, empty list if failed
    """
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not embedding_model:
        raise ValueError("EMBEDDING_MODEL not configured")
    
    try_count = max_retries
    while try_count > 0:
        try_count -= 1
        try:
            client = get_embedding_client()
            embedding = client.embeddings.create(
                input=content,
                model=embedding_model,
            )
            return embedding.data[0].embedding
        except Exception as e:
            print(f"❌ Embedding API error: {e}")
            if try_count == 0:
                break
    
    return []


def chat_with_azure_openai(messages: List[dict], tools: Optional[List[dict]] = None, max_retries: int = 3) -> Tuple[object, int, int]:
    """
    Chat with Azure OpenAI GPT model
    
    Args:
        messages (List[dict]): Conversation messages
        tools (Optional[List[dict]]): Available tools for function calling
        max_retries (int): Maximum number of retry attempts
        
    Returns:
        Tuple[object, int, int]: (response_message, input_tokens, output_tokens)
    """
    error_count = max_retries
    
    while error_count > 0:
        error_count -= 1
        try:
            client = get_azure_openai_client()
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                tools=tools,
                tool_choice="auto" if tools else None,
                temperature=0.1,
                max_tokens=2000
            )
            
            return (
                response.choices[0].message,
                response.usage.prompt_tokens,
                response.usage.total_tokens - response.usage.prompt_tokens,
            )
            
        except Exception as e:
            print(f"❌ Azure OpenAI API error: {str(e)}")
            if error_count == 0:
                # Return empty response if all retries failed
                class EmptyMessage:
                    def __init__(self):
                        self.content = ""
                        self.tool_calls = None
                return EmptyMessage(), 0, 0
            continue
    
    # This should not be reached, but just in case
    class EmptyMessage:
        def __init__(self):
            self.content = ""
            self.tool_calls = None
    return EmptyMessage(), 0, 0