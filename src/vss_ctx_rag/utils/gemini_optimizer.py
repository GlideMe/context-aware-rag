# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Google Gemini Basic Integration - Simple retry logic matching Llama/OpenAI approach.

This module provides basic Gemini model integration with simple retry logic,
removing complex optimizations to match the approach used by Llama/OpenAI models.
"""

import os
import time
from typing import Dict, List, Optional, Any
from enum import Enum

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langchain_core.exceptions import LangChainException
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from vss_ctx_rag.utils.ctx_rag_logger import logger


class UseCaseType(Enum):
    """Simple use case types."""
    SUMMARIZATION = "summarization"
    CHAT = "chat"
    NOTIFICATION = "notification"
    ANALYSIS = "analysis"
    GENERAL = "general"


class GeminiOptimizer:
    """Basic Google Gemini integration with langchain wrapper."""
    
    def __init__(self):
        self._client = None
        self._api_key = None
    
    def _initialize_client(self) -> ChatGoogleGenerativeAI:
        """Initialize langchain ChatGoogleGenerativeAI client."""
        if self._client is not None:
            return self._client
            
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "langchain_google_genai library is not installed. "
                "Install it with: pip install langchain_google_genai"
            )
        
        self._api_key = os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable."
            )
        
        try:
            self._client = ChatGoogleGenerativeAI(
                model="models/gemini-2.5-flash",
                google_api_key=self._api_key,
                temperature=0.1,
                max_output_tokens=4096
            )
            logger.info("Gemini langchain client initialized successfully")
            return self._client
        except Exception as e:
            logger.error(f"Failed to initialize Gemini langchain client: {e}")
            raise
    
    def _format_messages_for_langchain(self, messages: List[Dict[str, str]]) -> List:
        """Format messages for langchain, handling system messages properly."""
        formatted_messages = []
        system_content = None
        
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            elif msg.get("role") == "user":
                content = msg.get("content", "")
                # Prepend system message to first user message (langchain pattern)
                if system_content:
                    content = f"{system_content}\n\n{content}"
                    system_content = None  # Only use once
                formatted_messages.append(HumanMessage(content=content))
            elif msg.get("role") == "assistant":
                formatted_messages.append(AIMessage(content=msg.get("content", "")))
        
        return formatted_messages
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        use_case: UseCaseType = UseCaseType.CHAT,
        config_overrides: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response using langchain with basic retry logic."""
        client = self._initialize_client()
        
        # Basic configuration - no complex optimization
        model_name = "models/gemini-2.5-flash"
        max_tokens = 4096
        temperature = 0.1
        top_p = 0.9
        max_retries = 3
        base_delay = 1.0
        
        # Apply simple overrides if provided
        if config_overrides:
            max_tokens = config_overrides.get("max_tokens", max_tokens)
            temperature = config_overrides.get("temperature", temperature)
            top_p = config_overrides.get("top_p", top_p)
        
        # Update client configuration
        client.temperature = temperature
        client.max_output_tokens = max_tokens
        # Note: top_p not directly supported in langchain_google_genai
        
        # Basic retry logic (similar to Llama/OpenAI)
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                # Format messages for langchain
                formatted_messages = self._format_messages_for_langchain(messages)
                
                # Generate response
                start_time = time.time()
                
                logger.info(f"GEMINI BATCH DEBUG: Number of messages being sent: {len(formatted_messages)}")
                total_length = sum(len(str(msg.content)) for msg in formatted_messages)
                logger.info(f"GEMINI BATCH DEBUG: Total conversation length: {total_length} characters")
                logger.info(f"GEMINI BATCH DEBUG: Estimated tokens (rough): {total_length // 4}")
                logger.info(f"GEMINI BATCH DEBUG: max_output_tokens setting: {max_tokens}")
                logger.info(f"GEMINI BATCH DEBUG: temperature: {temperature}, top_p: {top_p}")
                
                # Call langchain client
                response = client.invoke(formatted_messages)
                
                # Extract response content
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Simple response format
                result = {
                    "content": response_text,
                    "model": model_name,
                    "response_time": time.time() - start_time
                }
                
                return result
                
            except LangChainException as e:
                last_exception = e
                error_msg = str(e).lower()

                # Don't retry on authentication or validation errors
                if any(keyword in error_msg for keyword in ['access', 'auth', 'invalid', 'validation']):
                    logger.error(f"Non-retryable langchain error: {e}")
                    raise e

                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Simple exponential backoff
                    logger.warning(f"Gemini attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} Gemini attempts failed. Last error: {e}")
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()

                # Don't retry on authentication or validation errors
                if any(keyword in error_msg for keyword in ['unauthorized', 'forbidden', 'invalid', 'api_key']):
                    logger.error(f"Non-retryable error: {e}")
                    raise e

                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)  # Simple exponential backoff
                    logger.warning(f"Gemini attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {max_retries + 1} Gemini attempts failed. Last error: {e}")
        
        raise last_exception


# Global optimizer instance
gemini_optimizer = GeminiOptimizer()