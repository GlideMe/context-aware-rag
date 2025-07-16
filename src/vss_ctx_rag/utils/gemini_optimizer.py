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
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
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
    """Basic Google Gemini integration with simple retry logic."""
    
    def __init__(self):
        self._initialized = False
    
    def _initialize_client(self) -> None:
        """Initialize Gemini client with API key."""
        if self._initialized:
            return
            
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Google Generative AI library is not installed. "
                "Install it with: pip install google-generativeai"
            )
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable."
            )
        
        try:
            genai.configure(api_key=api_key)
            self._initialized = True
            logger.info("Gemini client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            raise
    
    def _format_messages_for_gemini(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Format messages for Gemini API, handling system messages properly."""
        formatted_messages = []
        system_content = None
        
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            elif msg.get("role") == "user":
                content = msg.get("content", "")
                # Prepend system message to first user message (Gemini pattern)
                if system_content:
                    content = f"{system_content}\n\n{content}"
                    system_content = None  # Only use once
                formatted_messages.append({
                    "role": "user",
                    "parts": [content]
                })
            elif msg.get("role") == "assistant":
                formatted_messages.append({
                    "role": "model",
                    "parts": [msg.get("content", "")]
                })
        
        return formatted_messages
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        use_case: UseCaseType = UseCaseType.CHAT,
        config_overrides: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response with basic retry logic (matching Llama/OpenAI approach)."""
        self._initialize_client()
        
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
        
        # Basic retry logic (similar to Llama/OpenAI)
        last_exception = None
        for attempt in range(max_retries + 1):
            try:
                # Format messages for Gemini
                formatted_messages = self._format_messages_for_gemini(messages)
                
                # Create model instance
                model = genai.GenerativeModel(model_name)
                
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                # Basic safety settings (permissive like other models)
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                }

                # Generate response
                start_time = time.time()
                if len(formatted_messages) == 1:
                    logger.info(f"GEMINI BATCH DEBUG: Number of messages being sent: {len(formatted_messages)}")
                    if len(formatted_messages) == 1:
                        # Single message case
                        content = formatted_messages[0]["parts"][0]
                        logger.info(f"GEMINI BATCH DEBUG: Single message content length: {len(content)} characters")
                        logger.info(f"GEMINI BATCH DEBUG: Content preview: {content[:200]}...")
                        logger.info(f"GEMINI BATCH DEBUG: Estimated tokens (rough): {len(content) // 4}")
                    else:
                        # Multi-turn case
                        total_length = 0
                        for i, msg in enumerate(formatted_messages):
                            content = msg["parts"][0]
                            total_length += len(content)
                            logger.info(f"GEMINI BATCH DEBUG: Message {i+1} ({msg['role']}): {len(content)} characters")
                        logger.info(f"GEMINI BATCH DEBUG: Total conversation length: {total_length} characters")
                        logger.info(f"GEMINI BATCH DEBUG: Estimated tokens (rough): {total_length // 4}")

                    logger.info(f"GEMINI BATCH DEBUG: max_output_tokens setting: {max_tokens}")
                    logger.info(f"GEMINI BATCH DEBUG: temperature: {temperature}, top_p: {top_p}")
                    # Single message
                    response = model.generate_content(
                        formatted_messages[0]["parts"][0],
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    # ADD THESE LOGS RIGHT AFTER the model.generate_content() calls:
                    logger.info(f"GEMINI LLM SAFETY DEBUG: Response finish_reason = {getattr(response, 'finish_reason', 'N/A')}")
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]  # <- ADD THIS LINE
                        logger.info(f"GEMINI LLM SAFETY DEBUG: Candidate finish_reason = {getattr(candidate, 'finish_reason', 'N/A')}")
                        if hasattr(candidate, 'safety_ratings'):
                            logger.info(f"GEMINI LLM SAFETY DEBUG: Safety ratings = {candidate.safety_ratings}")
                            logger.info(f"GEMINI LLM SAFETY DEBUG: Safety ratings type = {type(candidate.safety_ratings)}")
                            logger.info(f"GEMINI LLM SAFETY DEBUG: Safety ratings length = {len(candidate.safety_ratings) if candidate.safety_ratings else 'None'}")
                            if candidate.safety_ratings:
                                for i, rating in enumerate(candidate.safety_ratings):
                                    logger.info(f"GEMINI LLM SAFETY DEBUG: Rating {i}: {rating}")
                        else:
                            logger.info(f"GEMINI LLM SAFETY DEBUG: No safety_ratings attribute found")
                else:
                    # Multi-turn conversation
                    logger.info(f"GEMINI LLM SAFETY DEBUG: Safety settings being sent (multi-turn):")
                    for category, threshold in safety_settings.items():
                        logger.info(f"GEMINI LLM SAFETY DEBUG: {category.name} = {threshold.name}")
                    chat = model.start_chat(history=formatted_messages[:-1])
                    response = chat.send_message(
                        formatted_messages[-1]["parts"][0],
                        generation_config=generation_config,
                        safety_settings=safety_settings
                    )
                    logger.info(f"GEMINI LLM SAFETY DEBUG: Response finish_reason = {getattr(response, 'finish_reason', 'N/A')}")
                    if hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]  
                        logger.info(f"GEMINI LLM SAFETY DEBUG: Candidate finish_reason = {getattr(candidate, 'finish_reason', 'N/A')}")
                        if hasattr(candidate, 'safety_ratings'):
                            logger.info(f"GEMINI LLM SAFETY DEBUG: Safety ratings = {candidate.safety_ratings}")
                            logger.info(f"GEMINI LLM SAFETY DEBUG: Safety ratings type = {type(candidate.safety_ratings)}")
                            logger.info(f"GEMINI LLM SAFETY DEBUG: Safety ratings length = {len(candidate.safety_ratings) if candidate.safety_ratings else 'None'}")
                            if candidate.safety_ratings:
                                for i, rating in enumerate(candidate.safety_ratings):
                                    logger.info(f"GEMINI LLM SAFETY DEBUG: Rating {i}: {rating}")
                        else:
                            logger.info(f"GEMINI LLM SAFETY DEBUG: No safety_ratings attribute found")

                
                # Extract response (handle safety filters)
                try:
                    response_text = response.text if hasattr(response, 'text') else ""
                except ValueError as e:
                    if "finish_reason" in str(e):
                        logger.warning(f"Gemini safety filter triggered: {e}")
                        response_text = "Content filtered by safety settings"
                    else:
                        raise e
                
                # Simple response format
                result = {
                    "content": response_text,
                    "model": model_name,
                    "response_time": time.time() - start_time
                }
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Simple retry logic - don't retry auth errors
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['api_key', 'invalid', 'unauthorized', 'forbidden']):
                    logger.error(f"Non-retryable Gemini error: {e}")
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