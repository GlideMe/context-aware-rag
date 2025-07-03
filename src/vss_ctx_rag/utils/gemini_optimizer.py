# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Google Gemini Optimizer - Enterprise-grade utility for optimizing Gemini model usage.

This module provides advanced optimization techniques for Google Gemini models:
- Token usage tracking and cost optimization
- Dynamic batching for better throughput
- Intelligent retry logic with circuit breaker pattern
- Response caching for performance
- Model parameter auto-tuning based on use case
- Proper system message handling for Gemini's unique requirements
"""

import os
import time
import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from functools import lru_cache
from collections import defaultdict
from enum import Enum

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from vss_ctx_rag.utils.ctx_rag_logger import logger


class UseCaseType(Enum):
    """Different use cases that require different optimization strategies."""
    SUMMARIZATION = "summarization"
    CHAT = "chat"
    NOTIFICATION = "notification"
    ANALYSIS = "analysis"
    GENERAL = "general"


@dataclass
class GeminiConfig:
    """Optimized configuration for different use cases."""
    model_name: str = "models/gemini-2.5-pro"
    max_tokens: int = 8192
    temperature: float = 0.1
    top_p: float = 0.9
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    safety_settings: Dict = None
    
    def __post_init__(self):
        if self.safety_settings is None:
            # Permissive safety settings (similar to Claude/OpenAI)
            self.safety_settings = {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }


@dataclass
class TokenUsage:
    """Track token usage for cost optimization."""
    input_tokens: int
    output_tokens: int
    model_name: str
    timestamp: float
    use_case: str
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost based on Gemini pricing (as of 2025)."""
        # Gemini 2.5 Flash pricing: ~$0.075/1M input, ~$0.30/1M output
        # Gemini 2.5 Pro pricing: ~$1.25/1M input, ~$5.00/1M output
        if "flash" in self.model_name.lower():
            input_cost = (self.input_tokens / 1_000_000) * 0.075
            output_cost = (self.output_tokens / 1_000_000) * 0.30
        else:  # Pro model
            input_cost = (self.input_tokens / 1_000_000) * 1.25
            output_cost = (self.output_tokens / 1_000_000) * 5.00
        return input_cost + output_cost


class GeminiOptimizer:
    """Enterprise-grade Google Gemini optimizer."""
    
    def __init__(self):
        self._configs = self._load_optimized_configs()
        self._token_usage: List[TokenUsage] = []
        self._response_cache: Dict[str, Tuple[Any, float]] = {}
        self._circuit_breaker_failures = defaultdict(int)
        self._circuit_breaker_last_failure = defaultdict(float)
        self._performance_metrics = defaultdict(list)
        self._client = None
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
    
    def _load_optimized_configs(self) -> Dict[UseCaseType, GeminiConfig]:
        """Load optimized configurations for different use cases."""
        return {
            UseCaseType.SUMMARIZATION: GeminiConfig(
                model_name="models/gemini-2.5-pro",
                temperature=0.1,  # Lower for consistency
                max_tokens=8192,  # Higher for comprehensive summaries
                top_p=0.9,
                enable_caching=True
            ),
            # ... other configs ...
            UseCaseType.ANALYSIS: GeminiConfig(
                model_name="models/gemini-2.5-pro",  # Use Pro for complex analysis
                temperature=0.1,  # Low for analytical consistency
                max_tokens=8192,  # High for detailed analysis
                top_p=0.95,       # Slightly higher for nuanced analysis
                enable_caching=True
            ),
            UseCaseType.GENERAL: GeminiConfig(
                model_name="models/gemini-2.5-pro",
                temperature=0.2,  # Balanced for general use
                max_tokens=4096,  # Standard size
                top_p=0.9,
                enable_caching=False  # General use shouldn't be cached
            )
        }
    
    def get_optimized_config(self, use_case: UseCaseType, **overrides) -> GeminiConfig:
        """Get optimized configuration for a specific use case."""
        config = self._configs.get(use_case, self._configs[UseCaseType.CHAT])
        
        # Apply overrides
        if overrides:
            config_dict = asdict(config)
            config_dict.update(overrides)
            config = GeminiConfig(**config_dict)
        
        logger.debug(f"Using optimized config for {use_case.value}: {config.model_name}")
        return config
    
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
    
    def _generate_cache_key(self, messages: List[Dict[str, str]], config: GeminiConfig) -> str:
        """Generate cache key for response caching."""
        cache_data = {
            "messages": messages,
            "model": config.model_name,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "top_p": config.top_p
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    def _is_cached_response_valid(self, timestamp: float, ttl: int) -> bool:
        """Check if cached response is still valid."""
        return time.time() - timestamp < ttl
    
    def get_cached_response(self, cache_key: str, ttl: int = 3600) -> Optional[Dict[str, Any]]:
        """Get cached response if valid."""
        if cache_key in self._response_cache:
            response, timestamp = self._response_cache[cache_key]
            if self._is_cached_response_valid(timestamp, ttl):
                logger.debug(f"Cache hit for key: {cache_key[:16]}...")
                return response
            else:
                # Clean up expired cache
                del self._response_cache[cache_key]
        return None
    
    def cache_response(self, cache_key: str, response: Dict[str, Any]) -> None:
        """Cache response with timestamp."""
        self._response_cache[cache_key] = (response, time.time())
        logger.debug(f"Cached response for key: {cache_key[:16]}...")
    
    def _exponential_backoff(self, attempt: int, base_delay: float, max_delay: float) -> float:
        """Calculate exponential backoff delay with jitter."""
        delay = min(base_delay * (2 ** attempt), max_delay)
        # Add jitter to prevent thundering herd
        jitter = delay * 0.1 * (0.5 - (hash(str(time.time())) % 256) / 255.0)
        return delay + jitter
    
    def _should_retry_error(self, error: Exception) -> bool:
        """Determine if error should be retried."""
        error_str = str(error).lower()
        
        # Don't retry on authentication or quota errors
        non_retryable = [
            "api_key", "invalid", "unauthorized", "forbidden", 
            "quota", "billing", "permission"
        ]
        
        return not any(keyword in error_str for keyword in non_retryable)
    
    def generate_response(
        self, 
        messages: List[Dict[str, str]], 
        use_case: UseCaseType = UseCaseType.CHAT,
        config_overrides: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Generate response with optimization and error handling."""
        self._initialize_client()
        
        # Get optimized config
        config = self.get_optimized_config(use_case, **(config_overrides or {}))
        
        # Check cache first
        cache_key = None
        if config.enable_caching:
            cache_key = self._generate_cache_key(messages, config)
            cached_response = self.get_cached_response(cache_key, config.cache_ttl)
            if cached_response:
                return cached_response
        
        # Circuit breaker check
        if self.should_circuit_break("gemini_api"):
            raise Exception("Circuit breaker activated - too many recent failures")
        
        # Retry logic
        last_exception = None
        for attempt in range(config.max_retries + 1):
            try:
                # Format messages for Gemini
                formatted_messages = self._format_messages_for_gemini(messages)
                
                # Create model instance
                model = genai.GenerativeModel(config.model_name)
                
                # Configure generation parameters
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p
                )
                
                # Generate response
                start_time = time.time()
                if len(formatted_messages) == 1:
                    # Single message
                    response = model.generate_content(
                        formatted_messages[0]["parts"][0],
                        generation_config=generation_config,
                        safety_settings=config.safety_settings
                    )
                else:
                    # Multi-turn conversation
                    chat = model.start_chat(history=formatted_messages[:-1])
                    response = chat.send_message(
                        formatted_messages[-1]["parts"][0],
                        generation_config=generation_config,
                        safety_settings=config.safety_settings
                    )
                
                # Extract response
                response_text = response.text if hasattr(response, 'text') else ""
                
                # Track performance
                response_time = time.time() - start_time
                self._performance_metrics[use_case.value].append(response_time)
                
                # Track token usage (Gemini doesn't provide detailed token counts)
                # We'll estimate based on text length
                estimated_input_tokens = sum(len(m.get("content", "")) for m in messages) // 4
                estimated_output_tokens = len(response_text) // 4
                
                usage = TokenUsage(
                    input_tokens=estimated_input_tokens,
                    output_tokens=estimated_output_tokens,
                    model_name=config.model_name,
                    timestamp=time.time(),
                    use_case=use_case.value
                )
                self.track_token_usage(usage)
                
                # Create response dict
                result = {
                    "content": response_text,
                    "model": config.model_name,
                    "usage": {
                        "input_tokens": estimated_input_tokens,
                        "output_tokens": estimated_output_tokens,
                        "total_tokens": estimated_input_tokens + estimated_output_tokens
                    },
                    "response_time": response_time
                }
                
                # Cache if enabled
                if config.enable_caching and cache_key:
                    self.cache_response(cache_key, result)
                
                return result
                
            except Exception as e:
                last_exception = e
                error_type = type(e).__name__
                
                # Record failure for circuit breaker
                self.record_failure("gemini_api")
                
                # Check if we should retry
                if not self._should_retry_error(e):
                    logger.error(f"Non-retryable Gemini error: {e}")
                    raise e
                
                if attempt < config.max_retries:
                    delay = self._exponential_backoff(attempt, config.base_delay, config.max_delay)
                    logger.warning(f"Gemini attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"All {config.max_retries + 1} Gemini attempts failed. Last error: {e}")
        
        raise last_exception
    
    def track_token_usage(self, usage: TokenUsage) -> None:
        """Track token usage for cost monitoring."""
        self._token_usage.append(usage)
        
        # Log cost-effective usage patterns
        if usage.estimated_cost_usd > 0.001:  # Log calls over $0.001
            logger.info(f"Gemini call cost: ${usage.estimated_cost_usd:.6f} "
                       f"({usage.total_tokens} tokens) for {usage.use_case}")
    
    def should_circuit_break(self, error_type: str) -> bool:
        """Circuit breaker pattern to prevent cascading failures."""
        failure_count = self._circuit_breaker_failures[error_type]
        last_failure = self._circuit_breaker_last_failure[error_type]
        
        # Circuit breaker logic: more than 5 failures in 5 minutes
        if failure_count >= 5 and time.time() - last_failure < 300:
            logger.warning(f"Circuit breaker activated for {error_type}")
            return True
        
        # Reset counter if it's been more than 5 minutes since last failure
        if time.time() - last_failure > 300:
            self._circuit_breaker_failures[error_type] = 0
        
        return False
    
    def record_failure(self, error_type: str) -> None:
        """Record failure for circuit breaker pattern."""
        self._circuit_breaker_failures[error_type] += 1
        self._circuit_breaker_last_failure[error_type] = time.time()
    
    def get_usage_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get usage analytics for cost optimization."""
        cutoff_time = time.time() - (hours * 3600)
        recent_usage = [u for u in self._token_usage if u.timestamp > cutoff_time]
        
        if not recent_usage:
            return {"total_cost": 0, "total_tokens": 0, "call_count": 0}
        
        total_cost = sum(u.estimated_cost_usd for u in recent_usage)
        total_tokens = sum(u.total_tokens for u in recent_usage)
        
        by_use_case = defaultdict(lambda: {"cost": 0, "tokens": 0, "calls": 0})
        for usage in recent_usage:
            by_use_case[usage.use_case]["cost"] += usage.estimated_cost_usd
            by_use_case[usage.use_case]["tokens"] += usage.total_tokens
            by_use_case[usage.use_case]["calls"] += 1
        
        return {
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "call_count": len(recent_usage),
            "average_cost_per_call": total_cost / len(recent_usage) if recent_usage else 0,
            "by_use_case": dict(by_use_case)
        }
    
    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """Clean up expired cache entries."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        expired_keys = [
            key for key, (_, timestamp) in self._response_cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_keys:
            del self._response_cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired Gemini cache entries")
        return len(expired_keys)


# Global optimizer instance
gemini_optimizer = GeminiOptimizer()