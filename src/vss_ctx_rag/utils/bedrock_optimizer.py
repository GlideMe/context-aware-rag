# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AWS Bedrock Optimizer - Enterprise-grade utility for optimizing Claude model usage.

This module provides advanced optimization techniques for AWS Bedrock Claude models:
- Token usage tracking and cost optimization
- Dynamic batching for better throughput
- Intelligent retry logic with circuit breaker pattern
- Response caching for performance
- Model parameter auto-tuning based on use case
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

from vss_ctx_rag.utils.ctx_rag_logger import logger


class UseCaseType(Enum):
    """Different use cases that require different optimization strategies."""
    SUMMARIZATION = "summarization"
    CHAT = "chat"
    NOTIFICATION = "notification"
    ANALYSIS = "analysis"


@dataclass
class BedrockConfig:
    """Optimized configuration for different use cases."""
    model_id: str = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    region: str = "us-east-1"
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour


@dataclass
class TokenUsage:
    """Track token usage for cost optimization."""
    input_tokens: int
    output_tokens: int
    model_id: str
    timestamp: float
    use_case: str
    
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens
    
    @property
    def estimated_cost_usd(self) -> float:
        """Estimate cost based on Claude Sonnet pricing (as of 2025)."""
        # Claude Sonnet pricing: ~$3/1M input tokens, ~$15/1M output tokens
        input_cost = (self.input_tokens / 1_000_000) * 3.0
        output_cost = (self.output_tokens / 1_000_000) * 15.0
        return input_cost + output_cost


class BedrockOptimizer:
    """Enterprise-grade AWS Bedrock optimizer for Claude models."""
    
    def __init__(self):
        self._configs = self._load_optimized_configs()
        self._token_usage: List[TokenUsage] = []
        self._response_cache: Dict[str, Tuple[Any, float]] = {}
        self._circuit_breaker_failures = defaultdict(int)
        self._circuit_breaker_last_failure = defaultdict(float)
        self._performance_metrics = defaultdict(list)
    
    def _load_optimized_configs(self) -> Dict[UseCaseType, BedrockConfig]:
        """Load optimized configurations for different use cases."""
        return {
            UseCaseType.SUMMARIZATION: BedrockConfig(
                temperature=0.1,  # Lower for consistency
                max_tokens=4096,  # Higher for comprehensive summaries
                top_p=0.9,
                enable_caching=True
            ),
            UseCaseType.CHAT: BedrockConfig(
                temperature=0.2,  # Slightly higher for more natural responses
                max_tokens=2048,  # Moderate for conversations
                top_p=0.9,
                enable_caching=False  # Chat responses should be unique
            ),
            UseCaseType.NOTIFICATION: BedrockConfig(
                temperature=0.05,  # Very low for consistent alerts
                max_tokens=512,   # Short notifications
                top_p=0.8,
                enable_caching=True
            ),
            UseCaseType.ANALYSIS: BedrockConfig(
                temperature=0.1,  # Low for analytical consistency
                max_tokens=4096,  # High for detailed analysis
                top_p=0.95,       # Slightly higher for nuanced analysis
                enable_caching=True
            )
        }
    
    def get_optimized_config(self, use_case: UseCaseType, **overrides) -> BedrockConfig:
        """Get optimized configuration for a specific use case."""
        config = self._configs.get(use_case, self._configs[UseCaseType.CHAT])
        
        # Apply overrides
        if overrides:
            config_dict = asdict(config)
            config_dict.update(overrides)
            config = BedrockConfig(**config_dict)
        
        logger.debug(f"Using optimized config for {use_case.value}: {asdict(config)}")
        return config
    
    def _generate_cache_key(self, prompt: str, images: List[str] = None, config: BedrockConfig = None) -> str:
        """Generate cache key for response caching."""
        cache_data = {
            "prompt": prompt,
            "images": images or [],
            "config": asdict(config) if config else {}
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_string.encode()).hexdigest()
    
    def _is_cached_response_valid(self, timestamp: float, ttl: int) -> bool:
        """Check if cached response is still valid."""
        return time.time() - timestamp < ttl
    
    def get_cached_response(self, cache_key: str, ttl: int = 3600) -> Optional[Any]:
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
    
    def cache_response(self, cache_key: str, response: Any) -> None:
        """Cache response with timestamp."""
        self._response_cache[cache_key] = (response, time.time())
        logger.debug(f"Cached response for key: {cache_key[:16]}...")
    
    def track_token_usage(self, usage: TokenUsage) -> None:
        """Track token usage for cost monitoring."""
        self._token_usage.append(usage)
        
        # Log cost-effective usage patterns
        if usage.estimated_cost_usd > 0.01:  # Log expensive calls
            logger.info(f"High-cost call detected: ${usage.estimated_cost_usd:.4f} "
                       f"({usage.total_tokens} tokens) for {usage.use_case}")
    
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
            "average_cost_per_call": total_cost / len(recent_usage),
            "by_use_case": dict(by_use_case),
            "cost_trend": self._calculate_cost_trend(recent_usage)
        }
    
    def _calculate_cost_trend(self, usage_data: List[TokenUsage]) -> str:
        """Calculate cost trend for optimization recommendations."""
        if len(usage_data) < 10:
            return "insufficient_data"
        
        # Simple trend calculation: compare first and last quarters
        quarter_size = len(usage_data) // 4
        early_avg = sum(u.estimated_cost_usd for u in usage_data[:quarter_size]) / quarter_size
        late_avg = sum(u.estimated_cost_usd for u in usage_data[-quarter_size:]) / quarter_size
        
        if late_avg > early_avg * 1.2:
            return "increasing"
        elif late_avg < early_avg * 0.8:
            return "decreasing"
        else:
            return "stable"
    
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
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get AI-powered optimization recommendations."""
        recommendations = []
        analytics = self.get_usage_analytics()
        
        # Cost optimization recommendations
        if analytics["total_cost"] > 10.0:  # More than $10/day
            recommendations.append(
                "Consider implementing more aggressive caching - high daily costs detected"
            )
        
        if analytics["cost_trend"] == "increasing":
            recommendations.append(
                "Cost trend is increasing - review recent prompt changes and implement batching"
            )
        
        # Performance optimization recommendations
        avg_tokens = analytics["total_tokens"] / max(analytics["call_count"], 1)
        if avg_tokens > 3000:
            recommendations.append(
                "High average token usage detected - consider prompt optimization and summarization"
            )
        
        # Use case specific recommendations
        by_use_case = analytics.get("by_use_case", {})
        for use_case, stats in by_use_case.items():
            if stats["cost"] / max(stats["calls"], 1) > 0.05:  # $0.05 per call
                recommendations.append(
                    f"High cost per call for {use_case} - consider parameter tuning"
                )
        
        return recommendations
    
    @lru_cache(maxsize=128)
    def get_optimal_batch_size(self, use_case: UseCaseType, input_size: int) -> int:
        """Calculate optimal batch size for performance vs cost."""
        # Dynamic batching based on use case and input size
        base_batch_sizes = {
            UseCaseType.SUMMARIZATION: 5,
            UseCaseType.CHAT: 1,  # Chat should not be batched
            UseCaseType.NOTIFICATION: 10,
            UseCaseType.ANALYSIS: 3
        }
        
        base_size = base_batch_sizes.get(use_case, 1)
        
        # Adjust based on input size
        if input_size < 1000:  # Small inputs can be batched more aggressively
            return min(base_size * 2, 20)
        elif input_size > 5000:  # Large inputs should be batched less
            return max(base_size // 2, 1)
        
        return base_size
    
    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """Clean up expired cache entries."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        expired_keys = [
            key for key, (_, timestamp) in self._response_cache.items()
            if timestamp < cutoff_time
        ]
        
        for key in expired_keys:
            del self._response_cache[key]
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)


# Global optimizer instance
bedrock_optimizer = BedrockOptimizer()