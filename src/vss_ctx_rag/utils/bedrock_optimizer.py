# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AWS Bedrock Basic Integration - Simple retry logic matching Llama/OpenAI approach.

This module provides basic Claude model integration with simple retry logic,
removing complex optimizations to match the approach used by Llama/OpenAI models.
"""

import os
import time
from typing import Dict, List, Optional, Any
from enum import Enum

from vss_ctx_rag.utils.ctx_rag_logger import logger


class UseCaseType(Enum):
    """Simple use case types."""
    SUMMARIZATION = "summarization"
    CHAT = "chat"
    NOTIFICATION = "notification"
    ANALYSIS = "analysis"


class BedrockOptimizer:
    """Basic AWS Bedrock optimizer for Claude models with simple retry logic."""
    
    def __init__(self):
        pass
    
    def get_basic_config(self, use_case: UseCaseType = UseCaseType.CHAT) -> Dict[str, Any]:
        """Get basic configuration without complex optimization."""
        # Simple, consistent configuration regardless of use case
        return {
            "model_id": "us.anthropic.claude-sonnet-4-20250514-v1:0",
            "region": "us-east-1",
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_retries": 3,
            "base_delay": 1.0
        }


# Global optimizer instance
bedrock_optimizer = BedrockOptimizer()