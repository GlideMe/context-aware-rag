# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables.utils import ConfigurableField
from vss_ctx_rag.base import Tool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.utils.globals import DEFAULT_LLM_BASE_URL
from vss_ctx_rag.utils.utils import (
    is_openai_model,
    is_claude_model,
)
from langchain_aws.chat_models import ChatBedrock
from langchain_core.runnables.base import Runnable
from langchain_nvidia_ai_endpoints import register_model, Model, ChatNVIDIA


class LLMTool(Tool, Runnable):
    """A Tool class wrapper for LLMs.

    Returns:
        LLMTool: A Tool that wraps an LLM.
    """

    llm: BaseChatModel

    def __init__(self, llm, name="llm_tool") -> None:
        Tool.__init__(self, name)
        self.llm = llm

    def __getattr__(self, attr):
        return getattr(self.llm, attr)

    def invoke(self, *args, **kwargs):
        return self.llm.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self.llm.stream(*args, **kwargs)

    def batch(self, *args, **kwargs):
        return self.llm.batch(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.llm.ainvoke(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        return await self.llm.astream(*args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await self.llm.abatch(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.name


class ChatOpenAITool(LLMTool):
    def __init__(
        self, model=None, api_key=None, base_url=DEFAULT_LLM_BASE_URL, **llm_params
    ) -> None:
        if model and is_openai_model(model):
            base_url = ""
            super().__init__(
                llm=ChatOpenAI(
                    model=model, api_key=api_key, base_url=base_url, **llm_params
                ).configurable_fields(
                    top_p=ConfigurableField(id="top_p"),
                    temperature=ConfigurableField(id="temperature"),
                    max_tokens=ConfigurableField(id="max_tokens"),
                )
            )
        elif model and "llama-3.1-70b-instruct" in model and "nvcf" in base_url:
            register_model(
                Model(
                    id=model, model_type="chat", client="ChatNVIDIA", endpoint=base_url
                )
            )
            super().__init__(
                ChatNVIDIA(
                    model=model, api_key=api_key, **llm_params
                ).configurable_fields(
                    top_p=ConfigurableField(id="top_p"),
                    temperature=ConfigurableField(id="temperature"),
                    max_tokens=ConfigurableField(id="max_tokens"),
                )
            )
        else:
            super().__init__(
                llm=ChatOpenAI(
                    model=model, api_key=api_key, base_url=base_url, **llm_params
                ).configurable_fields(
                    top_p=ConfigurableField(id="top_p"),
                    temperature=ConfigurableField(id="temperature"),
                    max_tokens=ConfigurableField(id="max_tokens"),
                )
            )
        try:
            if os.getenv("CA_RAG_ENABLE_WARMUP", "false").lower() == "true":
                self.warmup(model)
        except Exception as e:
            logger.error(f"Error warming up LLM: {e}")
            raise

    def warmup(self, model_name):
        try:
            logger.info(f"Warming up LLM {model_name}")
            logger.info(str(self.invoke("Hello, world!")))
        except Exception as e:
            logger.error(f"Error warming up LLM {model_name}: {e}")
            raise

    def update(self, top_p=None, temperature=None, max_tokens=None):
        configurable_dict = {}
        if top_p is not None:
            configurable_dict["top_p"] = top_p
        if temperature is not None:
            configurable_dict["temperature"] = temperature
        if max_tokens is not None:
            configurable_dict["max_tokens"] = max_tokens
        logger.debug(f"Updating LLM with config:{configurable_dict}")
        self.llm = self.llm.with_config(configurable=configurable_dict)


class ChatClaudeTool(LLMTool):
    def __init__(self, model=None, api_key=None, **llm_params) -> None:
        # Validate AWS credentials for Bedrock access
        self._validate_aws_credentials()
        
        # Set optimal defaults for Claude Sonnet performance and cost balance
        bedrock_params = {
            "region_name": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            "model_kwargs": {
                "max_tokens": llm_params.get("max_tokens", 4096),
                "temperature": llm_params.get("temperature", 0.1),  # Lower for consistency
                "top_p": llm_params.get("top_p", 0.9),
                "anthropic_version": "bedrock-2023-05-31"
            },
            "streaming": llm_params.get("streaming", False),
            "verbose": llm_params.get("verbose", False)
        }
        
        # Remove processed params to avoid conflicts
        for param in ["max_tokens", "temperature", "top_p", "streaming", "verbose"]:
            llm_params.pop(param, None)
            
        bedrock_params.update(llm_params)
        
        logger.info(f"Initializing Claude Bedrock client for model: {model}")
        
        super().__init__(
            llm=ChatBedrock(
                model_id=model or "us.anthropic.claude-sonnet-4-20250514-v1:0",
                **bedrock_params
            ).configurable_fields(
                top_p=ConfigurableField(id="top_p"),
                temperature=ConfigurableField(id="temperature"),
                max_tokens=ConfigurableField(id="max_tokens"),
            )
        )
        
        # Enhanced warmup with retry logic
        try:
            if os.getenv("CA_RAG_ENABLE_WARMUP", "false").lower() == "true":
                self.warmup(model or "us.anthropic.claude-sonnet-4-20250514-v1:0")
        except Exception as e:
            logger.error(f"Error warming up Claude LLM: {e}")
            # Don't raise - allow initialization to continue for graceful degradation
            logger.warning("Continuing without warmup - model will initialize on first use")
    
    def _validate_aws_credentials(self) -> None:
        """Validate required AWS credentials for Bedrock access."""
        required_vars = ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required AWS credentials for Bedrock: {', '.join(missing_vars)}")
            
        logger.debug("AWS credentials validated for Bedrock access")

    def warmup(self, model_name):
        """Warm up the Claude model with retry logic for reliability."""
        max_warmup_attempts = 3
        
        for attempt in range(max_warmup_attempts):
            try:
                logger.info(f"Warming up Claude LLM {model_name} (attempt {attempt + 1}/{max_warmup_attempts})")
                
                # Use a simple, cost-effective warmup prompt
                warmup_response = self.invoke("Hello")
                
                logger.info(f"Claude warmup successful: {str(warmup_response)[:100]}...")
                return
                
            except Exception as e:
                if attempt < max_warmup_attempts - 1:
                    logger.warning(f"Warmup attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All warmup attempts failed for {model_name}: {e}")
                    raise

    def update(self, top_p=None, temperature=None, max_tokens=None):
        """Update Claude model configuration with validation and optimization."""
        configurable_dict = {}
        
        # Validate and optimize parameters for Claude Sonnet
        if top_p is not None:
            top_p = max(0.0, min(1.0, float(top_p)))  # Clamp to valid range
            configurable_dict["top_p"] = top_p
            
        if temperature is not None:
            temperature = max(0.0, min(1.0, float(temperature)))  # Clamp to valid range
            configurable_dict["temperature"] = temperature
            
        if max_tokens is not None:
            # Optimize for Claude Sonnet's capabilities
            max_tokens = max(1, min(4096, int(max_tokens)))  # Clamp to model limits
            configurable_dict["max_tokens"] = max_tokens
        
        if configurable_dict:
            logger.debug(f"Updating Claude LLM configuration: {configurable_dict}")
            self.llm = self.llm.with_config(configurable=configurable_dict)
        else:
            logger.debug("No valid configuration updates provided")
