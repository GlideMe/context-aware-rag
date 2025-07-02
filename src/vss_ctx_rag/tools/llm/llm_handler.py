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
import json
import boto3
from typing import Optional, Iterator, Dict, Any, List
from botocore.exceptions import BotoCoreError, ClientError

try:
    from vss_ctx_rag.utils.gemini_optimizer import gemini_optimizer, UseCaseType
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables.utils import ConfigurableField
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from vss_ctx_rag.base import Tool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.utils.globals import DEFAULT_LLM_BASE_URL
from vss_ctx_rag.utils.utils import (
    is_openai_model,
    is_claude_model,
    is_gemini_model,
)
from langchain_core.runnables.base import Runnable
from langchain_nvidia_ai_endpoints import register_model, Model, ChatNVIDIA
from typing import Optional, Iterator, Dict, Any, List

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


class ClaudeBedrockLLM(BaseChatModel):
    """Direct boto3 implementation for Claude on AWS Bedrock"""
    
    # Pydantic field declarations
    model_id: str
    region_name: str = "us-east-1"
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    bedrock_client: Any = None
    
    def __init__(self, model_id: str, region_name: str = "us-east-1", **kwargs):
        # Pass fields to parent class
        super().__init__(
            model_id=model_id,
            region_name=region_name,
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            bedrock_client=None,
            **kwargs
        )
        
        # Initialize boto3 client
        try:
            self.bedrock_client = boto3.client(
                service_name='bedrock-runtime',
                region_name=region_name
            )
            logger.info(f"Initialized Bedrock client for region: {region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    def _format_messages_for_claude(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Convert LangChain messages to Claude Bedrock format"""
        formatted_messages = []
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": formatted_messages
        }
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({
                    "role": "user",
                    "content": msg.content
                })
            elif isinstance(msg, AIMessage):
                formatted_messages.append({
                    "role": "assistant", 
                    "content": msg.content
                })
            elif isinstance(msg, SystemMessage):
                # SystemMessage - Claude handles these separately in the "system" field
                body["system"] = str(msg.content)
        
        return body
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> ChatResult:
        """Generate a single response using Bedrock"""
        try:
            # Format messages for Claude
            body = self._format_messages_for_claude(messages)
            #This log is the entire body of message ssent to claude. it is very large 
            #so only use when needed.
            #logger.info(f"Claude request body: {json.dumps(body, indent=2)}")
            # Call Bedrock
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            content_list = response_body.get('content', [])
            if content_list:  # Check if list has elements
                content = content_list[0].get('text', '')
            else:
                content = ''  # Handle empty list safely
            
            # Create ChatGeneration
            generation = ChatGeneration(message=AIMessage(content=content))
            
            return ChatResult(generations=[generation])
            
        except (BotoCoreError, ClientError) as e:
            logger.error(f"AWS Bedrock error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error calling Claude: {e}")
            raise
    
    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> Iterator[ChatGeneration]:
        """Stream response from Bedrock (simplified - yields single response)"""
        # For now, just return the full response
        # Full streaming would require invoke_model_with_response_stream
        result = self._generate(messages, stop, **kwargs)
        yield result.generations[0]
    
    @property
    def _llm_type(self) -> str:
        return "claude-bedrock-boto3"


class ChatClaudeTool(LLMTool):
    def __init__(self, model=None, api_key=None, **llm_params) -> None:
        # Validate AWS credentials for Bedrock access
        self._validate_aws_credentials()
        
        # Set optimal defaults for Claude Sonnet performance and cost balance
        region_name = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
        model_id = model or "anthropic.claude-3-5-sonnet-20241022-v2:0"
        
        # Extract parameters for our custom LLM
        max_tokens = llm_params.get("max_tokens", 4096)
        temperature = llm_params.get("temperature", 0.1)
        top_p = llm_params.get("top_p", 0.9)
        
        logger.info(f"Initializing Claude Bedrock client for model: {model_id}")
        
        # Create our custom boto3-based LLM
        claude_llm = ClaudeBedrockLLM(
            model_id=model_id,
            region_name=region_name
        )
        
        # Set initial parameters
        claude_llm.max_tokens = max_tokens
        claude_llm.temperature = temperature
        claude_llm.top_p = top_p
        
        super().__init__(
            llm=claude_llm,
            name="claude_bedrock_tool"
        )
        
        # Enhanced warmup with retry logic
        try:
            if os.getenv("CA_RAG_ENABLE_WARMUP", "false").lower() == "true":
                self.warmup(model_id)
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
        
        # Validate and optimize parameters for Claude Sonnet
        if top_p is not None:
            top_p = max(0.0, min(1.0, float(top_p)))  # Clamp to valid range
            self.llm.top_p = top_p
            
        if temperature is not None:
            temperature = max(0.0, min(1.0, float(temperature)))  # Clamp to valid range
            self.llm.temperature = temperature
            
        if max_tokens is not None:
            # Optimize for Claude Sonnet's capabilities
            max_tokens = max(1, min(4096, int(max_tokens)))  # Clamp to model limits
            self.llm.max_tokens = max_tokens
        
        logger.debug(f"Updated Claude LLM configuration: top_p={self.llm.top_p}, temperature={self.llm.temperature}, max_tokens={self.llm.max_tokens}")


class GeminiLLM(BaseChatModel):
    """Direct Google API implementation for Gemini models"""
    
    # Pydantic field declarations
    model_name: str
    api_key: str
    max_tokens: int = 4096
    temperature: float = 0.1
    top_p: float = 0.9
    
    def __init__(self, model_name: str, api_key: str, **kwargs):
        # Pass fields to parent class
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            max_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            **kwargs
        )
        
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Gemini optimizer is not available. "
                "Check gemini_optimizer.py import"
            )
        
        logger.info(f"Initialized Gemini LLM for model: {model_name}")
    
    def _format_messages_for_gemini(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        """Convert LangChain messages to Gemini format with proper system message handling"""
        formatted_messages = []
        system_message = None
        
        # First pass: extract system message
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_message = str(msg.content)
                break
        
        # Second pass: format user/assistant messages
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = str(msg.content)
                # If we have a system message and this is the first user message, prepend it
                if system_message and len(formatted_messages) == 0:
                    content = f"{system_message}\n\n{content}"
                    system_message = None  # Clear it so we don't add it again
                
                formatted_messages.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif isinstance(msg, AIMessage):
                formatted_messages.append({
                    "role": "model",
                    "parts": [{"text": str(msg.content)}]
                })
        
        return {
            "contents": formatted_messages,
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature": self.temperature,
                "topP": self.top_p,
            }
        }
    
    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> ChatResult:
        """Generate a single response using Gemini API"""
        try:
            # Format messages for Gemini
            request_data = self._format_messages_for_gemini(messages)
            
            # Call Gemini through optimizer
            response = gemini_optimizer.generate_content(
                model_name=self.model_name,
                api_key=self.api_key,
                request_data=request_data,
                use_case=UseCaseType.GENERAL
            )
            
            # Extract content from response
            content = ""
            if response and "candidates" in response:
                candidates = response["candidates"]
                if candidates and len(candidates) > 0:
                    candidate = candidates[0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        parts = candidate["content"]["parts"]
                        if parts and len(parts) > 0:
                            content = parts[0].get("text", "")
            
            # Create ChatGeneration
            generation = ChatGeneration(message=AIMessage(content=content))
            
            return ChatResult(generations=[generation])
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def _stream(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> Iterator[ChatGeneration]:
        """Stream response from Gemini (simplified - yields single response)"""
        # For now, just return the full response
        # Full streaming would require separate streaming implementation
        result = self._generate(messages, stop, **kwargs)
        yield result.generations[0]
    
    @property
    def _llm_type(self) -> str:
        return "gemini-direct-api"


class ChatGeminiTool(LLMTool):
    def __init__(self, model=None, api_key=None, **llm_params) -> None:
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "Gemini optimizer is not available. "
                "Check gemini_optimizer.py import"
            )
        
        # Set default model if not provided - Flash is faster for testing, Pro for production
        model_name = model or "gemini-2.5-pro-exp"  # or "gemini-2.5-flash-exp"
        
        # Configure API key
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Extract parameters for our custom LLM
        max_tokens = llm_params.get("max_tokens", 4096)
        temperature = llm_params.get("temperature", 0.1)
        top_p = llm_params.get("top_p", 0.9)
        
        logger.info(f"Initializing Gemini client for model: {model_name}")
        
        # Create our custom Gemini LLM
        gemini_llm = GeminiLLM(
            model_name=model_name,
            api_key=api_key
        )
        
        # Set initial parameters
        gemini_llm.max_tokens = max_tokens
        gemini_llm.temperature = temperature
        gemini_llm.top_p = top_p
        
        super().__init__(
            llm=gemini_llm,
            name="gemini_tool"
        )
        
        # Enhanced warmup with retry logic
        try:
            if os.getenv("CA_RAG_ENABLE_WARMUP", "false").lower() == "true":
                self.warmup(model_name)
        except Exception as e:
            logger.error(f"Error warming up Gemini LLM: {e}")
            # Don't raise - allow initialization to continue for graceful degradation
            logger.warning("Continuing without warmup - model will initialize on first use")
    
    def warmup(self, model_name):
        """Warm up the Gemini model with retry logic for reliability."""
        max_warmup_attempts = 3
        
        for attempt in range(max_warmup_attempts):
            try:
                logger.info(f"Warming up Gemini LLM {model_name} (attempt {attempt + 1}/{max_warmup_attempts})")
                
                # Use a simple, cost-effective warmup prompt
                warmup_response = self.invoke("Hello")
                
                logger.info(f"Gemini warmup successful: {str(warmup_response)[:100]}...")
                return
                
            except Exception as e:
                if attempt < max_warmup_attempts - 1:
                    logger.warning(f"Warmup attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All warmup attempts failed for {model_name}: {e}")
                    raise

    def update(self, top_p=None, temperature=None, max_tokens=None):
        """Update Gemini model configuration with validation and optimization."""
        
        # Validate and optimize parameters for Gemini 2.5
        if top_p is not None:
            top_p = max(0.0, min(1.0, float(top_p)))  # Clamp to valid range
            self.llm.top_p = top_p
            
        if temperature is not None:
            temperature = max(0.0, min(1.0, float(temperature)))  # Clamp to valid range
            self.llm.temperature = temperature
            
        if max_tokens is not None:
            # Optimize for Gemini 2.5's capabilities
            max_tokens = max(1, min(32768, int(max_tokens)))  # Clamp to model limits
            self.llm.max_tokens = max_tokens
        
        logger.debug(f"Updated Gemini LLM configuration: top_p={self.llm.top_p}, temperature={self.llm.temperature}, max_tokens={self.llm.max_tokens}")