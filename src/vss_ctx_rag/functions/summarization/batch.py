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

"""summarization.py: File contains Function class"""

import asyncio
import os
from pathlib import Path
import time
from langchain_community.callbacks import get_openai_callback
from langchain_community.callbacks.manager import get_bedrock_anthropic_callback
from schema import Schema
import base64

from vss_ctx_rag.base import Function
from vss_ctx_rag.utils.utils import remove_think_tags, call_token_safe
from vss_ctx_rag.utils.common_utils import is_claude_model
from vss_ctx_rag.tools.storage import StorageTool
from vss_ctx_rag.tools.health.rag_health import SummaryMetrics
from vss_ctx_rag.utils.ctx_rag_logger import logger, TimeMeasure
from vss_ctx_rag.utils.ctx_rag_batcher import Batcher
from vss_ctx_rag.utils.globals import DEFAULT_SUMM_RECURSION_LIMIT, LLM_TOOL_NAME
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vss_ctx_rag.utils.common_utils import is_gemini_model, is_claude_model # Make sure is_gemini_model is imported
from vss_ctx_rag.base import Function



class BatchSummarization(Function):
    """Batch Summarization Function"""

    config: dict
    aggregation_prompt: str
    output_parser = StrOutputParser()
    batch_size: int
    curr_batch: str
    curr_batch_size: int
    curr_batch_i: int
    batch_pipeline: RunnableSequence
    aggregation_pipeline: RunnableSequence
    vector_db: StorageTool
    timeout: int = 120  # seconds
    call_schema: Schema = Schema(
        {"start_index": int, "end_index": int}, ignore_extra_keys=True
    )
    metrics = SummaryMetrics()
    semaphore: asyncio.Semaphore 

    def setup(self):
        def prepare_messages(inputs):
            system_prompt = self.get_param("prompts", "caption_summarization")
            content_blocks = []
            if self.endless_ai_enabled:
                # Add image blocks if any are present
                images = inputs.get("images", [])

                content_blocks.extend({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in images)

            # Add the user question after the images (if any)
            if inputs["input"] and inputs["input"].strip():
                content_blocks.append({"type": "text", "text": inputs["input"]})

            return [SystemMessage(content=system_prompt), HumanMessage(content=content_blocks)]

        self.aggregation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_param("prompts", "summary_aggregation")),
                ("user", "{input}"),
            ]
        )
        self.output_parser = StrOutputParser()
        self.batch_pipeline = (
            RunnableLambda(prepare_messages)
            | self.get_tool(LLM_TOOL_NAME)
            | self.output_parser
            | remove_think_tags
        )
        self.aggregation_pipeline = (
            self.aggregation_prompt
            | self.get_tool(LLM_TOOL_NAME)
            | self.output_parser
            | remove_think_tags
        )
        self.batch_size = self.get_param("params", "batch_size")
        self.vector_db = self.get_tool("vector_db")
        self.timeout = (
            self.get_param("timeout_sec", required=False)
            if self.get_param("timeout_sec", required=False)
            else self.timeout
        )

        # working params
        self.curr_batch_i = 0
        self.batcher = Batcher(self.batch_size)
        self.recursion_limit = (
            self.get_param("summ_rec_lim", required=False)
            if self.get_param("summ_rec_lim", required=False)
            else DEFAULT_SUMM_RECURSION_LIMIT
        )

        self.endless_ai_enabled = self.get_param("params", "endless_ai_enabled")
        logger.info(f"Batch endless_ai_enabled value: {self.endless_ai_enabled}")

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        self.summary_start_time = None
        self.enable_summary = True
        self.semaphore = asyncio.Semaphore(10)

    def _get_appropriate_callback(self):
        """Get the appropriate callback based on the LLM being used"""
        model_name = self.get_param("llm", "model")
        if is_claude_model(model_name):
            return get_bedrock_anthropic_callback()
        else:
            return get_openai_callback()

    async def _process_full_batch(self, batch):
        """Process a full batch immediately"""
        async with self.semaphore:
            with TimeMeasure(
                "Batch "
                + str(batch._batch_index)
                + " Summary IS LAST "
                + str(
                    any(
                        doc_meta.get("is_last", False) for _, _, doc_meta in batch.as_list()
                    )
                ),
                "pink",
            ):
                logger.info(
                    "=== BATCH PROCESSING START: Batch %d is full. Processing ...", batch._batch_index
                )
                
                # LOG: Batch content details
                batch_list = batch.as_list()
                logger.debug(f"DEBUG: Batch {batch._batch_index} contains {len(batch_list)} documents")
                for idx, (doc, doc_i, doc_meta) in enumerate(batch_list):
                    logger.info(f"DEBUG: Batch {batch._batch_index} Doc {idx}: length={len(doc)}, doc_i={doc_i}, meta={doc_meta}")
                
                batch_summary = "" # Ensure batch_summary is initialized
                try:
                    with self._get_appropriate_callback() as cb:
                        # --- CHANGE 1: Get the model name ---
                        model_name = self.get_param("llm", "model")
                        logger.info(f"DEBUG: Batch {batch._batch_index} using model: {model_name}")
                        
                        if self.endless_ai_enabled:
                            def image_file_to_base64(filepath):
                                with open(filepath, 'rb') as image_file:
                                    image_data = image_file.read()
                                return base64.b64encode(image_data).decode('utf-8')

                            unique_images = set()
                            for doc, doc_i, doc_meta in batch.as_list():
                                if doc_meta.get("grid_filenames"):
                                    unique_images.update(doc_meta["grid_filenames"].split('|'))
                            images = [image_file_to_base64(img) for img in list(unique_images)]
                            logger.info(f"DEBUG: Batch {batch._batch_index} has {len(images)} images")
                        else:
                            images = []

                        # LOG: Input being sent to call_token_safe
                        input_text = " ".join([doc for doc, _, _ in batch.as_list()])
                        logger.info(f"DEBUG: Batch {batch._batch_index} input text length: {len(input_text)}")
                        logger.info(f"DEBUG: Batch {batch._batch_index} input text preview: {input_text[:200]}...")

                        if len(batch.as_list()) > 1 or len(images) > 0:
                            logger.info(f"DEBUG: Batch {batch._batch_index} calling call_token_safe (multi-doc path)")
                            batch_summary = await call_token_safe(
                                {"input": input_text, "images": images}, 
                                self.batch_pipeline, 
                                self.recursion_limit,
                                model_name=model_name,  # <-- CHANGE 2: Pass model_name
                                batch_index=batch._batch_index
                            )
                        else:
                            doc, _, doc_meta = batch.as_list()[0]
                            if doc.strip() == "." and doc_meta.get("is_last", False):
                                logger.info(f"DEBUG: Batch {batch._batch_index} is final marker batch")
                                batch_summary = "Video Analysis completed."
                            else:
                                logger.info(f"DEBUG: Batch {batch._batch_index} calling call_token_safe (single-doc path)")
                                batch_summary = await call_token_safe(
                                    {"input": input_text, "images": images}, 
                                    self.batch_pipeline, 
                                    self.recursion_limit,
                                    model_name=model_name,
                                    batch_index=batch._batch_index
                                )

                        # LOG: Result from call_token_safe
                        logger.info(f"DEBUG: Batch {batch._batch_index} summary length: {len(batch_summary) if batch_summary else 0}")
                        logger.info(f"DEBUG: Batch {batch._batch_index} summary preview: {batch_summary[:200] if batch_summary else 'EMPTY'}...")

                except Exception as e:
                    logger.error(f"ERROR: Batch {batch._batch_index} exception during processing: {e}")
                    batch_summary = "."
                
                # Final safety check in case the retry logic returns an error or is empty
                if not batch_summary or not batch_summary.strip():
                    logger.warning(f"WARNING: Batch {batch._batch_index} returned empty summary, using fallback")
                    batch_summary = "."

                self.metrics.summary_tokens += cb.total_tokens
                self.metrics.summary_requests += cb.successful_requests
                logger.info("=== BATCH PROCESSING END: Batch %d summary: %s", batch._batch_index, batch_summary)
                logger.info(
                    "Total Tokens: %s, Prompt Tokens: %s, Completion Tokens: %s, Successful Requests: %s, Total Cost (USD): $%s"
                    % (
                        cb.total_tokens,
                        cb.prompt_tokens,
                        cb.completion_tokens,
                        cb.successful_requests,
                        cb.total_cost,
                    ),
                )
            
            try:
                batch_list = batch.as_list()
                last_doc_meta = batch_list[-1][2] if batch_list else {}
                batch_meta = {
                    **last_doc_meta,
                    "batch_i": batch._batch_index,
                    "doc_type": "caption_summary",
                }
                logger.debug(f"DEBUG: Batch {batch._batch_index} saving to vector_db with metadata: {batch_meta}")
                self.vector_db.add_summary(summary=batch_summary, metadata=batch_meta)
            
            except Exception as e:
                logger.error(f"ERROR: Batch {batch._batch_index} failed to save to vector_db: {e}")

    async def acall(self, state: dict):
        """batch summarization function call

        Args:
            state (dict): should validate against call_schema
        Returns:
            dict: the state dict will contain result:
            {
                # ...
                # The following key is overwritten or added
                "result" : "summary",
                "error_code": "Error String" # Optional
            }
        """
        with TimeMeasure("OffBatchSumm/Acall", "blue"):
            logger.info("=== FINAL AGGREGATION START ===")
            batches = []
            self.call_schema.validate(state)
            stop_time = time.time() + self.timeout
            target_start_batch_index = self.batcher.get_batch_index(
                state["start_index"]
            )
            target_end_batch_index = self.batcher.get_batch_index(state["end_index"])
            logger.info(f"DEBUG: Target Batch Start: {target_start_batch_index}")
            logger.info(f"DEBUG: Target Batch End: {target_end_batch_index}")
            logger.info(f"DEBUG: Timeout in {self.timeout} seconds")
            
            if target_end_batch_index == -1:
                logger.info(f"DEBUG: Current batch index: {self.curr_batch_i}")
                target_end_batch_index = self.curr_batch_i

            # Track fetched batch indices
            fetched_batch_indices = set()
            total_expected_batches = target_end_batch_index - target_start_batch_index + 1
            logger.info(f"DEBUG: Expecting {total_expected_batches} total batches")
            
            while time.time() < stop_time:
                # Only query for unfetched batches
                unfetched_indices = [
                    i
                    for i in range(target_start_batch_index, target_end_batch_index + 1)
                    if i not in fetched_batch_indices
                ]
                if not unfetched_indices:
                    logger.info("DEBUG: All batches found, breaking from wait loop")
                    break

                logger.info(f"DEBUG: Still waiting for batches: {unfetched_indices}")

                # Query only for new batches
                batch_filter = f"doc_type == 'caption_summary' and batch_i in [{','.join(map(str, unfetched_indices))}]"
                logger.info(f"DEBUG: Querying vector_db with filter: {batch_filter}")
                
                new_batches = await self.vector_db.aget_text_data(
                    fields=["text", "batch_i"], filter=batch_filter
                )
                logger.info(f"DEBUG: Vector_db returned {len(new_batches)} new batches")

                # Update fetched indices and add to results
                for batch in new_batches:
                    logger.info(f"DEBUG: Found batch {batch['batch_i']}, text length: {len(batch.get('text', ''))}")
                    logger.info(f"DEBUG: Batch {batch['batch_i']} text preview: {batch.get('text', '')[:200]}...")
                    fetched_batch_indices.add(batch["batch_i"])
                batches.extend(new_batches)

                # If we have all required batches, break
                if len(fetched_batch_indices) == total_expected_batches:
                    logger.info(
                        f"DEBUG: All {len(fetched_batch_indices)} batches fetched. Moving forward."
                    )
                    break
                else:
                    remaining = total_expected_batches - len(fetched_batch_indices)
                    logger.info(f"DEBUG: Need {remaining} more batches. Waiting...")
                    await asyncio.sleep(1)
                    continue

            # Sort batches by batch_i field
            batches.sort(key=lambda x: x["batch_i"])
            logger.info(f"DEBUG: Final batch count: {len(batches)}")
            logger.info(f"DEBUG: Batch indices found: {[b['batch_i'] for b in batches]}")
            
            # LOG: Complete aggregation input
            total_input_length = sum(len(batch.get("text", "")) for batch in batches)
            logger.info(f"DEBUG: Total aggregation input length: {total_input_length} characters")
            
            for i, batch in enumerate(batches):
                logger.info(f"DEBUG: Aggregation input batch {i} (batch_i={batch['batch_i']}): length={len(batch.get('text', ''))}")
                logger.info(f"DEBUG: Aggregation input batch {i} content: {batch.get('text', '')[:300]}...")

            if len(batches) == 0:
                logger.error("ERROR: No batch summaries found for final aggregation")
                state["result"] = ""
                state["error_code"] = "No batch summaries found"
            elif len(batches) > 0:
                with TimeMeasure("summ/acall/batch-aggregation-summary", "pink") as bas:
                    logger.info("DEBUG: Starting final aggregation with call_token_safe")
                    
                    # Get model name for logging
                    model_name = self.get_param("llm", "model")
                    logger.info(f"DEBUG: Final aggregation using model: {model_name}")
                    
                    with self._get_appropriate_callback() as cb:
                        logger.info(f"DEBUG: Calling call_token_safe for final aggregation with {len(batches)} batches")
                        result = await call_token_safe(
                            batches,
                            self.aggregation_pipeline,
                            self.recursion_limit,
                            model_name=model_name  # ADD MODEL NAME HERE TOO
                        )
                        logger.info(f"DEBUG: Final aggregation result length: {len(result) if result else 0}")
                        logger.info(f"DEBUG: Final aggregation result preview: {result[:300] if result else 'EMPTY'}...")
                        logger.info(f"DEBUG: Final aggregation result ending: ...{result[-300:] if result and len(result) > 300 else result}")
                        
                        state["result"] = result
                    logger.info("DEBUG: Summary Aggregation Done")
                    self.metrics.aggregation_tokens = cb.total_tokens
                    logger.info(
                        "Final Aggregation - Total Tokens: %s, "
                        "Prompt Tokens: %s, "
                        "Completion Tokens: %s, "
                        "Successful Requests: %s, "
                        "Total Cost (USD): $%s"
                        % (
                            cb.total_tokens,
                            cb.prompt_tokens,
                            cb.completion_tokens,
                            cb.successful_requests,
                            cb.total_cost,
                        ),
                    )
                self.metrics.aggregation_latency = bas.execution_time
            
            logger.info("=== FINAL AGGREGATION END ===")
            
        if self.log_dir:
            log_path = Path(self.log_dir).joinpath("summary_metrics.json")
            self.metrics.dump_json(log_path.absolute())
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        try:
            logger.info("Adding doc %d", doc_i)
            doc_meta.setdefault("is_first", False)
            doc_meta.setdefault("is_last", False)

            self.vector_db.add_summary(
                summary=doc,
                metadata={**doc_meta, "doc_type": "caption", "batch_i": -1},
            )

            with TimeMeasure("summ/aprocess_doc", "red") as bs:
                if not doc_meta["is_last"] and "file" in doc_meta:
                    if doc_meta["file"].startswith("rtsp://"):
                        # if live stream summarization
                        if "start_ntp" in doc_meta and "end_ntp" in doc_meta:
                            doc = (
                                f"<{doc_meta['start_ntp']}> <{doc_meta['end_ntp']}> "
                                + doc
                            )
                        else:
                            logger.info(
                                "start_ntp or end_ntp not found in doc_meta. "
                                "No timestamp will be added."
                            )
                    else:
                        # if file summmarization
                        if "start_pts" in doc_meta and "end_pts" in doc_meta:
                            doc = (
                                f"<{doc_meta['start_pts'] / 1e9:.2f}> <{doc_meta['end_pts'] / 1e9:.2f}> "
                                + doc
                            )
                        else:
                            logger.info(
                                "start_pts or end_pts not found in doc_meta. "
                                "No timestamp will be added."
                            )
                doc_meta["batch_i"] = doc_i // self.batch_size

                # logger.info("aprocess_doc() Add doc= %s doc_meta=%s", doc, doc_meta);

                #if self.endless_ai_enabled and doc != ".":
                    # Here we reset the image description that the RAG holds (e.g., "<0.00> <4.88> A boy in an orange shirt is dribbling a basketball and shooting at a basketball hoop.")
                    # Annoying, because the vlm spent time on it, but we do get better results this way - probably because the analysis is done using the grid images without the vlm results affecting it
                #    doc = ""

                batch = self.batcher.add_doc(doc, doc_i, doc_meta)
                if batch.is_full():
                    # Process the batch immediately when full
                    await asyncio.create_task(self._process_full_batch(batch))

            if self.summary_start_time is None:
                self.summary_start_time = bs.start_time

            self.metrics.summary_latency = bs.end_time - self.summary_start_time
        except Exception as e:
            logger.error(e)

    async def areset(self, state: dict):
        # TODO: use async method for drop data
        self.vector_db.drop_data(state["expr"])
        self.summary_start_time = None
        self.batcher.flush()
        self.metrics.reset()
        await asyncio.sleep(0.001)