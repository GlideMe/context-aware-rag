# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import asyncio
import os
from re import compile
import traceback

from langchain_core.output_parsers import StrOutputParser

from vss_ctx_rag.base import Function
from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from vss_ctx_rag.tools.storage import StorageTool
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.functions.rag.graph_rag.graph_retrieval import GraphRetrieval
from vss_ctx_rag.utils.globals import (
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
    DEFAULT_MULTI_CHANNEL,
    DEFAULT_CHAT_HISTORY,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from vss_ctx_rag.utils.utils import remove_think_tags

import base64
from pydantic import BaseModel, Field
from typing import List

class GraphRetrievalFunc(Function):
    """GraphRetrievalFunc Function"""

    config: dict
    output_parser = StrOutputParser()
    graph_db: Neo4jGraphDB
    vector_db: StorageTool
    metrics = GraphMetrics()

    def setup(self):
        self.graph_db = self.get_tool("graph_db")
        self.vector_db = self.get_tool("vector_db")
        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.top_k = (
            self.get_param("params", "top_k", required=False)
            if self.get_param("params", "top_k", required=False)
            else DEFAULT_RAG_TOP_K
        )
        self.multi_channel = (
            self.get_param("params", "multi_channel", required=False)
            if self.get_param("params", "multi_channel", required=False)
            else DEFAULT_MULTI_CHANNEL
        )
        self.chat_history = self.get_param("params", "chat_history", required=False)
        if self.chat_history is None:
            self.chat_history = DEFAULT_CHAT_HISTORY

        uuid = self.get_param("params", "uuid", required=False)
        self.log_dir = os.environ.get("VIA_LOG_DIR", None)

        self.endless_ai_enabled = self.get_param("params", "endless_ai_enabled")
        self.chat_system_prompt = self.get_param("params", "chat_system_prompt", required=False)
        self.chunk_size = self.get_param("params", "chunk_size", required=False)
        if self.chunk_size is None:
            self.chunk_size = 0

        try:
            self.graph_retrieval = GraphRetrieval(
                llm=self.chat_llm,
                graph=self.graph_db,
                multi_channel=self.multi_channel,
                uuid=uuid,
                top_k=self.top_k,
                endless_ai_enabled=self.endless_ai_enabled,
                chat_system_prompt=self.chat_system_prompt,
            )
        except Exception as e:
            logger.error(f"Error initializing GraphRetrieval: {e}")
            raise
        self.regex_object = compile(r"<(\d+[.]\d+)>")

    async def acall(self, state: dict) -> dict:

        def get_grids_using_graph(question: str, docs) -> set[str]:
            prompt_token_cutoff = 5
            sorted_documents = sorted(
                docs,
                key=lambda doc: doc.state.get("query_similarity_score", 0),
                reverse=True,
            )
            documents = sorted_documents[:prompt_token_cutoff]

            unique_images = []
            unique_images_set = set()
            for doc in documents:
                for chunkdetail in doc.metadata["chunkdetails"]:
                    logger.info(f"ERANERAN chunkIdx={chunkdetail['chunkIdx']}")
                    if chunkdetail['grid_filenames']:
                        for grid_filename in chunkdetail['grid_filenames'].split("|"):
                            if grid_filename not in unique_images_set:
                                unique_images_set.add(grid_filename)
                                unique_images.append(grid_filename)

            return unique_images

        async def get_grids_using_llm(question: str) -> set[str]:
            all_batches = await self.vector_db.aget_text_data(
                fields=["text", "batch_i", "grid_filenames"], filter="doc_type == 'caption_summary' and grid_filenames != ''"
            )
            content_blocks = []
            batch_index_to_grids = {}
            for batch in all_batches:
                content_blocks.append({"type": "text", "text": batch['text']})
                batch_index_to_grids[batch['batch_i']] = batch['grid_filenames']
            content_blocks.append({"type": "text", "text": f"User question: {question}"})

            class TimeRange(BaseModel):
                start: float = Field(..., ge=0, description="Start of the range in seconds.")
                end: float = Field(..., ge=0, description="End of the range in seconds; must be greater than or equal to start.")

            class TimeRangesList(BaseModel):
                time_ranges: List[TimeRange] = Field(..., description="A list of time ranges, each with 'start' and 'end' in seconds.")

            structured_llm = self.chat_llm.with_structured_output(TimeRangesList)

            system_content = """
Your task is to analyze the provided textual summaries and return 3 relevant time-ranges (in seconds) when the queried event occurs.
If the user asks for the earliest, initial, or first occurrence, return only the first 3 distinct moments when the event begins or is set up.
If the user asks for the latest, most recent, or final occurrence, return only the last 3 distinct moments when the event is initiated, set up, or adjusted.
Only select timestamps clearly described as starting an event or range (e.g., “starts,” “begins to,” “sets up,” “removes,” “concludes”).
If an event spans a time range, return the start of the range. Do not include multiple timestamps from the same continuous event or range—only one per distinct occurrence.
Return exactly 3 time-ranges from distinct event occurrences.
Output only the time-ranges, no extra text.
"""

            time_ranges_list = structured_llm.invoke([SystemMessage(content=system_content), HumanMessage(content=content_blocks)])
            logger.info(f"time_ranges_list={time_ranges_list}")

            # Add grid images according to chunks
            unique_images = []
            if time_ranges_list:
                unique_images_set = set()
                for time_range in time_ranges_list.time_ranges:
                    batch_index = int(time_range.start / self.chunk_size)
                    grid_str = batch_index_to_grids.get(batch_index)
                    if grid_str:
                        for grid_filename in grid_str.split("|"):
                            if grid_filename not in unique_images_set:
                                unique_images_set.add(grid_filename)
                                unique_images.append(grid_filename)
            return unique_images

        try:
            question = state.get("question", "").strip()
            if not question:
                raise ValueError("No input provided in state.")

            if question.lower() == "/clear":
                logger.debug("Clearing chat history...")
                self.graph_retrieval.clear_chat_history()
                state["response"] = "Cleared chat history"
                return state

            with TimeMeasure("GraphRetrieval/HumanMessage", "blue"):
                user_message = HumanMessage(content=question)
                self.graph_retrieval.add_message(user_message)

            bypass_graph = True # use LLM with batches summary to find the relevant grids
            if self.endless_ai_enabled and bypass_graph:
                unique_images = await get_grids_using_llm(question)
                formatted_docs = ""
            else:
                docs = self.graph_retrieval.retrieve_documents()
                if self.endless_ai_enabled:
                    unique_images = get_grids_using_graph(question, docs)
                else:
                    unique_images = []

                formatted_docs = self.graph_retrieval.process_documents(docs)
                # logger.info(f"formatted_docs={formatted_docs}")
                # logger.info(f"formatted_docs (first)={formatted_docs.split('\n\n\n')[0]}")
                # logger.info(f"first chunk info={docs[0].metadata['chunkdetails']}")


            logger.info(f"unique_images={unique_images}")
            if unique_images:
                def image_file_to_base64(filepath):
                    # Open the image file in binary mode
                    with open(filepath, 'rb') as image_file:
                        image_data = image_file.read()

                    # Encode the binary data to base64
                    base64_data = base64.b64encode(image_data)

                    # Convert bytes to a string
                    return base64_data.decode('utf-8')
                images = [image_file_to_base64(img) for img in unique_images]
            else:
                images = []

            if formatted_docs or images:
                ai_response = self.graph_retrieval.get_response(
                    question, formatted_docs, images
                )
                answer = remove_think_tags(ai_response.content)
                #logger.info(f"question={question}, answer={answer}")

                if self.chat_history:
                    with TimeMeasure("GraphRetrieval/AIMsg", "red"):
                        ai_message = AIMessage(content=answer)
                        self.graph_retrieval.add_message(ai_message)

                    self.graph_retrieval.summarize_chat_history()

                    logger.debug("Summarizing chat history thread started.")
                else:
                    self.graph_retrieval.clear_chat_history()
            else:
                formatted_docs = "No documents retrieved."
                answer = "Sorry, I don't see that in the video."
                self.graph_retrieval.chat_history.messages.pop()

            state["response"] = answer
            state["response"] = self.regex_object.sub(r"\g<1>", state["response"])

            if "formatted_docs" in state:
                state["formatted_docs"].append(formatted_docs)
            else:
                state["formatted_docs"] = [formatted_docs]

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in QA %s", str(e))
            state["response"] = "That didn't work. Try another question."

        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        pass

    async def areset(self, state: dict):
        self.graph_retrieval.clear_chat_history()
        await asyncio.sleep(0.01)
