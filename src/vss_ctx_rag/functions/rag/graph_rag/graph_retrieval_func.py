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

import json # TODO:REMOVE!!!

from langchain_core.output_parsers import StrOutputParser

from vss_ctx_rag.base import Function
from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.functions.rag.graph_rag.graph_retrieval import GraphRetrieval
from vss_ctx_rag.utils.globals import (
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
    DEFAULT_MULTI_CHANNEL,
    DEFAULT_CHAT_HISTORY,
)
from langchain_core.messages import HumanMessage, AIMessage
from vss_ctx_rag.utils.utils import remove_think_tags

from langchain_core.runnables import RunnableLambda, RunnableSequence
import base64

class GraphRetrievalFunc(Function):
    """GraphRetrievalFunc Function"""

    config: dict
    output_parser = StrOutputParser()
    graph_db: Neo4jGraphDB
    metrics = GraphMetrics()

    def setup(self):
        self.graph_db = self.get_tool("graph_db")
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

        self.endless_ai_enabled = self.get_param("endless_ai_enabled")
        self.chat_system_prompt = self.get_param("params", "chat_system_prompt", required=False)

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

            docs = self.graph_retrieval.retrieve_documents()

            if docs:
                if self.endless_ai_enabled:
                    # Add grid images according to chunks
                    #logger.info(f"docs={repr(docs)}")
                    prompt_token_cutoff = 5
                    sorted_documents = sorted(
                        docs,
                        key=lambda doc: doc.state.get("query_similarity_score", 0),
                        reverse=True,
                    )
                    documents = sorted_documents[:prompt_token_cutoff]

                    unique_images = set()
                    for doc in documents:
                        for chunkdetail in doc.metadata["chunkdetails"]:
                            if chunkdetail['grid_filenames']:
                                unique_images.update(chunkdetail['grid_filenames'].split('|'))

                    # logger.info(f"unique_images={list(unique_images)}")

                    def image_file_to_base64(filepath):
                        # Open the image file in binary mode
                        with open(filepath, 'rb') as image_file:
                            image_data = image_file.read()

                        # Encode the binary data to base64
                        base64_data = base64.b64encode(image_data)

                        # Convert bytes to a string
                        return base64_data.decode('utf-8')
                    images = [image_file_to_base64(img) for img in list(unique_images)]
                else:
                    images = []

                formatted_docs = self.graph_retrieval.process_documents(docs)
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
