import ast
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

utils_path = os.path.join(os.path.dirname(__file__), "..", "src", "vss_ctx_rag", "utils", "utils.py")
with open(utils_path) as f:
    utils_src = f.read()

utils_mod = ast.parse(utils_src)
helper_src = None
for node in utils_mod.body:
    if isinstance(node, ast.FunctionDef) and node.name == "model_supports_multimodal_messages":
        helper_src = ast.get_source_segment(utils_src, node)
        break

helper_ns = {}
exec(helper_src, helper_ns)
model_supports_multimodal_messages = helper_ns["model_supports_multimodal_messages"]

LLM_TOOL_NAME = "llm"

class DummyMessage:
    def __init__(self, content):
        self.content = content

SystemMessage = DummyMessage
HumanMessage = DummyMessage


def _extract_function(path, class_name, method_name, inner_name):
    with open(path) as f:
        src = f.read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for cls_node in node.body:
                if isinstance(cls_node, ast.FunctionDef) and cls_node.name == method_name:
                    for inner in cls_node.body:
                        if isinstance(inner, ast.FunctionDef) and inner.name == inner_name:
                            return ast.get_source_segment(src, inner)
    raise ValueError("function not found")

BATCH_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "vss_ctx_rag", "functions", "summarization", "batch.py")
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "..", "src", "vss_ctx_rag", "functions", "rag", "graph_rag", "graph_retrieval.py")


class DummyLLM:
    def __init__(self, model):
        self.model = model
        self.model_id = model

class DummyTool:
    def __init__(self, model):
        self.llm = DummyLLM(model)

class DummyBatchSelf:
    def __init__(self, model):
        self.tool = DummyTool(model)
    def get_tool(self, name):
        return self.tool
    def get_param(self, *keys):
        return {
            "prompts": {"caption_summarization": "cap"}
        }[keys[0]][keys[1]]

class DummyGraphSelf:
    def __init__(self, model, endless=True):
        self.chat_llm = DummyLLM(model)
        self.endless_ai_enabled = endless
    def get_tool(self, name):
        return None

CHAT_SYSTEM_TEMPLATE = "chat"
CHAT_SYSTEM_GRID_TEMPLATE = "grid"

batch_src = _extract_function(BATCH_PATH, "BatchSummarization", "setup", "prepare_messages")
graph_src = _extract_function(GRAPH_PATH, "GraphRetrieval", "__init__", "prepare_messages")


def test_batch_prepare_messages_multimodal():
    ns = {
        "SystemMessage": SystemMessage,
        "HumanMessage": HumanMessage,
        "LLM_TOOL_NAME": LLM_TOOL_NAME,
        "model_supports_multimodal_messages": model_supports_multimodal_messages,
    }
    self_obj = DummyBatchSelf("gpt-4o")
    ns["self"] = self_obj
    exec(batch_src, ns)
    res = ns["prepare_messages"]({"input": "hello", "images": ["img"]})
    assert isinstance(res[1].content, list)
    assert any(b.get("type") == "image_url" for b in res[1].content[1:])


def test_batch_prepare_messages_text_only():
    ns = {
        "SystemMessage": SystemMessage,
        "HumanMessage": HumanMessage,
        "LLM_TOOL_NAME": LLM_TOOL_NAME,
        "model_supports_multimodal_messages": model_supports_multimodal_messages,
    }
    self_obj = DummyBatchSelf("gpt-3.5-turbo")
    ns["self"] = self_obj
    exec(batch_src, ns)
    res = ns["prepare_messages"]({"input": "hello", "images": ["img"]})
    assert res[1].content == "hello"


def test_graph_prepare_messages_multimodal():
    ns = {
        "SystemMessage": SystemMessage,
        "HumanMessage": HumanMessage,
        "model_supports_multimodal_messages": model_supports_multimodal_messages,
        "CHAT_SYSTEM_TEMPLATE": CHAT_SYSTEM_TEMPLATE,
        "CHAT_SYSTEM_GRID_TEMPLATE": CHAT_SYSTEM_GRID_TEMPLATE,
    }
    self_obj = DummyGraphSelf("claude-3-opus", endless=True)
    ns["self"] = self_obj
    exec(graph_src, ns)
    res = ns["prepare_messages"]({"input": "hi", "images": ["img"], "messages": []})
    assert isinstance(res[-1].content, list)
    assert any(b.get("type") == "image_url" for b in res[-1].content[1:])


def test_graph_prepare_messages_text_only():
    ns = {
        "SystemMessage": SystemMessage,
        "HumanMessage": HumanMessage,
        "model_supports_multimodal_messages": model_supports_multimodal_messages,
        "CHAT_SYSTEM_TEMPLATE": CHAT_SYSTEM_TEMPLATE,
        "CHAT_SYSTEM_GRID_TEMPLATE": CHAT_SYSTEM_GRID_TEMPLATE,
    }
    self_obj = DummyGraphSelf("gpt-3.5-turbo", endless=True)
    ns["self"] = self_obj
    exec(graph_src, ns)
    res = ns["prepare_messages"]({"input": "hi", "images": ["img"], "messages": []})
    assert res[-1].content == "User question: hi"
