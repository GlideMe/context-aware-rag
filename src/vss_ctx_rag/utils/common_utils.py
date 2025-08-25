import re
from contextlib import contextmanager

def is_openai_model(model_name: str) -> bool:
    """Return True if the model name refers to an OpenAI model."""

    model = model_name.lower()

    openai_prefixes = (
        "gpt-",
        "gpt_image-",
        "dall-e",
        "davinci",
        "babbage",
        "codex",
        "chatgpt",
        "tts-",
        "whisper-",
    )

    if any(model.startswith(prefix) for prefix in openai_prefixes):
        return True

    # Matches o models like "o1", "o1-preview", "o3-mini", "o4-mini" etc.
    if re.match(r"^[o]\d", model):
        return True

    return False


def is_claude_model(model_name: str) -> bool:
    """Return True if the model name refers to an Anthropic Claude model."""

    model = model_name.lower()

    claude_prefixes = (
        "claude",
        "anthropic.claude",
        "anthropic.",
        "us.anthropic.claude",
        "us.anthropic.",
    )

    return any(model.startswith(prefix) for prefix in claude_prefixes)


def is_gemini_model(model_name: str) -> bool:
    """Return True if the model name refers to a Google Gemini model."""

    model = model_name.lower()

    gemini_prefixes = (
        "gemini",
        "gemini-",
        "models/gemini",
        "google/gemini",
    )

    return any(model.startswith(prefix) for prefix in gemini_prefixes)


@contextmanager
def dummy_callback():
    """Create a dummy callback for models that don't support token tracking (e.g., Gemini)."""
    class DummyCallback:
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        successful_requests = 1
        total_cost = 0.0
    yield DummyCallback()
