import re

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
