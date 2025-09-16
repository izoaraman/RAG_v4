import tiktoken


def count_num_tokens(text: str, model: str) -> int:
    """Returns the number of tokens in the given text."""
    try:
        # First try as an encoding name
        if model in ["cl100k_base", "p50k_base", "r50k_base"]:
            encoding = tiktoken.get_encoding(model)
        else:
            # Try as a model name
            encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # Fallback to cl100k_base if model not found
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))
