import re

import torch

TAG_RE = re.compile(r"<[^>]+>")


def get_default_device():
    """Get the default compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def preprocess_text(text):
    """Clean and preprocess text for model input."""
    if not isinstance(text, str):
        return ""

    # Remove HTML tags if present
    TAG_RE = re.compile(r"<[^>]+>")
    text = TAG_RE.sub("", text)

    # Remove extra whitespace
    text = " ".join(text.split())

    return text
