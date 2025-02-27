import re
import torch
from typing import Union, Dict, Any

# Define regex pattern for text cleaning
TAG_RE = re.compile(r"<[^>]+>")


def get_default_device() -> torch.device:
    """
    Get the default compute device for PyTorch.

    Returns:
        CUDA device if available, MPS for Apple Silicon, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess_text(text: Union[str, Any]) -> str:
    """
    Clean and normalize the input text.

    Args:
        text: Input text to preprocess

    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    text = TAG_RE.sub("", text)  # Remove HTML tags
    text = " ".join(text.split())  # Normalize whitespace
    return text


def create_directory_structure(base_path: str) -> Dict[str, str]:
    """
    Create a structured directory hierarchy for the project.

    Args:
        base_path: Base directory path

    Returns:
        Dictionary with directory paths for different purposes
    """
    import os

    # Define directory structure
    directories = {
        "data": os.path.join(base_path, "data"),
        "checkpoints": os.path.join(base_path, "checkpoints"),
        "logs": os.path.join(base_path, "logs"),
        "models": os.path.join(base_path, "models"),
        "evaluation": os.path.join(base_path, "evaluation"),
        "config": os.path.join(base_path, "config"),
    }

    # Create directories if they don't exist
    for dir_path in directories.values():
        os.makedirs(dir_path, exist_ok=True)

    return directories
