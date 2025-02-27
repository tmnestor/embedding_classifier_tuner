import torch


def get_device():
    """Get the best available device with proper MPS support for Mac.

    Returns:
        torch.device: The most appropriate device for tensor computation
    """
    if torch.cuda.is_available():
        print("CUDA GPU detected and will be used for acceleration.")
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("Apple Silicon GPU detected. Using MPS acceleration.")
        return torch.device("mps")
    else:
        print("No GPU acceleration available. Using CPU.")
        return torch.device("cpu")
