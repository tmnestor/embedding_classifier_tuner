# This file makes the FTC_utils directory a Python package

# Import commonly used utilities
from FTC_utils.LoaderSetup import join_constructor
from FTC_utils.utils import preprocess_text
from FTC_utils.device_utils import get_device

# Import file utilities for error handling
try:
    from FTC_utils.file_utils import (
        load_config,
        load_csv_safely,
        load_model_safely,
        load_tokenizer_safely,
        load_checkpoint_safely,
        save_model_safely,
        save_csv_safely,
        ensure_dir,
        process_with_retry,
        visualize_with_fallback
    )
except ImportError:
    # File utilities might not be available in older versions
    pass

# Import logging utilities
from FTC_utils.logging_utils import tee_to_file

__all__ = [
    # Core utilities
    "join_constructor", 
    "preprocess_text", 
    "get_device",
    "tee_to_file",
    
    # File handling utilities (might not be available)
    "load_config",
    "load_csv_safely",
    "load_model_safely",
    "load_tokenizer_safely",
    "load_checkpoint_safely",
    "save_model_safely",
    "save_csv_safely",
    "ensure_dir",
    "process_with_retry",
    "visualize_with_fallback"
]
