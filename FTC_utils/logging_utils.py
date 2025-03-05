"""
Logging utilities that tee all output to both console and log files.
"""

import os
import sys
import logging
import datetime
import yaml
from typing import Optional
from contextlib import contextmanager
from pathlib import Path

# Import path handling utilities
try:
    from utils.LoaderSetup import join_constructor, env_var_or_default_constructor
    
    # Register YAML constructors
    yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor("!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader)
except ImportError:
    # Fallback when imported early in initialization
    pass

# Load configuration to get BASE_ROOT path
try:
    # First try to find config.yml in the current directory
    if os.path.exists("config.yml"):
        config_path = "config.yml"
    else:
        # Try to find config.yml in the parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(parent_dir, "config.yml")
        
    # Load the configuration file
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        BASE_ROOT = config.get("BASE_ROOT", "./outputs")
except FileNotFoundError:
    # Fallback if config.yml cannot be found
    print("Warning: config.yml not found. Using default output directory.")
    BASE_ROOT = "./outputs"

# Create logs directory within BASE_ROOT using pathlib for safe path handling
LOGS_DIR = Path(BASE_ROOT) / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


class TeeLogger:
    """Logger that writes to both console and file simultaneously."""

    def __init__(self, filename: str, mode: str = "a", log_dir = LOGS_DIR, verbose: bool = False):
        """Initialize the TeeLogger.

        Args:
            filename (str): Base name of the log file
            mode (str): File opening mode ('a' for append, 'w' for write)
            log_dir: Directory to store log files (Path or str)
            verbose: Whether to print logging messages to console
        """
        self.terminal = sys.stdout
        self.verbose = verbose

        # Convert log_dir to Path object and create directory if it doesn't exist
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)

        # Add timestamp to filename to avoid overwriting previous logs
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sanitize filename
        clean_filename = "".join(
            c if c.isalnum() or c in "._-" else "_" for c in filename
        )

        # Create full log path with timestamp using pathlib for safe path handling
        log_path = str(log_dir_path / f"{clean_filename}_{timestamp}.log")

        self.log_file = open(log_path, mode, buffering=1, encoding="utf-8")
        self.filename = log_path

        if self.verbose:
            print(f"Logging output to: {log_path}")

    def write(self, message):
        """Write to both terminal and log file."""
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # Ensure immediate writing to disk

    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        """Return True if terminal is a tty (for compatibility)."""
        return self.terminal.isatty()

    def close(self):
        """Close the log file."""
        self.log_file.close()


@contextmanager
def capture_output(module_name: str):
    """Context manager that captures stdout and stderr to a log file.

    Args:
        module_name (str): Name of the module for the log file

    Yields:
        The path to the log file
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    # Create a TeeLogger for both stdout and stderr
    stdout_logger = TeeLogger(f"{module_name}_stdout")
    stderr_logger = TeeLogger(f"{module_name}_stderr")

    try:
        sys.stdout = stdout_logger
        sys.stderr = stderr_logger

        # Setup Python's logging module to also write to our file
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(stdout_logger),
                logging.FileHandler(stdout_logger.filename),
            ],
        )

        yield stdout_logger.filename
    finally:
        # Restore the original stdout and stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr

        print(f"Log file closed: {stdout_logger.filename}")
        stdout_logger.close()
        stderr_logger.close()


def configure_logger(module_name: str) -> logging.Logger:
    """Configure and return a logger for the given module."""
    # Create logs directory if it doesn't exist
    os.makedirs(LOGS_DIR, exist_ok=True)

    # Create timestamped log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOGS_DIR, f"{module_name}_{timestamp}.log")

    # Create and configure logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Inform user of log file location
    print(f"Logging to: {log_file}")

    return logger


def tee_to_file(module_name: str = None, verbose: bool = False):
    """Redirect stdout and stderr to both console and log file.

    Args:
        module_name (str, optional): Name of the module for the log file.
            If None, uses the caller's module name.
        verbose (bool): Whether to print log file locations to console.

    Returns:
        str: Path to the log file
    """
    if module_name is None:
        # Get the caller's module name if not provided
        import inspect

        frame = inspect.stack()[1]
        module_name = os.path.splitext(os.path.basename(frame.filename))[0]

    # Create TeeLogger for stdout and stderr with specified verbosity
    stdout_logger = TeeLogger(f"{module_name}_stdout", verbose=verbose)
    sys.stdout = stdout_logger

    stderr_logger = TeeLogger(f"{module_name}_stderr", verbose=verbose)
    sys.stderr = stderr_logger

    return stdout_logger.filename
