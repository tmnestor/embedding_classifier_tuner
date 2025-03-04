"""
Environment utilities for the embedding classifier tuner.
"""

import os
import yaml
import warnings
from pathlib import Path

from .LoaderSetup import join_constructor, env_var_or_default_constructor


def check_environment():
    """
    Check if environment variables are properly set.
    Returns the BASE_ROOT path and creates directories if needed.
    Uses pathlib for safe path handling.
    """
    # Register YAML constructors to ensure they're available
    yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor(
        "!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader
    )

    # Load config to determine what the BASE_ROOT path is
    # Try to find config.yml in different locations
    if os.path.exists("config.yml"):
        config_path = "config.yml"
    else:
        # Try to find config.yml in the parent directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(parent_dir, "config.yml")

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                "Could not find config.yml in current or parent directory"
            )

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    base_root = config.get("BASE_ROOT")

    # Determine if environment variable was used or default
    env_var_name = "FTC_OUTPUT_PATH"
    env_var_set = env_var_name in os.environ

    if not env_var_set:
        warnings.warn(
            f"Environment variable {env_var_name} not set. Using default path: {base_root}",
            UserWarning,
        )

    # Sanitize and normalize the path
    base_path = Path(base_root).resolve()

    # Make sure the path is safe - not in system directories
    system_dirs = ["/bin", "/sbin", "/usr/bin", "/usr/sbin", "/etc", "/lib", "/var/lib"]
    if any(str(base_path).startswith(sys_dir) for sys_dir in system_dirs):
        raise ValueError(
            f"BASE_ROOT path {base_path} is in a system directory. Please use a different path."
        )

    # Create base directories if they don't exist
    if not base_path.exists():
        print(f"Creating base output directory: {base_path}")
        os.makedirs(base_path, exist_ok=True)

    # Create required subdirectories using pathlib for safety
    required_dirs = [
        "data",
        "checkpoints",
        "config",
        "evaluation",
        "calibration",
    ]

    for dir_name in required_dirs:
        dir_path = base_path / dir_name
        if not dir_path.exists():
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)

    return str(base_path)


def set_output_paths(config_path=None):
    """
    Set output paths based on config file.

    Args:
        config_path: Optional path to config.yml. If None, it will be auto-detected.
    """
    # Register YAML constructors
    yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor(
        "!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader
    )

    # Auto-detect config path if not provided
    if config_path is None:
        if os.path.exists("config.yml"):
            config_path = "config.yml"
        else:
            # Try parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(parent_dir, "config.yml")

            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    "Could not find config.yml in current or parent directory"
                )

    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Check environment
    check_environment()

    return config
