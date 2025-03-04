"""
Standardized configuration management utilities to ensure consistent
model architecture across training, tuning, and prediction.
"""

import os
import yaml
import hashlib
import torch.nn as nn
from typing import Dict, List, Any, Optional


def validate_config_architecture(config: Dict) -> bool:
    """Validates that a configuration has complete architecture information.

    Args:
        config: Configuration dictionary to validate

    Returns:
        bool: True if config contains complete architecture info, False otherwise
    """
    # Check for explicit architecture definition
    if "architecture" in config:
        arch = config["architecture"]
        # Minimal validation of architecture list
        if not isinstance(arch, list) or len(arch) == 0:
            return False

        # Check each layer has required parameters
        for layer in arch:
            if not isinstance(layer, dict):
                return False
            if "layer_type" not in layer:
                return False
            if layer["layer_type"] in ["dense_block", "linear"]:
                if "input_size" not in layer or "output_size" not in layer:
                    return False

        # Validate architecture hash if available
        if "architecture_hash" in config:
            current_hash = hashlib.md5(str(arch).encode()).hexdigest()
            if current_hash != config["architecture_hash"]:
                return False

        return True

    # Check for n_hidden + hidden_units_x pattern
    elif "n_hidden" in config:
        n_hidden = config["n_hidden"]
        input_dim = config.get("input_dim")
        output_dim = config.get("output_dim")

        # Need both input and output dimensions
        if not input_dim or not output_dim:
            return False

        # Check all hidden layers have defined widths
        for i in range(n_hidden):
            if f"hidden_units_{i}" not in config:
                return False

        return True

    # No architecture information
    return False


def describe_architecture(config: Dict) -> str:
    """Returns a human-readable description of the model architecture.

    Args:
        config: Configuration dictionary

    Returns:
        str: Text description of the architecture
    """
    if "architecture" in config:
        arch = config["architecture"]
        layers = []

        for layer in arch:
            if layer["layer_type"] == "dense_block":
                activation = layer.get("activation", "unknown")
                layer_str = f"DenseBlock({layer['input_size']} → {layer['output_size']}, {activation})"
                layers.append(layer_str)
            elif layer["layer_type"] == "linear":
                layers.append(f"Linear({layer['input_size']} → {layer['output_size']})")

        return " → ".join(layers)

    elif "n_hidden" in config:
        n_hidden = config["n_hidden"]
        input_dim = config.get("input_dim", "?")
        layers = [f"Input({input_dim})"]

        prev_width = input_dim
        for i in range(n_hidden):
            width_key = f"hidden_units_{i}"
            if width_key in config:
                width = config[width_key]
                layers.append(f"Layer{i + 1}({width})")
                prev_width = width

        output_dim = config.get("output_dim", "?")
        layers.append(f"Output({output_dim})")
        return " → ".join(layers)

    else:
        return "Default architecture (not specified in config)"


def update_config_file(config: Dict, filepath: str):
    """Updates a configuration file with new or modified parameters.

    Args:
        config: Configuration dictionary with updates
        filepath: Path to the configuration file
    """
    # Create directories if needed
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Update architecture hash if architecture changed
    if "architecture" in config:
        arch_str = str(config["architecture"])
        config["architecture_hash"] = hashlib.md5(arch_str.encode()).hexdigest()

    # Write the updated configuration
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
