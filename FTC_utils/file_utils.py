import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer

def load_config(config_path="config.yml"):
    """Safely load configuration file with proper error handling.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        SystemExit: If configuration file cannot be loaded
    """
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        print("Please ensure the file exists in the current directory.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML configuration file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading configuration: {e}")
        sys.exit(1)

def load_csv_safely(csv_path, required_columns=None):
    """Safely load a CSV file with proper error handling.
    
    Args:
        csv_path: Path to the CSV file
        required_columns: List of column names that must be present
        
    Returns:
        Pandas DataFrame with the loaded data
        
    Raises:
        SystemExit: If CSV file cannot be loaded or lacks required columns
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns if specified
        if required_columns:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        
        return df
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the file path and ensure the file exists.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file '{csv_path}' is empty.")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file '{csv_path}': {e}")
        print("Please check the file format.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading CSV file '{csv_path}': {e}")
        sys.exit(1)

def load_model_safely(model_name_or_path, device=None):
    """Safely load a pre-trained model with proper error handling.
    
    Args:
        model_name_or_path: Name or path of the model to load
        device: Device to load the model to
        
    Returns:
        Loaded model
        
    Raises:
        SystemExit: If model cannot be loaded
    """
    from utils.device_utils import get_device
    
    if device is None:
        device = get_device()
    
    try:
        # Try to load the model
        model = AutoModel.from_pretrained(model_name_or_path)
        model = model.to(device)
        return model
    except OSError as e:
        print(f"Error loading model '{model_name_or_path}': {e}")
        print("Please check the model name or path.")
        sys.exit(1)
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print(f"CUDA out of memory error while loading model '{model_name_or_path}'.")
            print("Try using a smaller batch size or a smaller model.")
            sys.exit(1)
        print(f"Runtime error loading model '{model_name_or_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading model '{model_name_or_path}': {e}")
        sys.exit(1)

def load_tokenizer_safely(model_name_or_path):
    """Safely load a tokenizer with proper error handling.
    
    Args:
        model_name_or_path: Name or path of the model whose tokenizer to load
        
    Returns:
        Loaded tokenizer
        
    Raises:
        SystemExit: If tokenizer cannot be loaded
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return tokenizer
    except OSError as e:
        print(f"Error loading tokenizer '{model_name_or_path}': {e}")
        print("Please check the model name or path.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading tokenizer '{model_name_or_path}': {e}")
        sys.exit(1)

def load_checkpoint_safely(model, checkpoint_path, optimizer=None, map_location=None):
    """Safely load a model checkpoint with proper error handling.
    
    Args:
        model: Model to load checkpoint into
        checkpoint_path: Path to the checkpoint file
        optimizer: Optimizer to load state into (optional)
        map_location: Device mapping function or device name
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        SystemExit: If checkpoint cannot be loaded
    """
    try:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Load checkpoint with warning suppression for weights_only
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        try:
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                # Try loading directly if checkpoint is just the state dict
                model.load_state_dict(checkpoint)
        except RuntimeError as e:
            # More specific error for model architecture mismatch
            if "size mismatch" in str(e) or "shape mismatch" in str(e):
                print(f"Error: Model architecture mismatch when loading checkpoint.")
                print("The model structure doesn't match the saved checkpoint structure.")
                print(f"Original error: {e}")
                sys.exit(1)
            raise e
            
        # Optionally load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
        return checkpoint
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the checkpoint path and ensure the file exists.")
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error loading checkpoint '{checkpoint_path}': {e}")
        if "CUDA out of memory" in str(e):
            print("Try using a smaller batch size or moving to CPU.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in checkpoint file: {e}")
        print("The checkpoint file may be corrupted or in an incompatible format.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading checkpoint '{checkpoint_path}': {e}")
        sys.exit(1)

def ensure_dir(directory_path):
    """Safely create directory with proper error handling.
    
    Args:
        directory_path: Path to the directory to create
        
    Returns:
        String path to the created directory
        
    Raises:
        SystemExit: If directory cannot be created
    """
    try:
        directory = Path(directory_path)
        directory.mkdir(parents=True, exist_ok=True)
        return str(directory)
    except PermissionError:
        print(f"Error: Permission denied when creating directory '{directory_path}'")
        print("Please check your file permissions.")
        sys.exit(1)
    except OSError as e:
        print(f"Error creating directory '{directory_path}': {e}")
        sys.exit(1)

def save_model_safely(model, save_path, additional_data=None):
    """Safely save a model with proper error handling.
    
    Args:
        model: Model to save
        save_path: Path where to save the model
        additional_data: Additional data to save with the model
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Prepare data to save
        save_data = {'model_state_dict': model.state_dict()}
        if additional_data:
            save_data.update(additional_data)
            
        # Save the model
        torch.save(save_data, save_path)
        return True
    except PermissionError:
        print(f"Error: Permission denied when saving model to '{save_path}'")
        print("Please check your file permissions.")
        return False
    except OSError as e:
        print(f"Error saving model to '{save_path}': {e}")
        return False
    except Exception as e:
        print(f"Unexpected error saving model to '{save_path}': {e}")
        return False

def save_csv_safely(df, save_path, index=False):
    """Safely save a DataFrame to CSV with proper error handling.
    
    Args:
        df: DataFrame to save
        save_path: Path where to save the CSV
        index: Whether to save DataFrame index
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Save the DataFrame
        df.to_csv(save_path, index=index)
        return True
    except PermissionError:
        print(f"Error: Permission denied when saving CSV to '{save_path}'")
        print("Please check your file permissions.")
        return False
    except OSError as e:
        print(f"Error saving CSV to '{save_path}': {e}")
        return False
    except Exception as e:
        print(f"Unexpected error saving CSV to '{save_path}': {e}")
        return False

def process_with_retry(func, max_retries=3, *args, **kwargs):
    """Execute a function with retry logic for transient errors.
    
    Args:
        func: Function to execute
        max_retries: Maximum number of retry attempts
        *args: Arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        Result of the function call
        
    Raises:
        Exception: If function fails after all retry attempts
    """
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (ConnectionError, TimeoutError, OSError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Error: {e}. Retrying in {wait_time} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"Error: {e}. Maximum retry attempts reached.")
                raise

def visualize_with_fallback(visualization_func, output_path, *args, **kwargs):
    """Generate visualization with graceful degradation.
    
    Args:
        visualization_func: Function that generates the visualization
        output_path: Path where to save the visualization
        *args: Arguments to pass to the visualization function
        **kwargs: Keyword arguments to pass to the visualization function
        
    Returns:
        True if visualization was successful, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        
        # Ensure directory exists
        directory = os.path.dirname(output_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
            
        # Generate the visualization
        visualization_func(*args, **kwargs)
        
        # Save the visualization
        plt.savefig(output_path)
        plt.close()
        return True
    except ImportError:
        print("Warning: matplotlib not available. Skipping visualization.")
        return False
    except Exception as e:
        print(f"Warning: Failed to generate visualization: {e}")
        print("Continuing without visualization.")
        return False