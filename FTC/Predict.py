#!/usr/bin/env python3
# Standard library imports
import os
import sys
import argparse
from pathlib import Path

# Add the parent directory (project root) to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Third-party imports
import yaml
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer
from tqdm import tqdm

# Local imports
from FTC_utils.device_utils import get_device
from FTC_utils.LoaderSetup import join_constructor, env_var_or_default_constructor
from FTC_utils.conformal import ConformalPredictor
from FTC_utils.logging_utils import tee_to_file
from FTC_utils.env_utils import check_environment, set_output_paths

# Register YAML constructors
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
yaml.add_constructor(
    "!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader
)

# Check environment and create necessary directories
check_environment()

# Import shared components to ensure consistency
from FTC.BertClassification import (
    load_triplet_model,
    create_classifier_from_config,
    BertClassifier,
    preprocess_text,
    TextDataset,
)


# New on-the-fly prediction dataset
class PredictionDataset(Dataset):
    """Dataset for making predictions with text data.
    
    This dataset preprocesses text inputs for inference without requiring labels.
    It applies tokenization and returns input tensors with dummy labels.
    
    Args:
        texts (List[str]): List of text samples to process
        tokenizer: Tokenizer instance used for text encoding
        max_length (int): Maximum sequence length for tokenization
    """
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """Get the number of samples in the dataset.
        
        Returns:
            int: Number of text samples
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """Get tokenized input at the specified index.
        
        Args:
            idx (int): Index of the text sample to retrieve
            
        Returns:
            tuple: Dictionary of input tensors and a dummy label (0)
        """
        text = preprocess_text(self.texts[idx])
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Return as tuple (inputs, dummy_label) to match TokenizedDataset format
        return {k: v.squeeze(0) for k, v in encoded.items()}, 0


def load_model():
    """Load model components for programmatic usage.
    
    This function loads all necessary components for inference:
    1. Configuration from config.yml
    2. Pretrained tokenizer specified in the config
    3. Triplet model for embeddings
    4. Label encoder fit on training data
    5. Classifier model with the optimal configuration
    
    Returns:
        tuple: (model, tokenizer, label_encoder) where:
            - model (BertClassifier): The full classifier model for inference
            - tokenizer: Tokenizer for preprocessing text
            - label_encoder (LabelEncoder): Encoder to map numeric predictions to class labels
            
    Raises:
        Exception: If configuration file or model components cannot be loaded
    """
    # Load configuration with proper error handling
    try:
        from FTC_utils.file_utils import load_config

        config = load_config("config.yml")
    except ImportError:
        # Fallback if file_utils is not available
        try:
            with open("config.yml", "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration: {e}")
            print("Please ensure config.yml exists in the current directory.")
            sys.exit(1)

    device = get_device()

    # Load tokenizer from MODEL_PATH or MODEL_NAME
    if "MODEL_PATH" in config:
        print(f"Loading tokenizer from MODEL_PATH: {config['MODEL_PATH']}")
        tokenizer = AutoTokenizer.from_pretrained(config["MODEL_PATH"], local_files_only=True)
    else:
        print(f"Loading tokenizer from MODEL_NAME: {config['MODEL_NAME']}")
        tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])

    # Load triplet model
    triplet_model = load_triplet_model(device)
    classifier_input_size = triplet_model.base_model.config.hidden_size

    # Get class mapping from training data
    train_df = pd.read_csv(config["TRAIN_CSV"])
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["labels"])
    num_classes = len(label_encoder.classes_)

    # Load best config
    best_config_file = os.path.join(
        config["BEST_CONFIG_DIR"], f"best_config_{config['STUDY_NAME']}.yml"
    )
    with open(best_config_file, "r") as f:
        best_config = yaml.safe_load(f)

    # Create classifier and full model
    classifier = create_classifier_from_config(
        best_config, classifier_input_size, num_classes
    )
    model = BertClassifier(triplet_model, classifier).to(device)

    # Load weights
    checkpoint_path = os.path.join(config["CHECKPOINT_DIR"], "model", "best_model.pt")
    # Load checkpoint with warning suppression for weights_only
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, tokenizer, label_encoder


def predict_labels(model, tokenizer, texts, label_encoder, batch_size=32):
    """Predict labels for a list of texts.
    
    Args:
        model (BertClassifier): The model to use for predictions
        tokenizer: Tokenizer for preprocessing text
        texts (List[str]): List of text samples to classify
        label_encoder (LabelEncoder): Encoder to map numeric predictions to class labels
        batch_size (int, optional): Batch size for inference. Defaults to 32.
        
    Returns:
        numpy.ndarray: Array of predicted class labels
    """
    device = next(model.parameters()).device

    # Get max sequence length from config
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        max_seq_len = config["MAX_SEQ_LEN"]

    # Create dataset and dataloader
    dataset = PredictionDataset(texts, tokenizer, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch_input, _ in dataloader:
            batch_input = {k: v.to(device) for k, v in batch_input.items()}
            logits = model(batch_input)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)

    return label_encoder.inverse_transform(all_preds)


def predict_with_probs(model, tokenizer, texts, label_encoder, batch_size=32):
    """Predict labels and return probabilities for a list of texts.
    
    Args:
        model (BertClassifier): The model to use for predictions
        tokenizer: Tokenizer for preprocessing text
        texts (List[str]): List of text samples to classify
        label_encoder (LabelEncoder): Encoder to map numeric predictions to class labels
        batch_size (int, optional): Batch size for inference. Defaults to 32.
        
    Returns:
        tuple: A tuple containing:
            - numpy.ndarray: Array of predicted class labels
            - numpy.ndarray: Array of class probabilities for each prediction
    """
    device = next(model.parameters()).device

    # Get max sequence length from config
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        max_seq_len = config["MAX_SEQ_LEN"]

    # Create dataset and dataloader
    dataset = PredictionDataset(texts, tokenizer, max_seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_probs = []

    # Using tqdm imported at the top

    progress_bar = tqdm(
        dataloader, desc="Predicting", leave=False, total=len(dataloader)
    )

    with torch.no_grad():
        for batch_input, _ in progress_bar:
            batch_input = {k: v.to(device) for k, v in batch_input.items()}
            logits = model(batch_input)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())

            # Clear CUDA cache periodically for large datasets
            if torch.cuda.is_available() and len(all_preds) % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

    return label_encoder.inverse_transform(all_preds), np.array(all_probs)


def predict_large_file(
    model,
    tokenizer,
    input_file,
    output_file,
    label_encoder,
    batch_size=32,
    chunk_size=5000,
):
    """Process a large file in chunks to avoid memory issues.
    
    This function reads a large CSV file in chunks, makes predictions on each chunk,
    and writes the results to an output file. It includes progress reporting and
    memory management for large datasets.
    
    Args:
        model (BertClassifier): The model to use for predictions
        tokenizer: Tokenizer for preprocessing text
        input_file (str): Path to the input CSV file
        output_file (str): Path to save the predictions CSV
        label_encoder (LabelEncoder): Encoder to map numeric predictions to class labels
        batch_size (int, optional): Batch size for inference. Defaults to 32.
        chunk_size (int, optional): Number of rows to process per chunk. Defaults to 5000.
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    device = next(model.parameters()).device
    model.eval()

    # Get max sequence length from config
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        max_seq_len = config["MAX_SEQ_LEN"]

    # Initialize output file
    output_header = True

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Process in chunks to avoid memory issues
    print(f"Processing large file {input_file} in chunks of {chunk_size} rows...")

    # Get total number of rows for progress reporting
    total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract header
    processed_rows = 0

    for df_chunk in pd.read_csv(input_file, chunksize=chunk_size):
        # Make predictions on chunk
        texts = df_chunk["text"].apply(preprocess_text).tolist()
        predictions, probs = predict_with_probs(
            model, tokenizer, texts, label_encoder, batch_size
        )

        # Add predictions to chunk
        df_chunk["predicted_label"] = predictions
        df_chunk["confidence_score"] = np.max(probs, axis=1)

        # Add prediction sets using conformal predictor
        pred_loader = DataLoader(
            PredictionDataset(texts, tokenizer, max_seq_len),
            batch_size=batch_size,
            shuffle=False,
        )

        all_logits = []
        with torch.no_grad():
            for batch_input, _ in pred_loader:
                batch_input = {k: v.to(device) for k, v in batch_input.items()}
                logits = model(batch_input)
                all_logits.append(logits)

        # Get prediction sets using conformal prediction
        if all_logits:  # Ensure we have logits before proceeding
            combined_logits = torch.cat(all_logits, dim=0)

            # Initialize conformal predictor if needed
            from FTC_utils.conformal import ConformalPredictor

            significance = 0.1  # Default significance level
            conformal_predictor = ConformalPredictor(significance=significance)

            # Try to load calibration, if not available, use max class only
            try:
                with open("config.yml", "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                calibration_file = config["CALIBRATION_PATH"]
                conformal_predictor.load_calibration(calibration_file)

                # Get prediction sets and scores
                pred_sets, scores = conformal_predictor.get_prediction_sets(
                    combined_logits
                )

                # Convert prediction sets to labels
                prediction_sets = [
                    [label_encoder.inverse_transform([idx])[0] for idx in pred_set]
                    for pred_set in pred_sets
                ]

                # Add to DataFrame
                df_chunk["prediction_set"] = prediction_sets
                df_chunk["prediction_set_size"] = [
                    len(pred_set) for pred_set in pred_sets
                ]
            except:
                # Fallback if conformal calibration is not available
                print(
                    "Conformal calibration not available, using only max class predictions"
                )
                df_chunk["prediction_set"] = df_chunk["predicted_label"].apply(
                    lambda x: [x]
                )
                df_chunk["prediction_set_size"] = 1

        # Append to output file
        df_chunk.to_csv(output_file, mode="a", header=output_header, index=False)
        output_header = False  # Only write header once

        # Update progress
        processed_rows += len(df_chunk)
        print(
            f"Processed {processed_rows}/{total_rows} rows ({processed_rows / total_rows * 100:.1f}%)"
        )

        # Clear memory
        del df_chunk, texts, predictions, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Large file processing complete. Results saved to {output_file}")


def predict_with_confidence(
    model,
    tokenizer,
    texts,
    label_encoder,
    conformal_predictor,
    device=None,
    batch_size=32,
    max_seq_len=128,
):
    """Make predictions with conformal prediction sets.

    Args:
        model: The model to use for predictions
        tokenizer: Tokenizer for preprocessing
        texts: List of texts to predict
        label_encoder: Encoder for class labels
        conformal_predictor: Calibrated conformal predictor
        device: Torch device
        batch_size: Batch size for prediction
        max_seq_len: Maximum sequence length

    Returns:
        Dictionary with predictions, confidence scores, and prediction sets
    """
    if device is None:
        device = get_device()

    # Create dataset and loader
    pred_dataset = PredictionDataset(texts, tokenizer, max_seq_len)
    pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)

    # Make predictions with conformal prediction sets
    all_logits = []

    # Using tqdm imported at the top

    print(f"Generating predictions for {len(texts)} texts...")
    with torch.no_grad():
        for batch_input, _ in tqdm(pred_loader, desc="Predicting", leave=True):
            # Handle input dict
            batch_input = {k: v.to(device) for k, v in batch_input.items()}
            logits = model(batch_input)
            all_logits.append(logits)

    # Combine all batches
    logits = torch.cat(all_logits, dim=0)

    # Get predictions
    preds = torch.argmax(logits, dim=1).cpu().numpy()
    probs = torch.softmax(logits, dim=1).cpu().numpy()

    # Get conformal prediction sets
    pred_sets, scores = conformal_predictor.get_prediction_sets(logits)

    # Convert to labels
    predicted_labels = label_encoder.inverse_transform(preds)

    # Create prediction sets with labels
    prediction_sets = [
        [label_encoder.inverse_transform([idx])[0] for idx in pred_set]
        for pred_set in pred_sets
    ]

    # Complete explanation code removed

    # Combine results
    results = {
        "predicted_label": predicted_labels,
        "confidence_score": 1 - np.array(scores),
        "prediction_set": prediction_sets,
        "prediction_set_size": [len(pred_set) for pred_set in pred_sets],
        "logits": logits.cpu().numpy(),
        "probabilities": probs,
    }

    return results


def main():
    """Main function to make predictions with the trained model."""
    global log_file

    # Start capturing output to log file
    log_file = tee_to_file("Predict")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--output", type=str, help="Path to output predictions CSV file"
    )
    parser.add_argument(
        "--use-test-split", 
        action="store_true", 
        help="Use the test split created by TripletTraining.py instead of a custom input file"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed model information"
    )
    parser.add_argument(
        "--config", type=str, help="Path to specific config file (overrides default)"
    )
    parser.add_argument(
        "--significance",
        type=float,
        default=0.1,
        help="Significance level for conformal prediction",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for prediction"
    )
    # Explanation arguments removed
    parser.add_argument(
        "--large-file",
        action="store_true",
        help="Process input file in chunks for very large datasets",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Number of rows to process at once when using --large-file",
    )
    args = parser.parse_args()

    # Load configuration with proper error handling
    config_path = args.config or "config.yml"
    try:
        from FTC_utils.file_utils import load_config

        config = load_config(config_path)
    except ImportError:
        # Fallback if file_utils is not available
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading configuration from {config_path}: {e}")
            print("Please ensure the configuration file exists and is valid YAML.")
            sys.exit(1)

    # Device setup
    device = get_device()
    print(f"Using device: {device}")

    # Paths and constants - using pathlib for safe path handling
    DATA_ROOT = Path(config["DATA_PATH"])
    TRAIN_FILE = config["TRAIN_CSV"]
    VAL_FILE = config["VAL_CSV"] 
    TEST_FILE = config["TEST_CSV"]
    CHECKPOINT_DIR = config["CHECKPOINT_DIR"]
    EMBEDDING_PATH = config["EMBEDDING_PATH"]
    BEST_CONFIG_DIR = config["BEST_CONFIG_DIR"]
    STUDY_NAME = config["STUDY_NAME"]
    MAX_SEQ_LEN = config["MAX_SEQ_LEN"]
    
    # Determine input file based on arguments
    if args.use_test_split:
        # Use the test split created by TripletTraining.py
        if not os.path.exists(TEST_FILE):
            print(f"ERROR: Test split file not found at {TEST_FILE}")
            print("Please run TripletTraining.py first to create the splits.")
            print("Run: python FTC/TripletTraining.py --epochs 10 --lr 2e-5")
            sys.exit(1)
        
        INPUT_FILE = TEST_FILE
        print(f"Using test split from TripletTraining.py: {INPUT_FILE}")
    elif args.input:
        # Use custom input file
        INPUT_FILE = str(Path(args.input).resolve())
        print(f"Using custom input file: {INPUT_FILE}")
    else:
        # Default to using test split
        if os.path.exists(TEST_FILE):
            INPUT_FILE = TEST_FILE
            print(f"Using test split from TripletTraining.py: {INPUT_FILE}")
        else:
            # Fallback to a default test file
            INPUT_FILE = str(DATA_ROOT / "testing_data.csv")
            print(f"WARNING: Test split not found. Using default file: {INPUT_FILE}")
            if not os.path.exists(INPUT_FILE):
                print(f"ERROR: Default test file not found at {INPUT_FILE}")
                print("Please provide an input file with --input or run TripletTraining.py first.")
                sys.exit(1)
    
    # Output file handling
    if args.output:
        OUTPUT_FILE = args.output
    else:
        # Create a cleaner filename by using the base filename + "_predictions.csv"
        input_path = Path(INPUT_FILE)
        OUTPUT_FILE = str(input_path.parent / f"{input_path.stem}_predictions.csv")
    OUTPUT_FILE = str(Path(OUTPUT_FILE).resolve())

    # Load tokenizer from MODEL_PATH or MODEL_NAME
    if "MODEL_PATH" in config:
        print(f"Loading tokenizer from MODEL_PATH: {config['MODEL_PATH']}")
        tokenizer = AutoTokenizer.from_pretrained(config["MODEL_PATH"], local_files_only=True)
    else:
        print(f"Loading tokenizer from MODEL_NAME: {config['MODEL_NAME']}")
        tokenizer = AutoTokenizer.from_pretrained(config["MODEL_NAME"])

    # Load triplet model with proper error handling
    try:
        from FTC_utils.file_utils import load_model_safely

        # Try to use enhanced error handling first
        triplet_model = load_triplet_model(device)
    except Exception as e:
        print(f"Error loading triplet model: {e}")
        print("Unable to continue without a valid model. Exiting.")
        sys.exit(1)

    try:
        classifier_input_size = triplet_model.base_model.config.hidden_size
    except AttributeError as e:
        print(f"Error accessing model dimensions: {e}")
        print("Model structure appears to be invalid. Exiting.")
        sys.exit(1)

    # Get class mapping from training data with error handling
    try:
        from FTC_utils.file_utils import load_csv_safely

        # Check if training file exists
        if not os.path.exists(TRAIN_FILE):
            raise FileNotFoundError(f"Training data file not found: {TRAIN_FILE}")

        train_df = pd.read_csv(TRAIN_FILE)

        # Verify required columns exist
        if "labels" not in train_df.columns:
            raise ValueError(f"Required column 'labels' not found in {TRAIN_FILE}")

        label_encoder = LabelEncoder()
        label_encoder.fit(train_df["labels"])
        num_classes = len(label_encoder.classes_)

        if num_classes == 0:
            raise ValueError("No classes found in training data")

        print(f"Found {num_classes} classes: {label_encoder.classes_}")
    except Exception as e:
        print(f"Error loading training data: {e}")
        print("Unable to continue without valid training data. Exiting.")
        sys.exit(1)

    # Load the best configuration file
    if args.config:
        best_config_file = str(Path(args.config).resolve())
    else:
        best_config_file = str(Path(BEST_CONFIG_DIR) / f"best_config_{STUDY_NAME}.yml")

    if not os.path.exists(best_config_file):
        print(f"ERROR: Configuration file not found at {best_config_file}")
        return

    with open(best_config_file, "r") as f:
        best_config = yaml.safe_load(f)
    print(f"Loaded configuration from: {best_config_file}")

    # Configuration validation
    expected_input = best_config.get("input_dim")
    expected_output = best_config.get("output_dim")

    if expected_input and expected_input != classifier_input_size:
        print(
            f"WARNING: Config specifies input_dim={expected_input} but model uses {classifier_input_size}"
        )

    if expected_output and expected_output != num_classes:
        print(
            f"WARNING: Config specifies output_dim={expected_output} but data has {num_classes} classes"
        )

    # Print configuration details in verbose mode
    if args.verbose:
        print("\nConfiguration details:")
        for key, value in best_config.items():
            if key != "architecture":  # Skip architecture because it's large
                print(f"  {key}: {value}")

        if "architecture" in best_config:
            arch = best_config["architecture"]
            print(f"  Architecture: {len(arch)} layers")
            for layer in arch:
                if "input_size" in layer and "output_size" in layer:
                    print(
                        f"    {layer['layer_type']}: {layer['input_size']} → {layer['output_size']}"
                    )

    # Create classifier using the standardized function
    classifier = create_classifier_from_config(
        best_config, classifier_input_size, num_classes
    )

    # Create the full model
    model = BertClassifier(triplet_model, classifier).to(device)

    # Load model weights with proper error handling
    checkpoint_path = str(Path(CHECKPOINT_DIR) / "model" / "best_model.pt")

    try:
        # Use enhanced error handling if available
        from FTC_utils.file_utils import load_checkpoint_safely

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        print(f"Loading model weights from {checkpoint_path}")
        checkpoint = load_checkpoint_safely(model, checkpoint_path, map_location=device)
        print("Model loaded successfully!")

    except ImportError:
        # Fallback to basic error handling
        if not os.path.exists(checkpoint_path):
            print(f"ERROR: Checkpoint not found at {checkpoint_path}")
            print(f"Expected path: {checkpoint_path}")
            print("Please ensure you've trained a model first.")
            return

        print(f"Loading model weights from {checkpoint_path}")
        try:
            # Load with warning suppression for weights_only
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
                checkpoint = torch.load(checkpoint_path, map_location=device)

            try:
                if "model_state_dict" in checkpoint:
                    model.load_state_dict(checkpoint["model_state_dict"])
                else:
                    # Try loading directly if checkpoint is just the state dict
                    model.load_state_dict(checkpoint)
                print("Model loaded successfully!")
            except RuntimeError as e:
                # Provide detailed error message for model mismatch
                if "size mismatch" in str(e) or "shape mismatch" in str(e):
                    print(
                        f"ERROR: Model architecture mismatch when loading checkpoint."
                    )
                    print(
                        "The model structure doesn't match the saved checkpoint structure."
                    )
                    print(f"Original error: {e}")
                else:
                    print(f"ERROR: Failed to load model weights: {e}")

                print(
                    "\nArchitecture mismatch between saved weights and configuration."
                )
                print(
                    "This means the best_config.yml file does not match the architecture used to train the model."
                )
                print("Options to resolve this:")
                print(
                    "1. Run TuneBert.py again to create a new configuration with proper architecture details"
                )
                print(
                    "2. Run BertClassification.py with the same configuration used during training"
                )
                return
        except Exception as e:
            print(f"ERROR: Failed to load checkpoint file: {e}")
            print("Please ensure the checkpoint file is not corrupted.")
            return

    # Set model to evaluation mode
    model.eval()

    # Initialize conformal predictor
    conformal_predictor = ConformalPredictor(significance=args.significance)

    # Use CALIBRATION_PATH from config
    calibration_file = config["CALIBRATION_PATH"]

    try:
        conformal_predictor.load_calibration(calibration_file)
    except FileNotFoundError:
        print("No existing calibration found. Performing new calibration...")
        # Tokenize training texts for calibration
        train_texts = train_df["text"].apply(preprocess_text).tolist()
        train_encoded = tokenizer(
            train_texts,
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Create calibration dataset with correct parameters
        calibration_dataset = TextDataset(
            texts=train_texts,
            labels=label_encoder.transform(train_df["labels"]).tolist(),
            tokenizer=tokenizer,
            max_length=MAX_SEQ_LEN,
            device=device,
        )
        calibration_loader = DataLoader(
            calibration_dataset, batch_size=32, shuffle=True
        )

        # Calibrate the conformal predictor
        print("Calibrating conformal predictor...")
        conformal_predictor.calibrate(model, calibration_loader, device)

        # Save calibration for future use
        os.makedirs(os.path.dirname(calibration_file), exist_ok=True)
        conformal_predictor.save_calibration(calibration_file)
        print(f"Saved calibration to {calibration_file}")

    # Explanation functionality removed

    # Process large files in chunks if --large-file flag is set
    if args.large_file:
        print(f"Processing large file in chunks of {args.chunk_size} rows...")
        predict_large_file(
            model=model,
            tokenizer=tokenizer,
            input_file=INPUT_FILE,
            output_file=OUTPUT_FILE,
            label_encoder=label_encoder,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        print(f"Large file processing complete. Results saved to {OUTPUT_FILE}")
        return

    # Standard processing for smaller files
    df_test = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df_test)} samples from {INPUT_FILE}")

    # Make predictions
    print("Making predictions...")
    # Using tqdm imported at the top

    results = predict_with_confidence(
        model=model,
        tokenizer=tokenizer,
        texts=df_test["text"].tolist(),
        label_encoder=label_encoder,
        conformal_predictor=conformal_predictor,
        device=device,
        batch_size=args.batch_size,
        max_seq_len=MAX_SEQ_LEN,
    )

    # Add predictions to dataframe
    df_test["predicted_label"] = results["predicted_label"]
    df_test["confidence_score"] = results["confidence_score"]
    df_test["prediction_set"] = results["prediction_set"]
    df_test["prediction_set_size"] = [len(ps) for ps in results["prediction_set"]]

    # Save results
    df_test.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

    # Just show a minimal completion message
    print(f"\nPredictions complete. Results saved to {OUTPUT_FILE}")

    print(f"\nExecution log saved to: {log_file}")

    return df_test


if __name__ == "__main__":
    main()
