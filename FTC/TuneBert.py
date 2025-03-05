#!/usr/bin/env python3
# Standard library imports
import os
import sys
import argparse
import datetime
from pathlib import Path

# Add the parent directory (project root) to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Third-party imports
import yaml
import optuna
import torch.nn as nn
import torch.optim as optim

# Local imports
from FTC_utils.logging_utils import tee_to_file
from FTC_utils.LoaderSetup import join_constructor, env_var_or_default_constructor
from FTC_utils.env_utils import check_environment
from FTC_utils.device_utils import get_device
from FTC_utils.shared import (
    load_embedding_data,
    validate,
    STUDY_NAME,
    BEST_CONFIG_DIR,
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
)

# Start capturing output to log file
log_file = tee_to_file("TuneBert")

# Register YAML constructors
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
yaml.add_constructor(
    "!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader
)

# Check environment and create necessary directories
check_environment()

# Default constants (can be overridden by CLI arguments)
DEFAULT_NUM_TRIALS = 20  # Default number of hyperparameter combinations to try
DEFAULT_EPOCHS = (
    5  # Default number of epochs for each trial (keep low for faster experimentation)
)


# Updated DenseBlock with additional activations.
class DenseBlock(nn.Module):
    """A neural network block combining linear layer, batch normalization, activation, and dropout.

    This block implements a standard dense layer with normalization and regularization.

    Args:
        input_size (int): The dimensionality of the input features
        output_size (int): The dimensionality of the output features
        dropout_rate (float): The probability of zeroing a neuron during dropout
        activation_fn (str, optional): The name of the activation function to use
    """

    def __init__(self, input_size, output_size, dropout_rate, activation_fn=None):
        super(DenseBlock, self).__init__()
        activation = self._get_activation(activation_fn)

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.BatchNorm1d(output_size),
            activation,
            nn.Dropout(dropout_rate),
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        # Fix: Normalize the activation string to match dictionary keys.
        if not isinstance(name, str):
            return nn.GELU()

        name_lower = name.lower()
        if name_lower == "leakyrelu":
            name_lower = "leaky_relu"
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
        }
        if name_lower not in activations:
            return nn.GELU()
        return activations[name_lower]

    def forward(self, x):
        """Forward pass through the dense block.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output after applying linear layer, batch norm, activation, and dropout
        """
        return self.block(x)


# Tunable classifier architecture using a variable number of DenseBlock layers.
class TunableFFModel(nn.Module):
    """A flexible feed-forward model with tunable architecture designed for Optuna optimization.

    This model creates a variable depth neural network based on parameters suggested by an Optuna trial.
    The network consists of multiple DenseBlock layers and a final linear output layer.

    Args:
        input_dim (int): Dimensionality of the input features
        output_dim (int): Number of output classes
        trial (optuna.Trial): Optuna trial object that suggests hyperparameters
    """

    def __init__(self, input_dim, output_dim, trial):
        super(TunableFFModel, self).__init__()
        # Now include additional activation options.
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        activation_fn = trial.suggest_categorical(
            "activation", ["gelu", "relu", "leaky_relu", "silu", "elu"]
        )
        layers = []
        prev_width = input_dim
        # Tune number of hidden layers (depth)
        n_hidden = trial.suggest_int("n_hidden", 1, 3)
        for i in range(n_hidden):
            low = max(32, prev_width // 4)
            width = trial.suggest_int(f"hidden_units_{i}", low, prev_width)
            layers.append(DenseBlock(prev_width, width, dropout_rate, activation_fn))
            prev_width = width
        layers.append(nn.Linear(prev_width, output_dim))
        self.model = nn.Sequential(*layers)
        self.initialize_weights()

    def forward(self, x):
        """Forward pass through the feed-forward model.

        Args:
            x (torch.Tensor): Input tensor containing embeddings

        Returns:
            torch.Tensor: Model output logits
        """
        return self.model(x)

    def initialize_weights(self):
        """Initialize model weights using Xavier uniform initialization.

        Applies Xavier uniform initialization to all linear layers in the model
        and zeros to all bias terms.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# BERT classifier that uses the tunable FFModel as its classifier.
class BERTClassifierTuned(nn.Module):
    """A model that uses pre-computed BERT embeddings as input to a tunable classifier.

    This model takes pre-computed embeddings as input rather than raw text,
    making it more efficient for hyperparameter tuning.

    Args:
        classifier (nn.Module): The classification model to process embeddings
    """

    def __init__(self, classifier):
        super(BERTClassifierTuned, self).__init__()
        self.classifier = classifier

    def forward(self, inputs):
        """Forward pass for the BERT classifier.

        Args:
            inputs (torch.Tensor): Tensor of embedding vectors

        Returns:
            torch.Tensor: Output logits from the classifier
        """
        # Inputs are already embeddings
        return self.classifier(inputs)


def objective(trial, device, epochs=None, total_trials=None, large_dataset=False):
    """Objective function for Optuna optimization using F1 score as the target metric

    Args:
        trial: Optuna trial object
        device: Device to use for computation
        epochs: Number of epochs to train for (overrides default)
        total_trials: Total number of trials for display purposes
        large_dataset: Whether to use chunked loading for large datasets
        
    Returns:
        float: The validation F1 score achieved by the model
    """
    # Use default values if not provided
    if epochs is None:
        epochs = DEFAULT_EPOCHS
    if total_trials is None:
        total_trials = DEFAULT_NUM_TRIALS

    # Load embedding data - will regenerate if needed
    # Note: load_test=False to avoid unnecessary data loading and computation
    train_loader, val_loader, _, _, num_label = load_embedding_data(
        device, load_test=False, large_dataset=large_dataset, verbose=False
    )

    # Get input dimension from data and num classes
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    output_dim = num_label  # Use the returned num_label from load_embedding_data

    # Build classifier using the tunable design.
    classifier = TunableFFModel(input_dim, output_dim, trial)
    model = BERTClassifierTuned(classifier).to(device)

    # Tune optimizer choice: 'adamw', 'sgd', or 'rmsprop'
    optimizer_choice = trial.suggest_categorical(
        "optimizer", ["adamw", "sgd", "rmsprop"]
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    if optimizer_choice == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "sgd":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        nesterov = trial.suggest_categorical("nesterov", [True, False])
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif optimizer_choice == "rmsprop":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        alpha = trial.suggest_float("alpha", 0.9, 0.99)
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=alpha,
        )
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = nn.CrossEntropyLoss()

    # We've already set n_epochs at the beginning of the function
    n_epochs = epochs
    best_val_acc = 0  # Keep for backward compatibility
    best_val_f1 = 0   # Primary metric for optimization

    # Print trial information with 1-based indexing
    print(f"Trial {trial.number + 1}/{total_trials}: Testing hyperparameters...")

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Evaluation phase - use F1 score as primary metric 
        _, val_acc, val_f1 = validate(model, val_loader, criterion)
        best_val_acc = max(best_val_acc, val_acc)  # Keep for reporting
        best_val_f1 = max(best_val_f1, val_f1)     # Use for optimization

        # Report intermediate F1 score to Optuna without printing
        trial.report(val_f1, epoch)

        # Handle pruning (early stopping of unpromising trials)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Print the final result with 1-based indexing
    print(
        f"Trial {trial.number + 1}/{total_trials}: Complete - This trial's best F1: {best_val_f1:.4f} (accuracy: {best_val_acc:.4f})"
    )

    # Return F1 score as the primary optimization metric
    return best_val_f1


def get_arg_parser():
    """Create and return an argument parser for TuneBert.py"""
    parser = argparse.ArgumentParser(
        description="Optimize classifier architecture using Optuna"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=f"Number of Optuna trials to run (default: {DEFAULT_NUM_TRIALS})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of epochs per trial (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=STUDY_NAME,
        help=f"Name of the Optuna study (default: {STUDY_NAME})",
    )
    parser.add_argument(
        "--pruning",
        action="store_true",
        help="Enable pruning of unpromising trials (early stopping)",
    )
    parser.add_argument(
        "--large-dataset",
        action="store_true",
        help="Use chunked loading for large datasets to reduce memory usage",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override the output directory for saving the best configuration",
    )
    return parser


def main():
    """Main entry point for hyperparameter tuning.

    This function:
    1. Parses command line arguments
    2. Sets up the device (CPU/GPU)
    3. Creates an Optuna study to optimize hyperparameters
    4. Runs multiple trials with different hyperparameter configurations
    5. Saves the best configuration to a YAML file

    Returns:
        optuna.Study: The completed Optuna study object, or None if an error occurred
    """
    # Parse command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()

    # Set variables from arguments
    num_trials = args.trials
    epochs_per_trial = args.epochs
    study_name = args.study_name
    large_dataset = args.large_dataset

    # Initialize device and create directories
    device = get_device()  # Use the imported function
    print(f"Using device: {device}")

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(BEST_CONFIG_DIR, exist_ok=True)
        output_dir = BEST_CONFIG_DIR

    # Check if TripletTraining.py has been run first to prevent data leakage
    if not all(os.path.exists(path) for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]):
        print("ERROR: Data splits not found. Please run TripletTraining.py first.")
        print(
            "This is required to prevent data leakage between embedding training and tuning."
        )
        print("Run: python FTC/TripletTraining.py --epochs 10 --lr 2e-5")
        sys.exit(1)

    print("Starting hyperparameter tuning...")
    print(f"Will run {num_trials} trials using device: {device}")
    print(f"Using {epochs_per_trial} epochs per trial")
    print(
        "Using consistent train/val/test splits from TripletTraining.py to prevent data leakage."
    )
    print("Note: Test data is intentionally not loaded for hyperparameter tuning")
    print("This saves computation and prevents overfitting to the test set")

    try:
        # Create and run the study
        study = optuna.create_study(direction="maximize", study_name=study_name)

        # Track the global best F1 score across all trials
        global_best_f1 = 0.0

        # Define objective function with all necessary parameters
        def trial_objective(trial):
            """Objective function for an individual Optuna trial.

            Args:
                trial (optuna.Trial): The Optuna trial object

            Returns:
                float: The validation F1 score achieved by the model with the trial's hyperparameters
            """
            nonlocal global_best_f1
            trial_f1 = objective(
                trial,
                device=device,
                epochs=epochs_per_trial,
                total_trials=num_trials,
                large_dataset=large_dataset,
            )

            # Update and display the global best F1 score
            if trial_f1 > global_best_f1:
                global_best_f1 = trial_f1
                print(
                    f"New best F1 score across all trials: {global_best_f1:.4f}"
                )

            return trial_f1

        # Run optimization with the specified number of trials
        study.optimize(trial_objective, n_trials=num_trials)

        # Print and save results
        print("\n" + "=" * 50)
        print("HYPERPARAMETER OPTIMIZATION COMPLETED")
        print("=" * 50)
        print(f"Best trial: #{study.best_trial.number + 1}")
        trial = study.best_trial
        print(f"Global best validation F1 score: {trial.value:.4f}")

        print("\nBest hyperparameters: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Get the first dataloader batch to determine input dimension and output dimension
        # Still don't need test data here
        train_loader, _, _, _, num_classes = load_embedding_data(
            device, load_test=False, verbose=False
        )
        sample_batch, _ = next(iter(train_loader))
        input_dim = sample_batch.shape[1]

        # Create a complete configuration dictionary with EXPLICIT architecture details
        complete_config = {
            # Basic metadata about the configuration
            "model_type": "sequential",  # Explicitly store the model type
            "model_version": "1.0",  # Version tracking for backward compatibility
            "creation_timestamp": datetime.datetime.now().isoformat(),
            "study_name": study_name,
            "validation_accuracy": float(trial.value),
            # Copy all hyperparameters from the trial
            **trial.params,
            # Explicitly add dimensions that are inferred during model creation
            "input_dim": int(input_dim),
            "output_dim": int(num_classes),
        }

        # Import MAX_SEQ_LEN from config - FIX: Add this line to fix the error
        with open("config.yml", "r", encoding="utf-8") as f:
            config_yaml = yaml.safe_load(f)
            MAX_SEQ_LEN = config_yaml.get(
                "MAX_SEQ_LEN", 64
            )  # Default to 64 if not found

        # Now add MAX_SEQ_LEN to the complete config
        complete_config["max_seq_len"] = MAX_SEQ_LEN

        # If using a tuned architecture with variable hidden layers, ensure all layer dims are specified
        if "n_hidden" in complete_config:
            n_hidden = complete_config["n_hidden"]

            # Build a complete description of each layer's dimensions and parameters
            architecture = []
            prev_width = input_dim

            for i in range(n_hidden):
                width_key = f"hidden_units_{i}"
                # If the width wasn't part of the trial params, add it with clear defaults
                if width_key not in complete_config:
                    # Use a deterministic rule for layer width (half previous width)
                    layer_width = prev_width // 2
                    complete_config[width_key] = layer_width
                    print(
                        f"Adding missing layer dimension: {width_key} = {layer_width}"
                    )
                else:
                    layer_width = complete_config[width_key]

                # Add detailed layer descriptor to architecture
                layer = {
                    "layer_index": i,
                    "layer_type": "dense_block",
                    "input_size": prev_width,
                    "output_size": layer_width,
                    "dropout_rate": complete_config.get("dropout_rate", 0.3),
                    "activation": complete_config.get("activation", "gelu"),
                    "batch_norm": True,
                }
                architecture.append(layer)
                prev_width = layer_width

            # Add final output layer
            architecture.append(
                {
                    "layer_index": n_hidden,
                    "layer_type": "linear",
                    "input_size": prev_width,
                    "output_size": num_classes,
                }
            )

            # Store the complete architecture in the config
            complete_config["architecture"] = architecture

            # Also add a verification hash to ensure this architecture hasn't been tampered with
            import hashlib

            arch_str = str(architecture)
            complete_config["architecture_hash"] = hashlib.md5(
                arch_str.encode()
            ).hexdigest()

        # Save the complete configuration to a YAML file using pathlib for safe path handling
        best_config_file = str(Path(output_dir) / f"best_config_{study_name}.yml")

        with open(best_config_file, "w", encoding="utf-8") as f:
            yaml.dump(complete_config, f, default_flow_style=False, sort_keys=False)

        print(f"\nSaved complete configuration to {best_config_file}")
        print("Configuration includes:")

        # Print a subset of the most important configuration parameters
        for key in ["model_type", "n_hidden", "input_dim", "output_dim", "activation"]:
            if key in complete_config:
                print(f"  {key}: {complete_config[key]}")

        print(f"  Architecture: {len(complete_config.get('architecture', []))} layers")
        print("  Layer dimensions:", end=" ")
        for i in range(complete_config.get("n_hidden", 0)):
            width_key = f"hidden_units_{i}"
            if width_key in complete_config:
                print(f"{complete_config[width_key]}", end=" → ")
        print(f"{complete_config['output_dim']}")

        print(f"\nExecution log saved to: {log_file}")
        return study
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback

        traceback.print_exc()  # Print full stack trace for better debugging
        print(f"\nExecution log with error details saved to: {log_file}")
        return None


if __name__ == "__main__":
    main()
