import os
import torch
import optuna
import torch.nn as nn
import torch.optim as optim
import yaml

from utils.LoaderSetup import join_constructor

# Register the YAML constructor before importing shared which uses it
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)

from utils.shared import (
    load_embedding_data,
    validate,
    STUDY_NAME,
    BEST_CONFIG_DIR,
)

# Import the get_device function from our utility module
from utils.device_utils import get_device

# Constants
NUM_TRIALS = 20  # Number of hyperparameter combinations to try
EPOCHS = 5  # Number of epochs for each trial (keep low for faster experimentation)


# Updated DenseBlock with additional activations.
class DenseBlock(nn.Module):
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
        return self.block(x)


# Tunable classifier architecture using a variable number of DenseBlock layers.
class TunableFFModel(nn.Module):
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
        return self.model(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# BERT classifier that uses the tunable FFModel as its classifier.
class BERTClassifierTuned(nn.Module):
    def __init__(self, classifier):
        super(BERTClassifierTuned, self).__init__()
        self.classifier = classifier

    def forward(self, inputs):
        # Inputs are already embeddings
        return self.classifier(inputs)


def objective(trial, device):
    """Objective function for Optuna optimization"""
    # Load embedding data - will regenerate if needed
    train_loader, val_loader, test_loader, label_encoder, num_label = (
        load_embedding_data(device)
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

    n_epochs = EPOCHS
    best_val_acc = 0

    # Print trial information with 1-based indexing
    print(f"Trial {trial.number + 1}/{NUM_TRIALS}: Testing hyperparameters...")

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

        # Evaluation phase
        _, val_acc, _ = validate(model, val_loader, criterion)
        best_val_acc = max(best_val_acc, val_acc)

        # Report intermediate objective value to Optuna without printing
        trial.report(val_acc, epoch)

        # Handle pruning (early stopping of unpromising trials)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Print the final result with 1-based indexing
    print(
        f"Trial {trial.number + 1}/{NUM_TRIALS}: Complete - Best validation accuracy: {best_val_acc:.4f}"
    )

    return best_val_acc


def main():
    """Main entry point for hyperparameter tuning"""
    # Initialize device and create directories
    device = get_device()  # Use the imported function
    print(f"Using device: {device}")
    os.makedirs(BEST_CONFIG_DIR, exist_ok=True)

    print("Starting hyperparameter tuning...")
    print(f"Will run {NUM_TRIALS} trials using device: {device}")

    try:
        # Create and run the study
        study = optuna.create_study(direction="maximize", study_name=STUDY_NAME)
        study.optimize(lambda trial: objective(trial, device), n_trials=NUM_TRIALS)

        # Print and save results
        print("\nBest trial:")
        trial = study.best_trial
        print(f"Validation Accuracy: {trial.value:.4f}")
        print("Best hyperparameters: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Get the first dataloader batch to determine input dimension and output dimension
        train_loader, _, _, label_encoder, num_classes = load_embedding_data(device)
        sample_batch, _ = next(iter(train_loader))
        input_dim = sample_batch.shape[1]

        # Create a complete configuration dictionary with EXPLICIT architecture details
        complete_config = {
            # Basic metadata about the configuration
            "model_type": "sequential",  # Explicitly store the model type
            "model_version": "1.0",  # Version tracking for backward compatibility
            "creation_timestamp": datetime.datetime.now().isoformat(),
            "study_name": STUDY_NAME,
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

        # Save the complete configuration to a YAML file
        best_config_file = os.path.join(
            BEST_CONFIG_DIR, f"best_config_{STUDY_NAME}.yml"
        )

        with open(best_config_file, "w", encoding="utf-8") as f:
            yaml.dump(complete_config, f, default_flow_style=False, sort_keys=False)

        print(f"\nSaved complete configuration to {best_config_file}")
        print("Configuration includes:")

        # Print a subset of the most important configuration parameters
        for key in ["model_type", "n_hidden", "input_dim", "output_dim", "activation"]:
            if key in complete_config:
                print(f"  {key}: {complete_config[key]}")

        print(f"  Architecture: {len(complete_config.get('architecture', []))} layers")
        print(f"  Layer dimensions:", end=" ")
        for i in range(complete_config.get("n_hidden", 0)):
            width_key = f"hidden_units_{i}"
            if width_key in complete_config:
                print(f"{complete_config[width_key]}", end=" → ")
        print(f"{complete_config['output_dim']}")

        return study
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback

        traceback.print_exc()  # Print full stack trace for better debugging
        return None


# Add the import needed for timestamp
import datetime

if __name__ == "__main__":
    main()
