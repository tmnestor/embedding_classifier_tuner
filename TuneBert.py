import os

import optuna
import torch.nn as nn
import torch.optim as optim

# from transformers import AutoModel, AutoTokenizer
import yaml

from utils.shared import (
    generate_embedding_csv_data,
    load_embedding_data,
    load_triplet_model,
    validate,
)
from utils.utils import get_default_device

# Load configuration
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

##################################################
STUDY_NAME = config["STUDY_NAME"]
N_TRIALS = 50
MODEL = "paraphrase-MiniLM-L6-v2"  # or change as needed
MODEL_NAME = config["MODEL_NAME"]
DATA_PATH = config["DATA_PATH"]
CSV_PATH = config["CSV_PATH"]
TRAIN_CSV = config["TRAIN_CSV"]
VAL_CSV = config["VAL_CSV"]
TEST_CSV = config["TEST_CSV"]
MAX_SEQ_LEN = config["MAX_SEQ_LEN"]
BATCH_SIZE = config["BATCH_SIZE"]
SEED = config["SEED"]
##################################################


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
            "mish": nn.Mish(),  # Added Mish activation.
        }
        if name_lower not in activations:
            print("Unknown activation '%s', using GELU", name)
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
            "activation", ["GELU", "ReLU", "LeakyReLU", "SiLU", "Mish", "PReLU", "ELU"]
        )
        layers = []
        prev_width = input_dim
        # Tune number of hidden layers (depth)
        n_hidden = trial.suggest_int("n_hidden", 1, 4)
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
    def __init__(self, base_model, classifier):
        super(BERTClassifierTuned, self).__init__()
        self.base_model = base_model
        self.classifier = classifier

    def forward(self, inputs):
        normalized_output = inputs
        logits = self.classifier(normalized_output)
        return logits


def objective(trial):
    DEVICE = get_default_device()

    # Load the TripletEmbeddingModel

    triplet_model = load_triplet_model(DEVICE)

    # Load embedding data
    train_loader, val_loader, _, label_encoder = load_embedding_data(DEVICE)

    input_dim = triplet_model.base_model.config.hidden_size
    output_dim = len(label_encoder.classes_)

    # Build classifier using the tunable design.
    classifier = TunableFFModel(input_dim, output_dim, trial)
    model = BERTClassifierTuned(classifier, classifier).to(DEVICE)

    # Tune optimizer choice: 'adamw', 'sgd', or 'rmsprop'
    optimizer_choice = trial.suggest_categorical(
        "optimizer", ["adamw", "sgd", "rmsprop"]
    )
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    lr = 2e-5  # Keep a fixed learning rate for simplicity

    if optimizer_choice == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "sgd":
        momentum = trial.suggest_float("momentum", 0.0, 1.0)
        nesterov = trial.suggest_categorical("nesterov", [True, False])
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif optimizer_choice == "rmsprop":
        momentum = trial.suggest_float("momentum", 0.0, 1.0)
        alpha = trial.suggest_float("alpha", 0.85, 0.99)
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

    n_epochs = 10  # Use few epochs for tuning
    best_val_acc = 0
    for epoch in range(n_epochs):
        model.train()
        for batch_data, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        # Evaluate on validation set.
        _, val_acc, _ = validate(model, val_loader, criterion)
        best_val_acc = max(best_val_acc, val_acc)
    return best_val_acc


if __name__ == "__main__":
    DEVICE = get_default_device()
    # Load the TripletEmbeddingModel
    triplet_model = load_triplet_model(DEVICE)

    # Generate embedding data
    generate_embedding_csv_data(triplet_model, DEVICE)

    study = optuna.create_study(direction="maximize", study_name=STUDY_NAME)
    study.optimize(objective, n_trials=N_TRIALS)
    print("Best trial:")
    trial = study.best_trial
    print(f"Validation Accuracy: {trial.value:.4f}")
    print("Best hyperparameters: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    # Save the best configuration to a YAML file in BEST_CONFIG_DIR.
    os.makedirs(config["BEST_CONFIG_DIR"], exist_ok=True)  # Updated key name
    best_config_file = os.path.join(
        config["BEST_CONFIG_DIR"], f"best_config_{STUDY_NAME}.yml"
    )
    with open(best_config_file, "w", encoding="utf-8") as f:
        yaml.dump(trial.params, f)
    print(f"Saved best configuration to {best_config_file}")
