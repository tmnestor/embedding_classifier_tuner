#!/usr/bin/env python3
# Standard library imports
import ast
import os
import random
import re
import sys
import argparse
from pathlib import Path

# Add the parent directory (project root) to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Third-party imports
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Local imports
from FTC.TripletTraining import TripletEmbeddingModel
from FTC_utils.LoaderSetup import join_constructor, env_var_or_default_constructor
from FTC_utils.env_utils import check_environment

# Register the YAML constructors
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
yaml.add_constructor(
    "!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader
)

# Check environment and create necessary directories
check_environment()

# More local imports
from FTC_utils.device_utils import get_device
from FTC_utils.logging_utils import tee_to_file
from FTC_utils.shared import validate

# Start capturing output to log file
log_file = tee_to_file("BertClassification")

# Verify the file structure and print debug info
utils_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "FTC_utils"
)
print(f"Utils directory exists: {os.path.exists(utils_dir)}")
print(
    f"Files in FTC_utils: {os.listdir(utils_dir) if os.path.exists(utils_dir) else 'N/A'}"
)

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
        sys.exit(1)
SEED = config["SEED"]

# Initialize global seed
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# globals from config lookups:
NUM_EPOCHS = config["NUM_EPOCHS"]
BATCH_SIZE = config["BATCH_SIZE"]
MAX_SEQ_LEN = config["MAX_SEQ_LEN"]
MODEL_NAME = config["MODEL_NAME"]
MODEL_PATH = config["MODEL_PATH"]
CSV_PATH = config["CSV_PATH"]
TRAIN_CSV = config["TRAIN_CSV"]
VAL_CSV = config["VAL_CSV"]
TEST_CSV = config["TEST_CSV"]
DROP_OUT = config["DROP_OUT"]
ACT_FN = config["ACT_FN"]
LEARNING_RATE = config["LEARNING_RATE"]
PATIENCE = config["PATIENCE"]
LR_PATIENCE = config["LR_PATIENCE"]
MIN_LR = config["MIN_LR"]
CHECKPOINT_DIR = config["CHECKPOINT_DIR"]
EMBEDDING_PATH = config["EMBEDDING_PATH"]
STUDY_NAME = config["STUDY_NAME"]

TAG_RE = re.compile(r"<[^>]+>")

#########################################################


def load_csv_data(device, tokenizer):
    """
    Load and prepare data from CSV for training without precomputing embeddings.
    Uses pre-created splits from TripletTraining.py to prevent data leakage.
    
    Args:
        device: Torch device to use
        tokenizer: Tokenizer to use for text processing
        
    Returns:
        train_loader, val_loader, test_loader, label_encoder, num_classes
    """
    # Load the pre-created splits from TripletTraining.py
    # This ensures consistency and prevents data leakage
    try:
        print(f"Loading pre-created train/val/test splits...")
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VAL_CSV) 
        df_test = pd.read_csv(TEST_CSV)
        print(f"Successfully loaded pre-created splits")
    except FileNotFoundError:
        print("ERROR: Pre-created splits not found. Please run TripletTraining.py first to create the splits.")
        print("This is required to prevent data leakage between embedding training and classification.")
        print("Run: python FTC/TripletTraining.py --epochs 10 --lr 2e-5")
        sys.exit(1)
    
    # Apply text preprocessing
    df_train["text"] = df_train["text"].apply(preprocess_text)
    df_val["text"] = df_val["text"].apply(preprocess_text)
    df_test["text"] = df_test["text"].apply(preprocess_text)

    print(
        f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}"
    )

    # Fit LabelEncoder on training labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["labels"])
    y_val = label_encoder.transform(df_val["labels"])
    y_test = label_encoder.transform(df_test["labels"])

    # Get number of classes
    num_classes = len(label_encoder.classes_)
    print(f"Detected {num_classes} classes: {label_encoder.classes_}")

    # Create TextDataset objects instead of pre-computing embeddings
    train_ds = TextDataset(
        texts=df_train["text"].tolist(),
        labels=y_train,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
        device=device,
    )

    val_ds = TextDataset(
        texts=df_val["text"].tolist(),
        labels=y_val,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
        device=device,
    )

    test_ds = TextDataset(
        texts=df_test["text"].tolist(),
        labels=y_test,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LEN,
        device=device,
    )

    # Create data loaders
    # Note: We will use the batch_size variable if available (command line override)
    # or fall back to BATCH_SIZE global when the function is called directly
    # Use the global BATCH_SIZE as we don't have access to args here
    data_batch_size = BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=data_batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=data_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=data_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, label_encoder, num_classes


def save_checkpoint(model, optimizer, epoch, val_acc, best_val_acc, val_f1, best_val_f1, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
        "val_f1": val_f1,
        "best_val_f1": best_val_f1,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer):
    # Load checkpoint with warning suppression for weights_only
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
        checkpoint = torch.load(filename)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    # Handle both new F1-based checkpoints and old accuracy-based checkpoints for backwards compatibility
    epoch = checkpoint["epoch"]
    val_acc = checkpoint["val_acc"]
    best_val_acc = checkpoint["best_val_acc"]
    
    # Get F1 values if available, otherwise default to 0
    val_f1 = checkpoint.get("val_f1", 0.0)
    best_val_f1 = checkpoint.get("best_val_f1", 0.0)
    
    return epoch, val_acc, best_val_acc, val_f1, best_val_f1


def evaluate_model(model, test_loader, criterion):
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion)
    print("\nTest Set Evaluation:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    return test_loss, test_acc, test_f1


class TextDataset(Dataset):
    """Dataset for handling raw text data for classification.

    This dataset handles raw text directly, without pre-computing embeddings.
    It tokenizes text on-the-fly when items are requested.
    """

    def __init__(self, texts, labels, tokenizer, max_length=128, device=None):
        """
        Initialize the dataset.

        Args:
            texts: List of text strings
            labels: List of integer labels
            tokenizer: HuggingFace tokenizer for text encoding
            max_length: Maximum sequence length for tokenization
            device: Torch device to use
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device if device is not None else torch.device("cpu")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """Get tokenized text and label for a single item."""
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension and move to device
        encoding = {k: v.squeeze(0).to(self.device) for k, v in encoding.items()}

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long, device=self.device)

        return encoding, label_tensor


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
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
        }
        if name.lower() not in activations:
            # print("Unknown activation '%s', using GELU", name)
            return nn.GELU()
        return activations[name.lower()]

    def forward(self, x):
        return self.block(x)


class FFModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate, activation_fn=None):
        super(FFModel, self).__init__()
        self.block1 = DenseBlock(input_dim, input_dim // 2, dropout_rate, activation_fn)
        self.block2 = DenseBlock(
            input_dim // 2, input_dim // 4, dropout_rate, activation_fn
        )
        self.final_layer = nn.Linear(input_dim // 4, output_dim)
        self._initialize_weights()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_layer(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class BertClassifier(nn.Module):
    """
    End-to-end BERT classifier that processes raw text directly.
    Uses a trained TripletEmbeddingModel for the embedding part and a classifier on top.
    """

    def __init__(self, triplet_model, classifier):
        super(BertClassifier, self).__init__()
        self.triplet_model = triplet_model
        self.classifier = classifier

        # Freeze the triplet model weights by default
        for param in self.triplet_model.parameters():
            param.requires_grad = False

        # Only classifier weights will be updated during training

    def forward(self, inputs):
        """
        Forward pass that handles tokenized inputs and produces classification logits.

        Args:
            inputs: Dictionary containing 'input_ids' and 'attention_mask' from tokenizer

        Returns:
            logits: Classification logits
        """
        # Get embeddings from triplet model
        embeddings = self.triplet_model(inputs["input_ids"], inputs["attention_mask"])

        # Pass embeddings through the classifier
        logits = self.classifier(embeddings)
        return logits

    def unfreeze_bert_layers(self, num_layers=0):
        """
        Optionally unfreeze some layers of the BERT model for fine-tuning.

        Args:
            num_layers: Number of layers to unfreeze from the top.
                        0 means all layers stay frozen (default)
                        -1 means unfreeze all layers
        """
        if num_layers == 0:
            # Keep all layers frozen (default)
            return

        # Unfreeze embedding layer if requested to unfreeze all
        if num_layers == -1:
            for param in self.triplet_model.base_model.embeddings.parameters():
                param.requires_grad = True

        # Unfreeze the requested number of encoder layers from the top
        encoder_layers = list(self.triplet_model.base_model.encoder.layer)
        num_encoder_layers = len(encoder_layers)

        if num_layers == -1:
            # Unfreeze all layers
            layers_to_unfreeze = range(num_encoder_layers)
        else:
            # Unfreeze only the last num_layers
            layers_to_unfreeze = range(
                max(0, num_encoder_layers - num_layers), num_encoder_layers
            )

        for layer_idx in layers_to_unfreeze:
            for param in encoder_layers[layer_idx].parameters():
                param.requires_grad = True


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch_data, batch_labels in progress_bar:
        optimizer.zero_grad()
        if isinstance(batch_data, tuple):
            input_ids, attention_mask = batch_data[0], batch_data[1]
            outputs = model(input_ids, attention_mask)
        else:
            outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        all_preds.append(preds.detach().cpu().numpy())
        all_labels.append(batch_labels.detach().cpu().numpy())
        progress_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    # Use the same approach as the standardized validate function
    from sklearn.metrics import f1_score

    # Calculate both macro and weighted F1
    train_f1_macro = f1_score(all_labels, all_preds, average="macro")
    train_f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    # Use weighted F1 if there's significant class imbalance
    if abs(train_f1_macro - train_f1_weighted) > 0.05:
        print(
            f"Note: Using weighted F1 due to class imbalance (macro: {train_f1_macro:.4f}, weighted: {train_f1_weighted:.4f})"
        )
        train_f1 = train_f1_weighted
    else:
        train_f1 = train_f1_macro

    return total_loss / len(dataloader), train_f1


# We already imported validate from FTC_utils.shared above


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler, num_epochs=None, patience=None):
    # Use passed parameters or fall back to globals
    if num_epochs is None:
        num_epochs = NUM_EPOCHS
    if patience is None:
        patience = PATIENCE
        
    train_metrics = []
    best_val_acc = 0  # Keep for backwards compatibility
    best_val_f1 = 0   # Primary metric for model selection
    no_improve_count = 0

    for epoch in range(num_epochs):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion)
        
        # Get current learning rate before update
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Use F1 score instead of accuracy for scheduler
        scheduler.step(val_f1)
        
        # Check if learning rate changed
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr != current_lr:
            print(f"Learning rate changed: {current_lr:.2e} → {new_lr:.2e}")
            
        current_lr = new_lr
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_acc": val_acc,
            "lr": current_lr,
        }
        train_metrics.append(epoch_metrics)
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"Training Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(
            f"Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%, Val F1: {val_f1:.4f}"
        )
        # Save model based on F1 score rather than accuracy
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = max(best_val_acc, val_acc)  # Keep this updated for backwards compatibility
            no_improve_count = 0
            # Ensure model subdirectory exists
            model_dir = os.path.join(CHECKPOINT_DIR, "model")
            os.makedirs(model_dir, exist_ok=True)
            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_acc,
                best_val_acc,
                val_f1,
                best_val_f1,
                os.path.join(
                    model_dir, "best_model.pt"
                ),  # Updated to save in checkpoints/model
            )
            print(
                f"New best model saved with validation F1: {val_f1:.4f} (accuracy: {val_acc * 100:.2f}%)"
            )
        else:
            no_improve_count += 1

        if no_improve_count >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        print("-" * 60)
    return train_metrics


def plot_learning_curves(df_metrics):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.lineplot(
        data=df_metrics,
        x="epoch",
        y="train_loss",
        marker="o",
        ax=axes[0],
        label="Train Loss",
    )
    sns.lineplot(
        data=df_metrics,
        x="epoch",
        y="val_loss",
        marker="o",
        ax=axes[0],
        label="Validation Loss",
    )
    axes[0].set_title("Loss vs Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    sns.lineplot(
        data=df_metrics,
        x="epoch",
        y="train_f1",
        marker="o",
        ax=axes[1],
        label="Train F1",
    )
    sns.lineplot(
        data=df_metrics,
        x="epoch",
        y="val_f1",
        marker="o",
        ax=axes[1],
        label="Validation F1",
    )
    axes[1].set_title("F1 Score vs Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F1 Score")
    plt.tight_layout()
    # Use LEARNING_CURVES_DIR from config:
    learning_curves_dir = config.get("LEARNING_CURVES_DIR")
    os.makedirs(learning_curves_dir, exist_ok=True)
    plt.savefig(os.path.join(learning_curves_dir, "learning_curves.png"))
    # plt.show() has been removed


def generate_test_embeddings(triplet_model, device, test_df=None):
    """Generate embeddings for test data after model training to prevent data leakage"""
    print("\nPHASE 2: Generating test embeddings after model training...")

    # Use stored test data if available, otherwise load from TEST_CSV
    if test_df is not None:
        df_test = test_df
    else:
        global test_data_for_later
        if "test_data_for_later" in globals() and test_data_for_later is not None:
            df_test = test_data_for_later
        else:
            # Fallback: load from placeholder CSV and manually remove placeholder embeddings
            print("Warning: Using fallback method to load test data")
            df_test = pd.read_csv(TEST_CSV)
            if "embedding" in df_test.columns:
                del df_test["embedding"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Re-use the compute_embeddings_batched function for consistency
    def compute_embeddings_batched(texts, batch_size=32):
        embeddings = []

        # Process in batches to reduce memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=MAX_SEQ_LEN,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                outputs = triplet_model(input_ids, attention_mask)

            normalized = F.normalize(outputs, p=2, dim=1)
            embeddings.extend(normalized.cpu().numpy().tolist())

            # Print progress for large datasets
            if i % (batch_size * 10) == 0:
                print(f"Processed {i}/{len(texts)} texts")

        return embeddings

    # Generate embeddings for test data
    print("PHASE 2: Generating embeddings for test set...")
    df_test["embedding"] = compute_embeddings_batched(df_test["text"].tolist())

    # Save test data with embeddings
    df_test.to_csv(TEST_CSV, index=False)
    print("PHASE 2: Test embeddings generated and saved.")

    # Create a dataset and loader for the test data
    test_ds = EmbeddingDataset(TEST_CSV, device)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return test_loader


def evaluate_best_model(model, triplet_model, device, criterion):
    """Evaluate the best model with fresh test embeddings to prevent data leakage"""
    # First, generate test embeddings with the trained model
    test_loader = generate_test_embeddings(triplet_model, device)

    # Then load and evaluate the best model
    model_dir = os.path.join(CHECKPOINT_DIR, "model")
    best_checkpoint = torch.load(
        os.path.join(model_dir, "best_model.pt")
    )  # Load from checkpoints/model
    model.load_state_dict(best_checkpoint["model_state_dict"])

    print("\nEvaluating model on freshly generated test embeddings...")
    return evaluate_model(model, test_loader, criterion)


# Function to generate and save embedding CSV data, preventing data leakage
def generate_embedding_csv_data(triplet_model, device):
    """Generate embedding data from raw csv and save to CSV
    with strict separation to prevent data leakage"""

    print("Using pre-created splits from TripletTraining.py to prevent data leakage...")
    
    # Load the pre-created splits from TripletTraining.py
    try:
        df_train = pd.read_csv(TRAIN_CSV)
        df_val = pd.read_csv(VAL_CSV) 
        df_test = pd.read_csv(TEST_CSV)
        print(f"Successfully loaded pre-created splits")
    except FileNotFoundError:
        print("ERROR: Pre-created splits not found. Please run TripletTraining.py first to create the splits.")
        print("This is required to prevent data leakage between embedding training and classification.")
        print("Run: python FTC/TripletTraining.py --epochs 10 --lr 2e-5")
        sys.exit(1)
        
    # Apply text preprocessing
    df_train["text"] = df_train["text"].apply(preprocess_text)
    df_val["text"] = df_val["text"].apply(preprocess_text)
    df_test["text"] = df_test["text"].apply(preprocess_text)

    # Fit label encoder ONLY on training data
    label_encoder = LabelEncoder()
    df_train["label_enc"] = label_encoder.fit_transform(df_train["labels"])
    df_val["label_enc"] = label_encoder.transform(df_val["labels"])
    df_test["label_enc"] = label_encoder.transform(df_test["labels"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # Helper: compute mean pooled embedding from triplet_model output with batching
    def compute_embeddings_batched(texts, batch_size=32):
        embeddings = []

        # Process in batches to reduce memory usage
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                max_length=MAX_SEQ_LEN,
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                input_ids = encoded["input_ids"]
                attention_mask = encoded["attention_mask"]
                outputs = triplet_model(input_ids, attention_mask)

            normalized = F.normalize(outputs, p=2, dim=1)
            embeddings.extend(normalized.cpu().numpy().tolist())

            # Clear CUDA cache periodically
            if torch.cuda.is_available() and i % (batch_size * 10) == 0:
                torch.cuda.empty_cache()

            # Print progress for large datasets
            if i % (batch_size * 10) == 0:
                print(f"Processed {i}/{len(texts)} texts")

        return embeddings

    # PHASE 1: ONLY generate embeddings for training and validation sets
    # to train the classifier without leaking test data
    print("PHASE 1: Generating embeddings for training set...")
    df_train["embedding"] = compute_embeddings_batched(df_train["text"].tolist())
    print("PHASE 1: Generating embeddings for validation set...")
    df_val["embedding"] = compute_embeddings_batched(df_val["text"].tolist())

    # Save train/val files for classifier training
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV, index=False)

    # Save test dataframe without embeddings for later processing
    # Store in a global variable for later use
    global test_data_for_later
    test_data_for_later = df_test.copy()

    # Create a placeholder test CSV with the structure but no real embeddings
    # This is to prevent embeddings_are_outdated() from regenerating all embeddings
    df_test_structure = df_test.copy()
    # Add placeholder empty embeddings
    embedding_dim = triplet_model.base_model.config.hidden_size
    df_test_structure["embedding"] = [
        [0.0] * embedding_dim for _ in range(len(df_test))
    ]
    df_test_structure.to_csv(TEST_CSV, index=False)

    print("PHASE 1: Training embeddings generated and saved.")
    print("Test set embeddings will be generated separately after model training.")


class EmbeddingDataset(Dataset):
    """Dataset for handling precomputed embeddings loaded from a CSV file."""

    def __init__(self, csv_file, device):
        self.df = pd.read_csv(csv_file)
        self.df["embedding"] = self.df["embedding"].apply(ast.literal_eval)
        self.embeddings = torch.tensor(
            self.df["embedding"].tolist(), dtype=torch.float, device=device
        )
        self.labels = torch.tensor(
            self.df["label_enc"].tolist(), dtype=torch.long, device=device
        )

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


# New loader to read CSV files with embeddings
def load_embedding_data(device):
    global NUM_LABEL
    # Load CSVs using new EmbeddingDataset and a LabelEncoder from train split.
    train_ds = EmbeddingDataset(TRAIN_CSV, device)
    val_ds = EmbeddingDataset(VAL_CSV, device)
    test_ds = EmbeddingDataset(TEST_CSV, device)
    # Re-fit LabelEncoder using train CSV labels (assuming same order as used in CSV generation)
    df_train = pd.read_csv(TRAIN_CSV)
    label_encoder = LabelEncoder()
    label_encoder.fit(df_train["labels"])
    NUM_LABEL = len(label_encoder.classes_)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader, label_encoder


# Add new function to load the TripletEmbeddingModel
def load_triplet_model(device):
    base_model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    triplet_model = TripletEmbeddingModel(base_model).to(device)

    try:
        # Use weights_only for safe loading when available in PyTorch version
        import torch

        torch_version = tuple(map(int, torch.__version__.split(".")[:2]))

        if torch_version >= (1, 13):
            # PyTorch 1.13+ supports weights_only parameter
            checkpoint = torch.load(
                EMBEDDING_PATH, map_location=device, weights_only=True
            )
        else:
            # Fallback for older PyTorch versions
            checkpoint = torch.load(EMBEDDING_PATH, map_location=device)

        # Check if model is saved with nested state_dict
        if "model_state_dict" in checkpoint:
            triplet_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            triplet_model.load_state_dict(checkpoint)

    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Trying alternative loading method...")
        # Last resort fallback method
        checkpoint = torch.load(EMBEDDING_PATH, map_location=device)
        if "model_state_dict" in checkpoint:
            triplet_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            triplet_model.load_state_dict(checkpoint)

    triplet_model.eval()
    return triplet_model


def embeddings_are_outdated():
    # Check if CSV files exist; if not, need update.
    if not (
        os.path.exists(TRAIN_CSV)
        and os.path.exists(VAL_CSV)
        and os.path.exists(TEST_CSV)
    ):
        return True
    model_mtime = os.path.getmtime(EMBEDDING_PATH)
    train_mtime = os.path.getmtime(TRAIN_CSV)
    val_mtime = os.path.getmtime(VAL_CSV)
    test_mtime = os.path.getmtime(TEST_CSV)
    # Regenerate if any CSV is older than the triplet model file.
    return (
        train_mtime < model_mtime or val_mtime < model_mtime or test_mtime < model_mtime
    )


def preprocess_text(text):
    """Clean and normalize the input text."""
    if not isinstance(text, str):
        return ""
    text = TAG_RE.sub("", text)
    text = " ".join(text.split())
    return text


def get_default_device():
    """Get the default compute device for PyTorch."""
    # Use the unified device detection function
    return get_device()


def create_classifier_from_config(config, input_dim, output_dim):
    """Create a classifier based on a complete configuration.
    This is a standardized function to ensure consistent architecture creation.
    """
    model_type = config.get("model_type", "sequential")

    # Standardized parameter extraction with clear defaults
    dropout_rate = config.get("dropout_rate", 0.3)
    activation_fn = config.get("activation", "gelu")

    # Check for explicit architecture information first
    if "architecture" in config:
        print("Creating classifier from explicit architecture definition")
        architecture = config["architecture"]
        layers = []

        # Update the architecture to match the current model's embedding dimensions
        if len(architecture) > 0 and architecture[0].get("layer_type") == "dense_block":
            actual_dim = architecture[0]["input_size"]
            if actual_dim != input_dim:
                print(f"WARNING: Architecture dimension mismatch! Updating from {actual_dim} to {input_dim}")
                # Update the first layer's input size to match the actual embedding dimension
                architecture[0]["input_size"] = input_dim

        # Create each layer according to its definition
        for layer in architecture:
            if layer["layer_type"] == "dense_block":
                block = DenseBlock(
                    layer["input_size"],
                    layer["output_size"],
                    layer.get("dropout_rate", dropout_rate),
                    layer.get("activation", activation_fn),
                )
                layers.append(block)
            elif layer["layer_type"] == "linear":
                layers.append(nn.Linear(layer["input_size"], layer["output_size"]))

        return nn.Sequential(*layers)

    # Fall back to n_hidden + hidden_units_X pattern if architecture not explicitly defined
    elif "n_hidden" in config:
        print("Creating classifier from n_hidden and hidden_units parameters")
        n_hidden = config["n_hidden"]
        layers = []
        prev_width = input_dim

        # For each hidden layer, get its width from config
        for i in range(n_hidden):
            width_key = f"hidden_units_{i}"
            if width_key in config:
                width = config[width_key]
                print(f"  Layer {i + 1}: {prev_width} → {width}")
            else:
                # Default fallback with clear warning
                width = prev_width // 2
                print(
                    f"  WARNING: Missing {width_key} in config, using default: {width}"
                )

            layers.append(DenseBlock(prev_width, width, dropout_rate, activation_fn))
            prev_width = width

        # Add output layer
        print(f"  Output layer: {prev_width} → {output_dim}")
        layers.append(nn.Linear(prev_width, output_dim))

        return nn.Sequential(*layers)

    # Fall back to default FFModel as last resort
    else:
        print("Using default FFModel architecture (no architecture details in config)")
        return FFModel(input_dim, output_dim, dropout_rate, activation_fn=activation_fn)


def get_arg_parser():
    """Create and return the argument parser"""
    parser = argparse.ArgumentParser(
        description="Train a BERT classifier for text classification"
    )
    parser.add_argument(
        "--unfreeze",
        type=int,
        default=0,
        help="Number of BERT layers to unfreeze for fine-tuning (0 = freeze all, -1 = unfreeze all)",
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=NUM_EPOCHS,
        help=f"Number of training epochs (default: {NUM_EPOCHS})"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=LEARNING_RATE,
        help=f"Learning rate (default: {LEARNING_RATE})"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=DROP_OUT,
        help=f"Dropout rate for classifier (default: {DROP_OUT})"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=BATCH_SIZE,
        help=f"Batch size for training (default: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd", "rmsprop"],
        help="Optimizer to use for training (default: from config)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=PATIENCE,
        help=f"Early stopping patience (default: {PATIENCE})"
    )
    return parser

def main():
    """Main function for BERT classification with direct text processing."""
    # Parse command line arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    
    # Check if TripletTraining.py has been run first to prevent data leakage
    if not all(os.path.exists(path) for path in [TRAIN_CSV, VAL_CSV, TEST_CSV]):
        print("ERROR: Data splits not found. Please run TripletTraining.py first.")
        print("This is required to prevent data leakage between embedding training and classification.")
        print("Run: python FTC/TripletTraining.py --epochs 10 --lr 2e-5")
        sys.exit(1)
    
    # Start of original main function
    DEVICE = get_default_device()
    print(f"Using device: {DEVICE}")
    print("Using consistent train/val/test splits from TripletTraining.py to prevent data leakage.")

    # Create checkpoint directory using pathlib
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    # Load best configuration if it exists from BEST_CONFIG_DIR in the config
    # Use pathlib for safe path handling
    best_config_file = str(
        Path(config["BEST_CONFIG_DIR"]) / f"best_config_{STUDY_NAME}.yml"
    )
    best_config = {}

    if os.path.exists(best_config_file):
        with open(best_config_file, "r", encoding="utf-8") as f:
            best_config = yaml.safe_load(f)
        print(f"Loaded best configuration from: {best_config_file}")

        # Option to verify configuration integrity if hash is available
        if "architecture_hash" in best_config and "architecture" in best_config:
            import hashlib

            arch_str = str(best_config["architecture"])
            current_hash = hashlib.md5(arch_str.encode()).hexdigest()
            stored_hash = best_config["architecture_hash"]
            if current_hash != stored_hash:
                print("WARNING: Architecture definition may have been modified!")
            else:
                print("Architecture integrity verified ✓")
    else:
        print("No best configuration file found. Using default parameters.")

    # Load the TripletEmbeddingModel that was trained by TripletTraining.py
    print("Loading pre-trained triplet embedding model...")
    triplet_model = load_triplet_model(DEVICE)

    # Initialize tokenizer for processing raw text
    print("Initializing tokenizer for text processing...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    # Load the data using our TextDataset approach (no precomputed embeddings)
    print("Loading and processing raw text data...")
    train_loader, val_loader, test_loader, label_encoder, num_classes = load_csv_data(
        DEVICE, tokenizer
    )

    # Use the number of classes from the data
    global NUM_LABEL
    NUM_LABEL = num_classes
    print(f"Detected {NUM_LABEL} classes: {label_encoder.classes_}")

    # Get input dimension from triplet model
    classifier_input_size = triplet_model.base_model.config.hidden_size

    # Ensure config has all necessary dimensions (update if needed)
    config_updated = False
    if (
        "input_dim" not in best_config
        or best_config["input_dim"] != classifier_input_size
    ):
        best_config["input_dim"] = classifier_input_size
        config_updated = True

    if "output_dim" not in best_config or best_config["output_dim"] != NUM_LABEL:
        best_config["output_dim"] = NUM_LABEL
        config_updated = True

    # Create the classifier using our standardized function
    classifier = create_classifier_from_config(
        best_config, classifier_input_size, NUM_LABEL
    )

    # Create the full model
    model = BertClassifier(triplet_model, classifier).to(DEVICE)

    # Apply unfreezing if requested
    if hasattr(args, "unfreeze") and args.unfreeze != 0:
        print(f"Unfreezing {args.unfreeze} BERT layers for fine-tuning...")
        model.unfreeze_bert_layers(args.unfreeze)
        print("Model parameter status:")
        for name, param in model.named_parameters():
            requires_grad = "Trainable" if param.requires_grad else "Frozen"
            print(f"  - {name}: {requires_grad}")
    else:
        print("BERT layers are frozen. Only training classifier layers.")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params / total_params:.2%})"
    )

    # Only update config file if necessary
    if config_updated:
        print("Updating configuration file with current dimensions")
        with open(best_config_file, "w", encoding="utf-8") as f:
            yaml.dump(best_config, f)

    # Override settings from command line args first, then config
    # Store local variables instead of using globals
    epochs = args.epochs
    if epochs != NUM_EPOCHS:
        print(f"Overriding epochs from command line: {epochs}")
        
    # Set learning rate - command line takes priority over config
    lr = args.lr if args.lr != LEARNING_RATE else float(best_config.get("lr", LEARNING_RATE))
    
    # Set optimizer - command line takes priority over config
    optimizer_choice = args.optimizer if args.optimizer else best_config.get("optimizer", "adamw")
    
    # Set other parameters
    weight_decay = float(best_config.get("weight_decay", 0))
    
    # Store batch size as local variable
    batch_size = args.batch_size
    if batch_size != BATCH_SIZE:
        print(f"Overriding batch size from command line: {batch_size}")
        
    # Store patience as local variable
    patience = args.patience
    if patience != PATIENCE:
        print(f"Overriding patience from command line: {patience}")
        
    # Store dropout rate as local variable
    dropout_rate = args.dropout
    if dropout_rate != DROP_OUT:
        print(f"Overriding dropout rate from command line: {dropout_rate}")
        # Also override in best_config to ensure it's used when creating the classifier
        best_config["dropout_rate"] = dropout_rate

    if optimizer_choice == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_choice == "sgd":
        momentum = best_config.get("momentum", 0.0)
        nesterov = best_config.get("nesterov", False)
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif optimizer_choice == "rmsprop":
        momentum = best_config.get("momentum", 0.0)
        alpha = best_config.get("alpha", 0.99)
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
    # Learning rate scheduler using F1 score as the metric to monitor
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE, min_lr=float(MIN_LR)
    )

    print("Starting training...")
    metrics = train_model(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        num_epochs=epochs, patience=patience
    )
    # Save metrics using pathlib for safe path handling
    np.save(str(Path(CHECKPOINT_DIR) / "training_metrics.npy"), metrics)
    df_metrics = pd.DataFrame(metrics)
    plot_learning_curves(df_metrics)
    # Evaluate the best model on the test set
    test_loss, test_acc, test_f1 = evaluate_model(model, test_loader, criterion)

    # Print a summary of the model architecture and saved location
    # Use pathlib for safe path handling
    model_save_path = str(Path(CHECKPOINT_DIR) / "model" / "best_model.pt")
    print("\n" + "=" * 80)
    print("MODEL TRAINING SUMMARY")
    print("=" * 80)
    print(f"Pretrained Model: {MODEL_NAME}")
    print("Architecture:")
    print(f"  - Embedding Model: {triplet_model.__class__.__name__}")

    # Improved classifier architecture reporting
    if isinstance(classifier, nn.Sequential):
        print("  - Classifier: Custom Sequential Architecture")
        print(f"    - Input Dimension: {classifier_input_size}")
        print(f"    - Output Classes: {NUM_LABEL}")

        # Get n_hidden safely from the config or count layers in sequential model
        n_hidden = best_config.get("n_hidden", 0)
        if n_hidden == 0:
            # Count DenseBlock layers in sequential model
            n_hidden = sum(1 for m in classifier if isinstance(m, DenseBlock))

        print(f"    - Hidden Layers: {n_hidden}")

        # Display each layer's configuration only if n_hidden > 0
        for i in range(n_hidden):
            width_key = f"hidden_units_{i}"
            width = best_config.get(width_key, classifier_input_size // (2 ** (i + 1)))
            print(f"      - Layer {i + 1}: {width} units")

        # Extract dropout rate and activation function if available
        classifier_dropout = best_config.get("dropout_rate", DROP_OUT)
        classifier_activation = best_config.get("activation", ACT_FN)

        print(f"    - Dropout Rate: {classifier_dropout}")
        print(f"    - Activation Function: {classifier_activation}")
    else:
        # For non-sequential classifiers like FFModel
        print(f"  - Classifier: {classifier.__class__.__name__}")
        print(f"    - Input Dimension: {classifier_input_size}")
        print(f"    - Output Classes: {NUM_LABEL}")
        print(f"    - Dropout Rate: {best_config.get('dropout_rate', DROP_OUT)}")
        print(f"    - Activation Function: {best_config.get('activation', ACT_FN)}")

    print("Training:")
    print(f"  - Optimizer: {optimizer.__class__.__name__}")
    print(f"  - Learning Rate: {lr}")
    if optimizer_choice == "sgd" or optimizer_choice == "rmsprop":
        print(f"  - Momentum: {best_config.get('momentum', 0.0)}")
    if optimizer_choice == "rmsprop":
        print(f"  - Alpha: {best_config.get('alpha', 0.99)}")
    print(f"  - Weight Decay: {weight_decay}")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Max Sequence Length: {MAX_SEQ_LEN}")
    print("Test Performance (with direct text processing):")
    print(f"  - Loss: {test_loss:.4f}")
    print(f"  - Accuracy: {test_acc * 100:.2f}%")
    print(f"  - F1 Score: {test_f1:.4f} (using standardized calculation)")
    print(f"  - Using end-to-end processing of text without precomputed embeddings")

    # Add note about cross-validation
    print("Note on Metrics:")
    print("  - These results are from a single test set split.")
    print("  - For more robust evaluation, use Evaluation.py for cross-validation.")
    print(
        "  - Cross-validation metrics may be lower but more reflective of true performance."
    )
    print("File Locations:")
    print(f"  - Model Saved To: {model_save_path}")
    print(f"  - Embeddings Model: {EMBEDDING_PATH}")
    print(f"  - Training Data: {TRAIN_CSV}")
    print(f"  - Configuration: {best_config_file}")
    print("=" * 80)

    # Print message about log file
    print(f"\nExecution log saved to: {log_file}")

    return model, test_acc, test_f1


if __name__ == "__main__":
    main()
