import ast
import os
import random
import re

# Add the current directory to sys.path to ensure module imports work
# current_dir = os.path.dirname(os.path.abspath(__file__))
# if current_dir not in sys.path:
#     sys.path.insert(0, current_dir)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from TripletTraining import TripletEmbeddingModel
from utils.utils import get_default_device, preprocess_text

# Safe import with fallbacks for utils
# try:
#     from utils.LoaderSetup import join_constructor
#     from utils.utils import get_default_device, preprocess_text

#     print("Successfully imported from utils package")
# except ImportError as e:
#     print(f"Error importing from utils package: {e}")

#     # Define fallbacks if imports fail
#     def join_constructor(loader, node):
#         seq = loader.construct_sequence(node)
#         return "".join(seq)

#     def get_default_device():
#         if torch.cuda.is_available():
#             return torch.device("cuda")
#         return torch.device("cpu")

#     def preprocess_text(text):
#         if not isinstance(text, str):
#             return ""
#         TAG_RE = re.compile(r"<[^>]+>")
#         text = TAG_RE.sub("", text)
#         text = " ".join(text.split())
#         return text

#     # Register YAML constructor
#     yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
#     print("Using fallback implementations for utils functions")

# Safe import with additional fallbacks for TripletEmbeddingModel
# try:
#     from TripletTraining import TripletEmbeddingModel

#     print("Successfully imported TripletEmbeddingModel from TripletTraining")
# except ImportError as e:
#     print(f"Error importing from TripletTraining: {e}")

#     # Define a fallback TripletEmbeddingModel if import fails
#     class TripletEmbeddingModel(nn.Module):
#         def __init__(self, base_model):
#             super(TripletEmbeddingModel, self).__init__()
#             self.base_model = base_model

#         def _mean_pooled(self, last_hidden_states, attention_mask):
#             last_hidden = last_hidden_states.masked_fill(
#                 ~attention_mask[..., None].bool(), 0.0
#             )
#             return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

#         def forward(self, input_ids, attention_mask):
#             outputs = self.base_model(
#                 input_ids=input_ids, attention_mask=attention_mask
#             )
#             last_hidden_state = outputs.last_hidden_state
#             pooled_output = self._mean_pooled(last_hidden_state, attention_mask)
#             normalized_output = F.normalize(pooled_output, p=2, dim=1)
#             return normalized_output

#     print("Using fallback implementation for TripletEmbeddingModel")

# Load configuration
# with open(os.path.join(current_dir, "config.yml"), "r", encoding="utf-8") as f:
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
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
MODEL = config["MODEL"]
MODEL_NAME = config["MODEL_NAME"]
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


def load_csv_data(device):
    # Read the CSV file which has columns "labels" and "text"
    df = pd.read_csv(CSV_PATH)
    df["text"] = df["text"].apply(preprocess_text)

    # Split out the test set (20% of the data)
    df_train_val, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["labels"]
    )
    # Split train and validation (from train_val, 25% becomes validation so overall: 60% train, 20% val, 20% test)
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.25, random_state=SEED, stratify=df_train_val["labels"]
    )

    print(
        f"Train samples: {len(df_train)}, Val samples: {len(df_val)}, Test samples: {len(df_test)}"
    )

    # Fit LabelEncoder on training labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train["labels"])
    y_val = label_encoder.transform(df_val["labels"])
    y_test = label_encoder.transform(df_test["labels"])

    # Update NUM_LABEL based on the number of classes
    NUM_LABEL = len(label_encoder.classes_)
    print(f"Detected {NUM_LABEL} classes: {label_encoder.classes_}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    train_encoded = tokenizer(
        df_train["text"].tolist(),
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    val_encoded = tokenizer(
        df_val["text"].tolist(),
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    test_encoded = tokenizer(
        df_test["text"].tolist(),
        max_length=MAX_SEQ_LEN,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)

    train_ds = TokenizedDataset(train_encoded, y_train, device)
    val_ds = TokenizedDataset(val_encoded, y_val, device)
    test_ds = TokenizedDataset(test_encoded, y_test, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader, label_encoder, NUM_LABEL


def save_checkpoint(model, optimizer, epoch, val_acc, best_val_acc, filename):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "best_val_acc": best_val_acc,
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["val_acc"], checkpoint["best_val_acc"]


def evaluate_model(model, test_loader, criterion):
    test_loss, test_acc, test_f1 = validate(model, test_loader, criterion)
    print("\nTest Set Evaluation:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    print(f"Test F1 Score: {test_f1:.4f}")
    return test_loss, test_acc, test_f1


class TokenizedDataset(Dataset):
    def __init__(self, wrapped_input, labels, device):
        self.wrapped_input = wrapped_input
        self.labels = torch.tensor(labels, dtype=torch.long, device=device)

    def __getitem__(self, idx):
        input_dict = {k: self.wrapped_input[k][idx] for k in self.wrapped_input.keys()}
        return input_dict, self.labels[idx]

    def __len__(self):
        return len(self.labels)


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
    def __init__(self, base_model, classifier):
        super(BertClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = classifier

    def _mean_pooled(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, inputs):
        # If input is a dict, process as tokenized input.
        if isinstance(inputs, dict):
            inputs.pop("token_type_ids", None)
            outputs = self.base_model(**inputs)
            if isinstance(outputs, torch.Tensor):
                normalized_output = outputs
            else:
                last_hidden_state = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                pooled_output = self._mean_pooled(last_hidden_state, attention_mask)
                normalized_output = F.normalize(pooled_output, p=2, dim=1)
        elif isinstance(inputs, tuple):
            input_ids, attention_mask = inputs
            outputs = self.base_model(
                input_ids=input_ids, attention_mask=attention_mask
            )
            if isinstance(outputs, torch.Tensor):
                normalized_output = outputs
            else:
                last_hidden_state = outputs.last_hidden_state
                pooled_output = self._mean_pooled(last_hidden_state, attention_mask)
                normalized_output = F.normalize(pooled_output, p=2, dim=1)
        else:
            normalized_output = inputs  # Already embeddings.
        logits = self.classifier(normalized_output)
        return logits


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
    train_f1 = f1_score(
        all_labels, all_preds, average="macro"
    )  # updated to handle multiclass
    return total_loss / len(dataloader), train_f1


def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validating", leave=False)
        for batch_data, batch_labels in progress_bar:
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(batch_labels.detach().cpu().numpy())
            progress_bar.set_postfix({"val_loss": f"{loss.item():.4f}"})
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    val_f1 = f1_score(
        all_labels, all_preds, average="macro"
    )  # updated to handle multiclass
    val_loss = total_loss / len(dataloader)
    val_acc = (all_preds == all_labels).mean()
    return val_loss, val_acc, val_f1


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler):
    train_metrics = []
    best_val_acc = 0
    no_improve_count = 0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_f1 = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion)
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]["lr"]
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
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"Training Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        print(
            f"Validation Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%, Val F1: {val_f1:.4f}"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
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
                os.path.join(
                    model_dir, "best_model.pt"
                ),  # Updated to save in checkpoints/model
            )
            print(
                f"New best model saved with validation accuracy: {val_acc * 100:.2f}%"
            )
        else:
            no_improve_count += 1

        if no_improve_count >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break

        print("-" * 60)
    return train_metrics


def plot_learning_curves(df_metrics):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
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


def evaluate_best_model(model, test_loader, criterion):
    model_dir = os.path.join(CHECKPOINT_DIR, "model")
    best_checkpoint = torch.load(
        os.path.join(model_dir, "best_model.pt")
    )  # Load from checkpoints/model
    model.load_state_dict(best_checkpoint["model_state_dict"])
    return evaluate_model(model, test_loader, criterion)


# New function to generate and save embedding CSV data
def generate_embedding_csv_data(triplet_model, device):
    # Load original CSV data with raw text and labels.
    df = pd.read_csv(CSV_PATH)
    df["text"] = df["text"].apply(preprocess_text)

    # Split into train, val, test (60/20/20)
    df_train_val, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["labels"]
    )
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.25, random_state=SEED, stratify=df_train_val["labels"]
    )

    # Fit and update label encoder.
    label_encoder = LabelEncoder()
    df_train["label_enc"] = label_encoder.fit_transform(df_train["labels"])
    df_val["label_enc"] = label_encoder.transform(df_val["labels"])
    df_test["label_enc"] = label_encoder.transform(df_test["labels"])

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # Helper: compute mean pooled embedding from triplet_model output.
    def compute_embeddings(texts):
        encoded = tokenizer(
            texts,
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
        return normalized.cpu().numpy().tolist()

    # Compute embeddings for each split
    df_train["embedding"] = compute_embeddings(df_train["text"].tolist())
    df_val["embedding"] = compute_embeddings(df_val["text"].tolist())
    df_test["embedding"] = compute_embeddings(df_test["text"].tolist())

    # Save CSV files with embeddings
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)

    print("Embedding CSV data generated and saved.")


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


# A function to load the TripletEmbeddingModel
def load_triplet_model(device):
    base_model = AutoModel.from_pretrained(MODEL).to(device)
    triplet_model = TripletEmbeddingModel(base_model).to(device)
    triplet_model.load_state_dict(torch.load(EMBEDDING_PATH, map_location=device))
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


def main():
    DEVICE = get_default_device()
    print(f"Using device: {DEVICE}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load best configuration if it exists.
    best_config_file = os.path.join(
        config["BEST_CONFIG_DIR"], f"best_config_{STUDY_NAME}.yml"
    )
    if os.path.exists(best_config_file):
        with open(best_config_file, "r", encoding="utf-8") as f:
            best_config = yaml.safe_load(f)
        print("Loaded best configuration from YAML:")
        # print(best_config)
    else:
        best_config = {}
        print("No best configuration file found. Using default parameters.")

    # Load the TripletEmbeddingModel
    triplet_model = load_triplet_model(DEVICE)

    # Regenerate embeddings if missing or outdated.
    if embeddings_are_outdated():
        print(
            "Embeddings CSV files are outdated or missing. Regenerating embeddings..."
        )
        generate_embedding_csv_data(triplet_model, DEVICE)

    # Load precomputed embedding data and proceed.
    train_loader, val_loader, test_loader, label_encoder = load_embedding_data(DEVICE)
    print(f"Detected {NUM_LABEL} classes: {label_encoder.classes_}")

    # Override default classifier parameters with best config if available.
    classifier_dropout = best_config.get("dropout_rate", DROP_OUT)
    classifier_activation = best_config.get("activation", ACT_FN)
    classifier_input_size = triplet_model.base_model.config.hidden_size

    # Initialize FFModel using the measured dimension.
    classifier = FFModel(
        classifier_input_size,
        NUM_LABEL,
        classifier_dropout,
        activation_fn=classifier_activation,
    )
    model = BertClassifier(triplet_model, classifier).to(DEVICE)

    # Override optimizer settings if available.
    lr = float(best_config.get("lr", LEARNING_RATE))  # Cast lr to float
    optimizer_choice = best_config.get("optimizer", "adamw")
    weight_decay = best_config.get("weight_decay", 0)

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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE, min_lr=float(MIN_LR)
    )

    print("Starting training...")
    metrics = train_model(
        model, train_loader, val_loader, optimizer, criterion, scheduler
    )
    np.save(os.path.join(CHECKPOINT_DIR, "training_metrics.npy"), metrics)
    df_metrics = pd.DataFrame(metrics)
    # print(df_metrics)
    plot_learning_curves(df_metrics)
    _, _, _ = evaluate_best_model(model, test_loader, criterion)


if __name__ == "__main__":
    main()
