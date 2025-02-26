import os
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from utils.utils import preprocess_text

# Import the join_constructor and register it explicitly here
from utils.LoaderSetup import join_constructor

# Ensure we register the constructor for this module's yaml usage
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)

# Load configuration from one level up
config_path = os.path.join(os.path.dirname(__file__), "../config.yml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

MODEL = config["MODEL"]
EMBEDDING_PATH = config["EMBEDDING_PATH"]
CSV_PATH = config["CSV_PATH"]
TRAIN_CSV = config["TRAIN_CSV"]
VAL_CSV = config["VAL_CSV"]
TEST_CSV = config["TEST_CSV"]
MAX_SEQ_LEN = config["MAX_SEQ_LEN"]
SEED = config["SEED"]
BATCH_SIZE = config["BATCH_SIZE"]
# Add these missing variables needed by TuneBert.py
STUDY_NAME = config["STUDY_NAME"]
DROP_OUT = config.get("DROP_OUT", 0.2)
ACT_FN = config.get("ACT_FN", "gelu")
LEARNING_RATE = config.get("LEARNING_RATE", 2e-5)
PATIENCE = config.get("PATIENCE", 5)
LR_PATIENCE = config.get("LR_PATIENCE", 2)
MIN_LR = config.get("MIN_LR", 0.000001)
CHECKPOINT_DIR = config["CHECKPOINT_DIR"]
BEST_CONFIG_DIR = config.get("BEST_CONFIG_DIR", "best_configs")
NUM_LABEL = config.get("NUM_LABEL", 5)  # Default to 5 if not specified


def get_default_device():
    """Get the default compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TripletEmbeddingModel(nn.Module):
    def __init__(self, base_model):
        super(TripletEmbeddingModel, self).__init__()
        self.base_model = base_model

    def _mean_pooled(self, last_hidden_states, attention_mask):
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self._mean_pooled(last_hidden_state, attention_mask)
        normalized_output = F.normalize(pooled_output, p=2, dim=1)
        return normalized_output


def load_triplet_model(device):
    """
    Load the triplet model with improved error handling for model architecture mismatches.
    """
    # Check if the saved model exists
    model_exists = os.path.exists(EMBEDDING_PATH)

    # Create a fresh model with the specified MODEL name regardless
    try:
        base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
        triplet_model = TripletEmbeddingModel(base_model).to(device)
    except Exception as e:
        print(f"Error loading model {MODEL_NAME}: {e}")
        print(f"Falling back to {MODEL} instead")
        base_model = AutoModel.from_pretrained(MODEL).to(device)
        triplet_model = TripletEmbeddingModel(base_model).to(device)

    # If saved model exists, try to load its weights with appropriate error handling
    if model_exists:
        try:
            print(f"Loading triplet model from {EMBEDDING_PATH}")
            # Load state dict but ignore mismatched keys
            checkpoint = torch.load(EMBEDDING_PATH, map_location=device)

            # Filter out size mismatched parameters
            model_dict = triplet_model.state_dict()
            pretrained_dict = {
                k: v
                for k, v in checkpoint.items()
                if k in model_dict and v.size() == model_dict[k].size()
            }

            # Update model with compatible parameters only
            model_dict.update(pretrained_dict)
            triplet_model.load_state_dict(model_dict)
            print(
                f"Loaded {len(pretrained_dict)}/{len(checkpoint)} parameters successfully"
            )
        except Exception as e:
            warnings.warn(f"Error loading saved model weights: {e}. Using fresh model.")
    else:
        print(f"Warning: No saved model found at {EMBEDDING_PATH}. Using fresh model.")

    triplet_model.eval()
    return triplet_model


def create_dummy_embedding_data(device, num_samples=1000, num_classes=5):
    """Create dummy embedding data for testing when real data is not available"""
    print("Creating dummy embedding data for testing...")

    # Create directories
    for path in [
        os.path.dirname(TRAIN_CSV),
        os.path.dirname(VAL_CSV),
        os.path.dirname(TEST_CSV),
    ]:
        os.makedirs(path, exist_ok=True)

    # Generate random embeddings and labels
    embedding_dim = 768  # Standard dimension for BERT embeddings

    for file_path, num in [
        (TRAIN_CSV, int(num_samples * 0.6)),
        (VAL_CSV, int(num_samples * 0.2)),
        (TEST_CSV, int(num_samples * 0.2)),
    ]:
        # Generate random data
        embeddings = torch.randn(num, embedding_dim).numpy().tolist()
        labels = np.random.randint(0, num_classes, num)
        label_names = [f"class_{i}" for i in labels]

        # Create DataFrame and save
        df = pd.DataFrame(
            {
                "text": [f"Sample text {i}" for i in range(num)],
                "labels": label_names,
                "label_enc": labels,
                "embedding": embeddings,
            }
        )
        df.to_csv(file_path, index=False)
        print(f"Saved {num} dummy samples to {file_path}")

    return True


def generate_embedding_csv_data(triplet_model, device):
    """Generate embedding data from raw text CSV"""
    if not os.path.exists(CSV_PATH):
        print(f"Error: Source CSV file not found: {CSV_PATH}")
        print("Creating dummy embedding data instead...")
        return create_dummy_embedding_data(device)

    print(f"Generating embeddings from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    df["text"] = df["text"].apply(preprocess_text)
    from sklearn.model_selection import train_test_split

    # Check if there's enough data to stratify
    stratify_by = df["labels"] if len(df["labels"].unique()) > 1 else None
    if stratify_by is None:
        print(
            "Warning: Not enough unique labels for stratification. Using random split."
        )

    df_train_val, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=stratify_by
    )

    # For the second split
    stratify_val = df_train_val["labels"] if stratify_by is not None else None
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.25, random_state=SEED, stratify=stratify_val
    )

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    df_train["label_enc"] = label_encoder.fit_transform(df_train["labels"])
    df_val["label_enc"] = label_encoder.transform(df_val["labels"])
    df_test["label_enc"] = label_encoder.transform(df_test["labels"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    def compute_embeddings(texts, batch_size=32):
        """Process embeddings in batches to avoid memory issues"""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                encoded = tokenizer(
                    batch,
                    max_length=MAX_SEQ_LEN,
                    add_special_tokens=True,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ).to(device)

                with torch.no_grad():
                    outputs = triplet_model(
                        encoded["input_ids"], encoded["attention_mask"]
                    )

                normalized = F.normalize(outputs, p=2, dim=1)
                all_embeddings.extend(normalized.cpu().numpy().tolist())
            except Exception as e:
                print(f"Error processing batch: {e}")
                # Add zero embeddings as fallback
                embedding_dim = triplet_model.base_model.config.hidden_size
                all_embeddings.extend([([0.0] * embedding_dim)] * len(batch))

        return all_embeddings

    # Create output directory if it doesn't exist
    for path in [
        os.path.dirname(TRAIN_CSV),
        os.path.dirname(VAL_CSV),
        os.path.dirname(TEST_CSV),
    ]:
        os.makedirs(path, exist_ok=True)

    # Compute embeddings for each split
    print(f"Computing embeddings for {len(df_train)} training samples...")
    df_train["embedding"] = compute_embeddings(df_train["text"].tolist())
    print(f"Computing embeddings for {len(df_val)} validation samples...")
    df_val["embedding"] = compute_embeddings(df_val["text"].tolist())
    print(f"Computing embeddings for {len(df_test)} test samples...")
    df_test["embedding"] = compute_embeddings(df_test["text"].tolist())

    # Save files
    print("Saving embedding files...")
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    print("Embedding CSV data generated and saved.")
    return True


class EmbeddingDataset(Dataset):
    """Dataset for handling precomputed embeddings loaded from a CSV file."""

    def __init__(self, csv_file, device):
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.df = pd.read_csv(csv_file)

        try:
            self.df["embedding"] = self.df["embedding"].apply(eval)
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Error parsing embeddings in {csv_file}: {e}")

        try:
            self.embeddings = torch.tensor(
                self.df["embedding"].tolist(), dtype=torch.float, device=device
            )
            self.labels = torch.tensor(
                self.df["label_enc"].tolist(), dtype=torch.long, device=device
            )
        except Exception as e:
            raise ValueError(f"Error converting embeddings to tensors: {e}")

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_embedding_data(device):
    """Load embedding data from CSV files, regenerating if needed"""
    # Check for missing files
    missing_files = []
    for csv_file in [TRAIN_CSV, VAL_CSV, TEST_CSV]:
        if not os.path.exists(csv_file):
            missing_files.append(csv_file)

    # If any files are missing, try to regenerate them
    if missing_files:
        print(f"Error: Missing embedding files: {', '.join(missing_files)}")
        print("Regenerating embedding files from source data...")
        triplet_model = load_triplet_model(device)
        success = generate_embedding_csv_data(triplet_model, device)

        if not success:
            raise FileNotFoundError("Failed to generate embedding files")

    # Load the embedding data
    try:
        train_ds = EmbeddingDataset(TRAIN_CSV, device)
        val_ds = EmbeddingDataset(VAL_CSV, device)
        test_ds = EmbeddingDataset(TEST_CSV, device)

        from sklearn.preprocessing import LabelEncoder

        df_train = pd.read_csv(TRAIN_CSV)
        label_encoder = LabelEncoder()
        label_encoder.fit(df_train["labels"])
        global NUM_LABEL
        NUM_LABEL = len(label_encoder.classes_)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        print(f"Loaded embedding data successfully: {NUM_LABEL} classes")
        return train_loader, val_loader, test_loader, label_encoder
    except Exception as e:
        print(f"Error loading embedding data: {e}")
        raise


def validate(model, dataloader, criterion):
    """Validate model on validation set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data, batch_labels in tqdm(
            dataloader, desc="Validating", leave=False
        ):
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(batch_labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    val_loss = total_loss / len(dataloader)
    val_acc = (all_preds == all_labels).mean()
    return val_loss, val_acc, 0.0  # F1 score can be implemented if required
