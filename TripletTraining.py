import argparse
import os
import random
import re
import sys

import yaml

# Add the current directory to sys.path to ensure module imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

# Handle import with careful error handling
try:
    # Try direct import
    # Import and register the YAML constructor
    from utils.LoaderSetup import join_constructor
    from utils.utils import preprocess_text

    # Register the YAML constructor
    yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)

    print("Successfully imported from utils modules")
except ImportError as e:
    print(f"Error importing from utils: {e}")

    # # Define fallback functions if imports fail
    # def preprocess_text(text):
    #     if not isinstance(text, str):
    #         return ""
    #     TAG_RE = re.compile(r"<[^>]+>")
    #     text = TAG_RE.sub("", text)
    #     text = " ".join(text.split())
    #     return text

    # def join_constructor(loader, node):
    #     seq = loader.construct_sequence(node)
    #     return "".join(seq)

    # # Register YAML constructor
    # yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
    # print("Using fallback implementations for utils functions")

# Load configuration with absolute path reference
config_path = os.path.join(current_dir, "config.yml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
    # Remove the pprint(config) call

# Constants
SEED = config["SEED"]
BATCH_SIZE = 32  # You can also add this to config if desired.
MAX_SEQ_LEN = config["MAX_SEQ_LEN"]
MODEL_NAME = config["MODEL_NAME"]
CSV_PATH = config["CSV_PATH"]
MARGIN = 1.0

# Set seeds for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
base_model.eval()

# Regex for text preprocessing
TAG_RE = re.compile(r"<[^>]+>")


class TripletDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        # Remove rows where "text" is missing or not a proper string
        df = df[df["text"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        self.df = df  # Store the filtered dataframe
        df["text"] = df["text"].apply(preprocess_text)
        # Group records by labels and then select the 'text' column
        self.data = (
            df.groupby("labels")["text"].apply(lambda x: list(x.values)).to_dict()
        )
        self.labels = list(self.data.keys())

    def __len__(self):
        # Return the size of the original dataframe
        return len(self.df)

    def __getitem__(self, idx):
        # Sample anchor and positive from same label and negative from a different label.
        anchor_label = random.choice(self.labels)
        anchor_sample = random.choice(self.data[anchor_label])
        pos_sample = random.choice(self.data[anchor_label])
        while pos_sample == anchor_sample:
            pos_sample = random.choice(self.data[anchor_label])
        neg_label = random.choice(self.labels)
        while neg_label == anchor_label:
            neg_label = random.choice(self.labels)
        neg_sample = random.choice(self.data[neg_label])
        return anchor_sample, pos_sample, neg_sample


def collate_fn(batch):
    anchors, positives, negatives = zip(*batch)

    def tokenize_texts(texts):
        return tokenizer(
            list(texts),
            max_length=MAX_SEQ_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    anchor_enc = tokenize_texts(anchors)
    pos_enc = tokenize_texts(positives)
    neg_enc = tokenize_texts(negatives)
    return anchor_enc, pos_enc, neg_enc


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


def train_triplet(epochs=5, lr=2e-5):  # Add epochs and lr as arguments
    dataset = TripletDataset(CSV_PATH)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )

    model = TripletEmbeddingModel(base_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)  # Use lr argument
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    model.train()
    for epoch in range(epochs):  # Use epochs argument
        epoch_loss = 0.0
        for anchor_enc, pos_enc, neg_enc in dataloader:
            # Move inputs to device
            anchor_ids = anchor_enc["input_ids"].to(device)
            anchor_mask = anchor_enc["attention_mask"].to(device)
            pos_ids = pos_enc["input_ids"].to(device)
            pos_mask = pos_enc["attention_mask"].to(device)
            neg_ids = neg_enc["input_ids"].to(device)
            neg_mask = neg_enc["attention_mask"].to(device)

            # Forward pass for triplet components
            anchor_emb = model(anchor_ids, anchor_mask)
            pos_emb = model(pos_ids, pos_mask)
            neg_emb = model(neg_ids, neg_mask)

            loss = criterion(anchor_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}"
        )  # Use epochs argument

    # Save the learned model
    os.makedirs(config["CHECKPOINT_DIR"], exist_ok=True)
    model_save_path = os.path.join(config["CHECKPOINT_DIR"], "triplet_model.pt")
    torch.save(model.state_dict(), model_save_path)

    # Add message showing where the model was saved
    print("\n" + "=" * 70)
    print("TRIPLET MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {os.path.abspath(model_save_path)}")
    print(f"Model type: {model.__class__.__name__}")
    print(f"Base model: {MODEL_NAME}")
    print(f"Input dimension: {base_model.config.hidden_size}")
    print(f"Output embedding dimension: {base_model.config.hidden_size}")
    print("=" * 70)
    print("Next steps:")
    print("1. Run BertClassification.py to generate embeddings and train classifier")
    print("   python BertClassification.py")
    print("2. Or run TuneBert.py to optimize classifier architecture")
    print("   python TuneBert.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BERT embeddings using triplet loss"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()
    train_triplet(epochs=args.epochs, lr=args.lr)
