import os

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
    base_model = AutoModel.from_pretrained(MODEL).to(device)
    triplet_model = TripletEmbeddingModel(base_model).to(device)
    triplet_model.load_state_dict(torch.load(EMBEDDING_PATH, map_location=device))
    triplet_model.eval()
    return triplet_model


def generate_embedding_csv_data(triplet_model, device):
    df = pd.read_csv(CSV_PATH)
    df["text"] = df["text"].apply(preprocess_text)
    from sklearn.model_selection import train_test_split

    df_train_val, df_test = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["labels"]
    )
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.25, random_state=SEED, stratify=df_train_val["labels"]
    )
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    df_train["label_enc"] = label_encoder.fit_transform(df_train["labels"])
    df_val["label_enc"] = label_encoder.transform(df_val["labels"])
    df_test["label_enc"] = label_encoder.transform(df_test["labels"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

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
            outputs = triplet_model(encoded["input_ids"], encoded["attention_mask"])
        normalized = F.normalize(outputs, p=2, dim=1)
        return normalized.cpu().numpy().tolist()

    df_train["embedding"] = compute_embeddings(df_train["text"].tolist())
    df_val["embedding"] = compute_embeddings(df_val["text"].tolist())
    df_test["embedding"] = compute_embeddings(df_test["text"].tolist())
    os.makedirs(os.path.dirname(TRAIN_CSV), exist_ok=True)
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    print("Embedding CSV data generated and saved.")


class EmbeddingDataset(Dataset):
    """Dataset for handling precomputed embeddings loaded from a CSV file."""

    def __init__(self, csv_file, device):
        self.df = pd.read_csv(csv_file)
        self.df["embedding"] = self.df["embedding"].apply(eval)
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


def load_embedding_data(device):
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
    return train_loader, val_loader, test_loader, label_encoder


def validate(model, dataloader, criterion):
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
