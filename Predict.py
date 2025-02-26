import os
import yaml
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from utils.utils import get_default_device, preprocess_text

# from utils.LoaderSetup import join_constructor
from BertClassification import (
    load_triplet_model,
    FFModel,
    BertClassifier,
    DenseBlock,
)  # Reuse existing model definitions


# New on-the-fly prediction dataset
class PredictionDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = preprocess_text(self.texts[idx])
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        # Remove extra batch dimension
        return {k: v.squeeze(0) for k, v in encoded.items()}


def main():
    # Load configuration
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Paths and constants
    DATA_ROOT = config["DATA_PATH"]
    TEST_FILE = os.path.join(DATA_ROOT, "testing_data.csv")
    TRAIN_FILE = config["TRAIN_CSV"]
    CHECKPOINT_DIR = config["CHECKPOINT_DIR"]
    EMBEDDING_PATH = config["EMBEDDING_PATH"]
    BEST_CONFIG_DIR = config["BEST_CONFIG_DIR"]
    STUDY_NAME = config["STUDY_NAME"]

    # Get device and tokenizer
    DEVICE = get_default_device()
    tokenizer = AutoTokenizer.from_pretrained(config["MODEL"])

    # Load triplet model
    triplet_model = load_triplet_model(DEVICE)

    # Load best configuration from BEST_CONFIG_DIR in the config
    best_config_file = os.path.join(BEST_CONFIG_DIR, f"best_config_{STUDY_NAME}.yml")
    best_config = {}

    if os.path.exists(best_config_file):
        with open(best_config_file, "r", encoding="utf-8") as f:
            best_config = yaml.safe_load(f)
        print(f"Loaded best configuration from: {best_config_file}")
    else:
        print("No best configuration found. Using default parameters.")

    # Re-create classifier architecture using best config or defaults.
    classifier_dropout = best_config.get("dropout_rate", config["DROP_OUT"])
    classifier_activation = best_config.get("activation", config["ACT_FN"])

    # Re-fit label encoder on training data to obtain class mapping
    train_df = pd.read_csv(TRAIN_FILE)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["labels"])
    NUM_LABEL = len(label_encoder.classes_)

    # Determine classifier input dimension from triplet model.
    classifier_input_size = triplet_model.base_model.config.hidden_size

    # Check if we should create a tuned model architecture or use the default FFModel
    if "n_hidden" in best_config and "hidden_units_0" in best_config:
        print("Using tuned architecture from best configuration")
        # Create a dynamic model based on the tuned architecture parameters
        layers = []
        prev_width = classifier_input_size
        n_hidden = best_config.get(
            "n_hidden", 2
        )  # Default to 2 hidden layers if not specified

        # Build the hidden layers based on best_config
        for i in range(n_hidden):
            width_key = f"hidden_units_{i}"
            if width_key in best_config:
                width = best_config[width_key]
            else:
                width = prev_width // 2  # Default to halving the previous width

            layers.append(
                DenseBlock(prev_width, width, classifier_dropout, classifier_activation)
            )
            prev_width = width

        # Add the final output layer
        layers.append(nn.Linear(prev_width, NUM_LABEL))

        # Create the classifier as a sequential model
        classifier = nn.Sequential(*layers)
        print(f"Created tuned classifier architecture with {n_hidden} hidden layers")
    else:
        # Use the default FFModel architecture
        print("Using default FFModel architecture")
        classifier = FFModel(
            classifier_input_size,
            NUM_LABEL,
            classifier_dropout,
            activation_fn=classifier_activation,
        )

    # Create the full model with the classifier
    model = BertClassifier(triplet_model, classifier).to(DEVICE)

    # Load best classifier checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model", "best_model.pt")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded classifier from {checkpoint_path}")
    else:
        raise FileNotFoundError("Best model checkpoint not found.")

    model.eval()

    # Print a summary of the model architecture and parameters
    print("\n" + "=" * 80)
    print("MODEL VERIFICATION")
    print("=" * 80)
    print(f"Pretrained Model: {config['MODEL_NAME']}")
    print(f"Architecture:")
    print(f"  - Embedding Model: {triplet_model.__class__.__name__}")
    print(f"  - Classifier: {classifier.__class__.__name__}")
    print(f"    - Input Dimension: {classifier_input_size}")
    print(f"    - Output Classes: {NUM_LABEL}")
    print(f"    - Dropout Rate: {classifier_dropout}")
    print(f"    - Activation Function: {classifier_activation}")
    print(f"File Locations:")
    print(f"  - Model Loaded From: {checkpoint_path}")
    print(f"  - Embeddings Model: {EMBEDDING_PATH}")
    print(f"  - Training Data: {TRAIN_FILE}")
    print(f"  - Test File: {TEST_FILE}")
    print("=" * 80)

    # Read testing data into DataFrame
    df_test = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(df_test)} samples from {TEST_FILE}")

    # Create DataLoader using on-the-fly tokenization.
    pred_dataset = PredictionDataset(
        df_test["text"].tolist(), tokenizer, config["MAX_SEQ_LEN"]
    )
    pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)

    all_preds = []
    with torch.no_grad():
        for batch in pred_loader:
            # Move tokens to DEVICE
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            # Pass through base model and classifier in one go.
            # If your BERT_classifier is set to accept tokenized inputs:
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)

    # Map prediction indices back to labels.
    predicted_labels = label_encoder.inverse_transform(all_preds)
    df_test["predicted_label"] = predicted_labels

    df_test.to_csv(TEST_FILE, index=False)
    print(f"Predictions saved to {TEST_FILE}")

    return df_test


if __name__ == "__main__":
    main()
