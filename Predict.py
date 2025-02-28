import os
import yaml
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from utils.device_utils import get_device
from utils.LoaderSetup import join_constructor

# Register YAML constructor
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)

# Import shared components to ensure consistency
from BertClassification import (
    load_triplet_model,
    create_classifier_from_config,
    BertClassifier,
    preprocess_text,
)


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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--input", type=str, help="Path to input CSV file")
    parser.add_argument(
        "--output", type=str, help="Path to output predictions CSV file"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed model information"
    )
    parser.add_argument(
        "--config", type=str, help="Path to specific config file (overrides default)"
    )
    args = parser.parse_args()

    # Load configuration
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Device setup
    device = get_device()
    print(f"Using device: {device}")

    # Paths and constants
    DATA_ROOT = config["DATA_PATH"]
    TEST_FILE = (
        args.input if args.input else os.path.join(DATA_ROOT, "testing_data.csv")
    )
    OUTPUT_FILE = args.output if args.output else TEST_FILE
    TRAIN_FILE = config["TRAIN_CSV"]
    CHECKPOINT_DIR = config["CHECKPOINT_DIR"]
    EMBEDDING_PATH = config["EMBEDDING_PATH"]
    BEST_CONFIG_DIR = config["BEST_CONFIG_DIR"]
    STUDY_NAME = config["STUDY_NAME"]
    MAX_SEQ_LEN = config["MAX_SEQ_LEN"]

    # Load tokenizer
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL"])

    # Load triplet model
    triplet_model = load_triplet_model(device)
    classifier_input_size = triplet_model.base_model.config.hidden_size

    # Get class mapping from training data
    train_df = pd.read_csv(TRAIN_FILE)
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df["labels"])
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} classes: {label_encoder.classes_}")

    # Load the best configuration file
    best_config_file = (
        args.config
        if args.config
        else os.path.join(BEST_CONFIG_DIR, f"best_config_{STUDY_NAME}.yml")
    )

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

    # Load model weights
    checkpoint_path = os.path.join(CHECKPOINT_DIR, "model", "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading model weights from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Model loaded successfully!")
    except RuntimeError as e:
        print(f"ERROR: Failed to load model weights: {e}")
        print("\nArchitecture mismatch between saved weights and configuration.")
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

    # Set model to evaluation mode
    model.eval()

    # Load test data
    df_test = pd.read_csv(TEST_FILE)
    print(f"Loaded {len(df_test)} samples from {TEST_FILE}")

    # Create DataLoader for prediction
    pred_dataset = PredictionDataset(df_test["text"].tolist(), tokenizer, MAX_SEQ_LEN)
    pred_loader = DataLoader(pred_dataset, batch_size=32, shuffle=False)

    # Make predictions
    all_preds = []
    with torch.no_grad():
        for batch in pred_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(batch)
            preds = torch.argmax(logits, dim=1).cpu().numpy().tolist()
            all_preds.extend(preds)

    # Convert predictions to labels
    predicted_labels = label_encoder.inverse_transform(all_preds)
    df_test["predicted_label"] = predicted_labels

    # Save results
    df_test.to_csv(OUTPUT_FILE, index=False)
    print(f"Predictions saved to {OUTPUT_FILE}")

    # Show sample predictions
    print("\nSample predictions:")
    for i in range(min(5, len(df_test))):
        text = (
            df_test["text"].iloc[i][:50] + "..."
            if len(df_test["text"].iloc[i]) > 50
            else df_test["text"].iloc[i]
        )
        pred = df_test["predicted_label"].iloc[i]
        print(f"{i + 1}. {text} → {pred}")

    return df_test


if __name__ == "__main__":
    main()
