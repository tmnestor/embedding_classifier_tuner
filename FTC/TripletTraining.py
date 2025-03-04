#!/usr/bin/env python3
# Standard library imports
import argparse
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

# Add the parent directory (project root) to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Third-party imports
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Local imports
from FTC_utils.logging_utils import tee_to_file

# Start capturing output to log file
log_file = tee_to_file("TripletTraining")

# Handle import with careful error handling
try:
    # Try direct import
    # Import and register the YAML constructor
    from FTC_utils.LoaderSetup import join_constructor, env_var_or_default_constructor
    from FTC_utils.utils import preprocess_text
    from FTC_utils.env_utils import check_environment

    # Register the YAML constructors
    yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor(
        "!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader
    )

    # Check environment and create necessary directories
    check_environment()

    print("Successfully imported from utils modules")
except ImportError as e:
    print(f"Error importing from utils: {e}")

# Load configuration with proper error handling
current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = Path(os.path.dirname(current_dir)) / "config.yml"
try:
    from FTC_utils.file_utils import load_config

    config = load_config(str(config_path))
except ImportError:
    # Fallback if file_utils is not available
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading configuration from {config_path}: {e}")
        sys.exit(1)

# Constants
SEED = config["SEED"]
BATCH_SIZE = config["BATCH_SIZE"]  # Use the value from config.yml
MAX_SEQ_LEN = config["MAX_SEQ_LEN"]
MODEL_NAME = config["MODEL_NAME"]
# Get explicit model path if available, otherwise None
MODEL_PATH = config.get("MODEL_PATH", None)
CSV_PATH = config["CSV_PATH"]
TRAIN_CSV = config["TRAIN_CSV"]
VAL_CSV = config["VAL_CSV"]
TEST_CSV = config["TEST_CSV"]
MARGIN = 1.0

# Set seeds for reproducibility
random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print(f"[DEBUG] Loading model '{MODEL_NAME}' in TripletTraining.py")
print(f"[DEBUG] MODEL_NAME type: {type(MODEL_NAME)}")
print(f"[DEBUG] AutoTokenizer imported from: {AutoTokenizer.__module__}")
print(f"[DEBUG] AutoModel imported from: {AutoModel.__module__}")

try:
    # If MODEL_PATH is explicitly set, use it
    if MODEL_PATH:
        print(f"Using explicit model path: {MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        base_model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
    else:
        # Otherwise use the model name
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
except Exception as e:
    print(f"ERROR loading model: {type(e).__name__}: {e}")
    raise

base_model.eval()

# Regex for text preprocessing
TAG_RE = re.compile(r"<[^>]+>")


class TripletDataset(Dataset):
    def __init__(self, df):
        """
        Initialize dataset with a DataFrame instead of loading directly from CSV
        to support proper train/test split.

        Args:
            df: DataFrame containing text data with 'text' and 'labels' columns
        """
        # Remove rows where "text" is missing or not a proper string
        df = df[df["text"].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        self.df = df  # Store the filtered dataframe
        df["text"] = df["text"].apply(preprocess_text)
        # Group records by labels and then select the 'text' column
        self.data = (
            df.groupby("labels")["text"].apply(lambda x: list(x.values)).to_dict()
        )
        self.labels = list(self.data.keys())

        # Store processed texts and labels for indexed access
        self.texts = df["text"].tolist()
        self.text_labels = df["labels"].tolist()

        # Ensure there are at least two samples per class for positive sampling
        for label, samples in self.data.items():
            if len(samples) == 1:
                # If there's only one sample in a class, duplicate it
                self.data[label].append(samples[0])

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a triplet (anchor, positive, negative) sample for training.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: Tuple containing (anchor_sample, positive_sample, negative_sample) where:
                - anchor_sample (str): Text from the dataset at the given index
                - positive_sample (str): Text from the same class as the anchor
                - negative_sample (str): Text from a different class than the anchor
        """
        # Use the specific text at index idx as the anchor
        # This ensures all texts are used as anchors during training
        anchor_sample = self.texts[idx]
        anchor_label = self.text_labels[idx]

        # Get a positive sample from the same class (different from anchor if possible)
        same_class_samples = [s for s in self.data[anchor_label] if s != anchor_sample]
        if same_class_samples:
            # If there are other samples in the same class, pick one randomly
            pos_sample = random.choice(same_class_samples)
        else:
            # If this is the only sample in its class, use itself
            pos_sample = anchor_sample

        # Get a negative sample from a different class
        other_labels = [l for l in self.labels if l != anchor_label]
        if other_labels:
            # If there are other classes, pick one randomly
            neg_label = random.choice(other_labels)
            neg_sample = random.choice(self.data[neg_label])
        else:
            # If there's only one class, pick any other sample as negative
            other_samples = [s for s in self.texts if s != anchor_sample]
            neg_sample = (
                random.choice(other_samples) if other_samples else anchor_sample
            )

        return anchor_sample, pos_sample, neg_sample


def collate_fn(batch):
    """
    Collate function for DataLoader that tokenizes triplets of text samples.
    
    Args:
        batch (list): List of triplets (anchor, positive, negative) from the dataset
        
    Returns:
        tuple: Tuple containing (anchor_enc, pos_enc, neg_enc) where each is a dict with tokenized outputs:
            - input_ids: Tensor of token ids
            - attention_mask: Tensor of attention masks
            - token_type_ids: Tensor of token type ids (if applicable)
    """
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
    """
    Neural network model that generates normalized embeddings from input text using a base model.
    Uses mean pooling on top of the base model's last hidden states for the embeddings.
    
    Attributes:
        base_model: Pre-trained transformer model to generate embeddings
    """
    
    def __init__(self, base_model):
        """
        Initialize TripletEmbeddingModel with a base transformer model.
        
        Args:
            base_model (transformers.PreTrainedModel): Pre-trained transformer model
        """
        super(TripletEmbeddingModel, self).__init__()
        self.base_model = base_model

    def _mean_pooled(self, last_hidden_states, attention_mask):
        """
        Apply mean pooling to the last hidden states, using attention mask to ignore padding tokens.
        
        Args:
            last_hidden_states (torch.Tensor): Last hidden states from the transformer model
                                              [batch_size, seq_len, hidden_size]
            attention_mask (torch.Tensor): Attention mask for the sequence [batch_size, seq_len]
            
        Returns:
            torch.Tensor: Mean-pooled representation [batch_size, hidden_size]
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.
        
        Args:
            input_ids (torch.Tensor): Tensor of token ids [batch_size, seq_len]
            attention_mask (torch.Tensor): Tensor of attention masks [batch_size, seq_len]
            
        Returns:
            torch.Tensor: L2-normalized embeddings [batch_size, hidden_size]
        """
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = self._mean_pooled(last_hidden_state, attention_mask)
        normalized_output = F.normalize(pooled_output, p=2, dim=1)
        return normalized_output


def generate_embeddings(df, model, tokenizer, device, max_seq_len=128):
    """
    Generate embeddings for visualization.
    
    Args:
        df (pd.DataFrame): DataFrame containing text data with 'text' and 'labels' columns
        model (nn.Module): Model to generate embeddings
        tokenizer: Tokenizer to preprocess text data
        device (torch.device): Device to run the model on
        max_seq_len (int, optional): Maximum sequence length for tokenization. Defaults to 128.
        
    Returns:
        tuple: Tuple containing:
            - all_embeddings (np.ndarray): Array of embeddings [n_samples, embedding_dim]
            - encoded_labels (np.ndarray): Array of encoded label indices [n_samples]
            - label_texts (np.ndarray): Array of label class names
    """
    model.eval()

    # Encode the texts
    texts = df["text"].tolist()
    labels = df["labels"].tolist()

    # Create label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    label_texts = label_encoder.classes_

    # Generate embeddings
    embeddings = []
    batch_size = 8

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_texts = [preprocess_text(text) for text in batch_texts]

            inputs = tokenizer(
                batch_texts,
                padding="max_length",
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            ).to(device)

            # Different handling for different model types
            if isinstance(model, TripletEmbeddingModel):
                # Direct embedding for TripletEmbeddingModel
                batch_embeddings = model(inputs["input_ids"], inputs["attention_mask"])
            else:
                # For base model, handle the different output structure
                outputs = model(inputs["input_ids"], inputs["attention_mask"])
                # Mean pooling for base model output
                last_hidden = outputs.last_hidden_state
                attention_mask = inputs["attention_mask"]
                # Apply mean pooling
                batch_embeddings = last_hidden.masked_fill(
                    ~attention_mask[..., None].bool(), 0.0
                )
                batch_embeddings = (
                    batch_embeddings.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                )

            # Normalize embeddings
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            embeddings.append(batch_embeddings.cpu().numpy())

    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings, axis=0)

    return all_embeddings, encoded_labels, label_texts


def visualize_embeddings(
    before_embeddings, after_embeddings, labels, label_texts, config
):
    """
    Visualize embeddings before and after triplet training using t-SNE.
    
    Args:
        before_embeddings (np.ndarray): Embeddings before training [n_samples, embedding_dim]
        after_embeddings (np.ndarray): Embeddings after training [n_samples, embedding_dim]
        labels (np.ndarray): Array of encoded label indices [n_samples]
        label_texts (np.ndarray): Array of label class names
        config (dict): Configuration dictionary with visualization settings
        
    Returns:
        str: Path to the saved visualization file
    """
    print("\nReducing dimensionality for visualization...")

    # Reduce dimensionality with t-SNE
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(before_embeddings) - 1)
    )
    before_2d = tsne.fit_transform(before_embeddings)

    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(after_embeddings) - 1)
    )
    after_2d = tsne.fit_transform(after_embeddings)

    # Create figure with two subplots
    plt.figure(figsize=(16, 8))

    # Plot Before
    ax1 = plt.subplot(1, 2, 1)
    scatter = ax1.scatter(
        before_2d[:, 0], before_2d[:, 1], c=labels, cmap="viridis", alpha=0.8
    )
    ax1.set_title("Before Triplet Training", fontsize=14)
    ax1.set_xlabel("t-SNE dimension 1")
    ax1.set_ylabel("t-SNE dimension 2")

    # Plot After
    ax2 = plt.subplot(1, 2, 2)
    scatter = ax2.scatter(
        after_2d[:, 0], after_2d[:, 1], c=labels, cmap="viridis", alpha=0.8
    )
    ax2.set_title("After Triplet Training", fontsize=14)
    ax2.set_xlabel("t-SNE dimension 1")
    ax2.set_ylabel("t-SNE dimension 2")

    # Add legend
    legend1 = ax2.legend(
        scatter.legend_elements()[0], label_texts, title="Categories", loc="upper right"
    )
    ax2.add_artist(legend1)

    plt.tight_layout()

    # Create visualization directory using config and pathlib for safe path handling
    if "LEARNING_CURVES_DIR" in config:
        viz_dir = config["LEARNING_CURVES_DIR"]
    else:
        viz_dir = str(Path(config["BASE_ROOT"]) / "visualizations")

    # Make sure the directory exists using pathlib
    Path(viz_dir).mkdir(parents=True, exist_ok=True)

    # Create filename for the visualization using pathlib
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(
        Path(viz_dir) / f"triplet_embeddings_visualization_{timestamp}.png"
    )

    # Save the visualization
    plt.savefig(output_path)
    print(f"Saved visualization to: {output_path}")
    plt.close()

    return output_path


def train_triplet(
    epochs=5,
    lr=2e-5,
    visualize=False,
    batch_size=None,
    gradient_accumulation_steps=1,
    test_size=0.2,
):  # Add batching options
    """
    Train a triplet embedding model with proper train/test split to prevent data leakage.

    Args:
        epochs: Number of training epochs
        lr: Learning rate
        visualize: Whether to generate embedding visualizations
        batch_size: Batch size for training (uses config default if None)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        test_size: Fraction of data to use for test/visualization (default 0.2)
    """
    # Load and preprocess data
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)

    # Create consistent train/val/test split to prevent data leakage

    # Use stratified split if possible (requires multiple classes)
    # First split off test data (20%)
    if len(df["labels"].unique()) > 1:
        df_train_val, df_test = train_test_split(
            df, test_size=0.2, random_state=SEED, stratify=df["labels"]
        )
        # Then split train_val into train (80% of train_val = 64% of total) and val (20% of train_val = 16% of total)
        df_train, df_val = train_test_split(
            df_train_val, test_size=0.2, random_state=SEED, stratify=df_train_val["labels"]
        )
        print(
            f"Created stratified train/val/test split: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test samples"
        )
    else:
        # Fallback to random split if only one class
        df_train_val, df_test = train_test_split(df, test_size=0.2, random_state=SEED)
        df_train, df_val = train_test_split(df_train_val, test_size=0.2, random_state=SEED)
        print(
            f"Created random train/val/test split: {len(df_train)} train, {len(df_val)} val, {len(df_test)} test samples"
        )
    
    # Save splits to ensure consistency across pipeline
    # This prevents data leakage between TripletTraining and BertClassification
    print(f"Saving data splits to {TRAIN_CSV}, {VAL_CSV}, and {TEST_CSV}...")
    # Ensure the data directory exists
    Path(os.path.dirname(TRAIN_CSV)).mkdir(parents=True, exist_ok=True)
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)

    # Create training dataset using only training data
    train_dataset = TripletDataset(df_train)

    # Use provided batch size or default from config/constants
    actual_batch_size = batch_size or BATCH_SIZE
    print(f"Using batch size: {actual_batch_size}")

    if gradient_accumulation_steps > 1:
        print(f"Using gradient accumulation with {gradient_accumulation_steps} steps")
        print(
            f"Effective batch size: {actual_batch_size * gradient_accumulation_steps}"
        )

    # Create data loader for training data only
    train_dataloader = DataLoader(
        train_dataset, batch_size=actual_batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Initialize model
    model = TripletEmbeddingModel(base_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)

    # Generate "before training" embeddings if visualization is requested
    if visualize:
        print("\nGenerating embeddings before triplet training...")
        # Use the test set for visualization to prevent data leakage
        # Use the base model for "before" embeddings
        before_embeddings, labels, label_texts = generate_embeddings(
            df_test, base_model, tokenizer, device, MAX_SEQ_LEN
        )

    # Train the model using only training data
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0

        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False
        )

        # Initialize gradient accumulation
        optimizer.zero_grad()
        batch_count = 0

        for batch_idx, (anchor_enc, pos_enc, neg_enc) in enumerate(progress_bar):
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

            # Scale loss for gradient accumulation
            loss = criterion(anchor_emb, pos_emb, neg_emb)
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()

            # Only update weights after accumulating gradients
            batch_count += 1
            if (
                batch_count % gradient_accumulation_steps == 0
                or batch_idx == len(train_dataloader) - 1
            ):
                optimizer.step()
                optimizer.zero_grad()

            # Track loss (scale back if using accumulation)
            if gradient_accumulation_steps > 1:
                epoch_loss += loss.item() * gradient_accumulation_steps
            else:
                epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Clear CUDA cache periodically for large datasets
            if torch.cuda.is_available() and batch_idx % 100 == 0 and batch_idx > 0:
                torch.cuda.empty_cache()

        # Print epoch summary
        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_dataloader):.4f}"
        )

    # Save the learned model with proper error handling
    try:
        # Try to use the file_utils function first
        from FTC_utils.file_utils import save_model_safely, ensure_dir

        # Ensure checkpoint directory exists
        checkpoint_dir = ensure_dir(config["CHECKPOINT_DIR"])
        model_save_path = str(Path(checkpoint_dir) / "triplet_model.pt")

        # Save model with additional training info
        additional_data = {
            "epochs": epochs,
            "lr": lr,
            "timestamp": datetime.now().isoformat(),
            "model_name": MODEL_NAME,
            "batch_size": BATCH_SIZE,
            "train_samples": len(df_train),
            "test_samples": len(df_test),
            "seed": SEED,
        }

        save_success = save_model_safely(model, model_save_path, additional_data)
        if not save_success:
            print("Warning: Failed to save model with enhanced error handling")
            # Fallback to basic saving
            torch.save(model.state_dict(), model_save_path)

    except ImportError:
        # Fallback to basic path and error handling
        try:
            # Ensure directory exists
            Path(config["CHECKPOINT_DIR"]).mkdir(parents=True, exist_ok=True)
            model_save_path = str(Path(config["CHECKPOINT_DIR"]) / "triplet_model.pt")
            torch.save(model.state_dict(), model_save_path)
        except Exception as e:
            print(f"Error saving model: {e}")
            print("Training completed but model could not be saved.")

    # Generate "after training" embeddings and visualize if requested
    if visualize:
        print("\nGenerating embeddings after triplet training...")
        # Use the test set for visualization (unseen during training)
        after_embeddings, _, _ = generate_embeddings(
            df_test, model, tokenizer, device, MAX_SEQ_LEN
        )

        # Visualize the embeddings (using test data only)
        viz_path = visualize_embeddings(
            before_embeddings, after_embeddings, labels, label_texts, config
        )
        print(f"Embedding visualization saved to: {viz_path}")

    # Add message showing where the model was saved
    print("\n" + "=" * 70)
    print("TRIPLET MODEL TRAINING COMPLETE")
    print("=" * 70)
    print(f"Model saved to: {Path(model_save_path).resolve()}")
    print(f"Model type: {model.__class__.__name__}")
    print(f"Base model: {MODEL_NAME}")
    print(f"Input dimension: {base_model.config.hidden_size}")
    print(f"Output embedding dimension: {base_model.config.hidden_size}")
    print(f"Training data size: {len(df_train)} samples")
    print(f"Test data size (not used in training): {len(df_test)} samples")
    print("=" * 70)
    print("Next steps:")
    print("1. Run BertClassification.py to generate embeddings and train classifier")
    print("   python FTC/BertClassification.py")
    print("2. Or run TuneBert.py to optimize classifier architecture")
    print("   python FTC/TuneBert.py")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train BERT embeddings using triplet loss"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument(
        "--visualize", action="store_true", help="Generate embedding visualization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=f"Batch size for training (default: {BATCH_SIZE} from config.yml)",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients (default: 1)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to hold out for testing (default: 0.2)",
    )
    args = parser.parse_args()

    # Display information about memory-efficient processing
    if args.batch_size or args.gradient_accumulation > 1:
        print("\nMemory-efficient training configuration:")
        if args.batch_size:
            print(f"- Using custom batch size: {args.batch_size}")
        if args.gradient_accumulation > 1:
            print(f"- Using gradient accumulation: {args.gradient_accumulation} steps")
            effective_batch = (
                args.batch_size or BATCH_SIZE
            ) * args.gradient_accumulation
            print(f"- Effective batch size: {effective_batch}")

    # Display data split information
    print("\nData split configuration:")
    print(f"- Training set: {100 * (1 - args.test_size):.1f}% of data")
    print(f"- Test set: {100 * args.test_size:.1f}% of data")
    print(f"- Random seed: {SEED} (for reproducibility)")

    train_triplet(
        epochs=args.epochs,
        lr=args.lr,
        visualize=args.visualize,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        test_size=args.test_size,
    )

    # Print log file location at the end
    print(f"\nExecution log saved to: {log_file}")
