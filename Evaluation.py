import os
import ast
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from collections import Counter
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder

from wordcloud import WordCloud
import re
from typing import List, Tuple, Iterator

# Import necessary components from existing modules
from utils.LoaderSetup import join_constructor
from utils.shared import load_triplet_model, validate
from BertClassification import (
    BertClassifier,
    FFModel,
    DenseBlock,
    preprocess_text,
    TokenizedDataset,
)

# Register the YAML constructor
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)


# Custom tokenization and n-gram functions
def simple_tokenize(text: str) -> List[str]:
    """Simple word tokenization function that doesn't require NLTK"""
    # Convert to lowercase and split on non-alphanumeric characters
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def generate_ngrams(tokens: List[str], n: int) -> Iterator[Tuple[str, ...]]:
    """Generate n-grams from a list of tokens without using NLTK"""
    return zip(*[tokens[i:] for i in range(n)])


def get_device():
    """Get the best available device with MPS support for Mac"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


class CrossValidationDataset(Dataset):
    """Dataset for handling text and embeddings during cross-validation"""

    def __init__(self, df, tokenizer=None, device=None, max_seq_len=64):
        self.df = df
        self.device = device if device is not None else torch.device("cpu")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Debug info
        print(
            f"Dataset initialized with: tokenizer={tokenizer is not None}, device={self.device}"
        )

        # If embeddings are available, prepare them for direct use
        if "embedding" in df.columns:
            print("Using embedding column from DataFrame")
            try:
                # First try to handle them as numpy arrays or lists directly
                if isinstance(df["embedding"].iloc[0], (list, np.ndarray)):
                    embeddings = np.stack(df["embedding"].values)
                    self.embeddings = torch.tensor(
                        embeddings, dtype=torch.float, device=self.device
                    )
                # If they're strings, try to use ast.literal_eval
                elif isinstance(df["embedding"].iloc[0], str):
                    self.df["embedding"] = self.df["embedding"].apply(ast.literal_eval)
                    self.embeddings = torch.tensor(
                        self.df["embedding"].tolist(),
                        dtype=torch.float,
                        device=self.device,
                    )
                else:
                    raise ValueError(
                        f"Unknown embedding type: {type(df['embedding'].iloc[0])}"
                    )
            except Exception as e:
                print(f"Error processing embeddings: {e}")
                print(f"Type of first embedding: {type(df['embedding'].iloc[0])}")
                print(
                    f"Sample of first embedding: {str(df['embedding'].iloc[0])[:100]}"
                )
                self.embeddings = None
        else:
            self.embeddings = None
            if tokenizer is None:
                print("WARNING: No embeddings found and no tokenizer provided!")

        # Convert labels
        if "label_enc" in df.columns:
            self.labels = torch.tensor(
                self.df["label_enc"].tolist(), dtype=torch.long, device=self.device
            )
        else:
            self.labels = None
            print("WARNING: No 'label_enc' column found in DataFrame!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # If we have embeddings, return them directly
        if self.embeddings is not None:
            return self.embeddings[idx], self.labels[idx]

        # Otherwise, tokenize on the fly if tokenizer is provided
        if self.tokenizer is not None:
            text = preprocess_text(self.df.iloc[idx]["text"])

            # Debug info for first few items
            if idx < 3:  # Only print for first few items to avoid flooding console
                print(f"Tokenizing example {idx}: '{text[:30]}...'")

            encoded = self.tokenizer(
                text,
                max_length=self.max_seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            # Remove batch dimension
            encoded = {k: v.squeeze(0).to(self.device) for k, v in encoded.items()}
            return encoded, self.labels[idx]

        # Fallback case - if we get here, we have a configuration problem
        raise ValueError(
            "Dataset must have either embeddings or a tokenizer. "
            f"Has embeddings: {self.embeddings is not None}, "
            f"Has tokenizer: {self.tokenizer is not None}"
        )


def create_classifier(input_dim: int, output_dim: int, config: Dict) -> nn.Module:
    """Create a classifier based on configuration"""

    # Get parameters
    dropout_rate = config.get("dropout_rate", 0.3)
    activation_fn = config.get("activation", "gelu")

    # Check if we should create a tuned architecture or use default
    if "n_hidden" in config and "hidden_units_0" in config:
        layers = []
        prev_width = input_dim
        n_hidden = config.get("n_hidden", 2)

        # Build hidden layers
        for i in range(n_hidden):
            width_key = f"hidden_units_{i}"
            width = config.get(width_key, prev_width // 2)
            layers.append(DenseBlock(prev_width, width, dropout_rate, activation_fn))
            prev_width = width

        # Add output layer
        layers.append(nn.Linear(prev_width, output_dim))
        return nn.Sequential(*layers)
    else:
        # Use default architecture
        return FFModel(input_dim, output_dim, dropout_rate, activation_fn=activation_fn)


def get_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer based on configuration"""

    optimizer_name = config.get("optimizer", "adamw")
    lr = float(config.get("lr", 2e-5))
    weight_decay = float(config.get("weight_decay", 0))

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = config.get("momentum", 0.0)
        nesterov = config.get("nesterov", False)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    elif optimizer_name == "rmsprop":
        momentum = config.get("momentum", 0.0)
        alpha = config.get("alpha", 0.99)
        return torch.optim.RMSprop(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=alpha,
        )
    else:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def cross_validate(
    df: pd.DataFrame,
    config: Dict,
    n_folds: int = 5,
    random_state: int = 42,
    use_stratified: bool = True,
    num_epochs: int = 10,  # Increase the default number of epochs
) -> Dict:
    """Perform cross-validation evaluation with improved training"""
    print(f"Starting {n_folds}-fold cross-validation...")
    device = get_device()
    print(f"Using device: {device}")

    # Initialize tokenizer if needed
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config["MODEL"])
    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")

    # Load triplet model
    triplet_model = load_triplet_model(device)
    embedding_dim = triplet_model.base_model.config.hidden_size

    # Load best configuration if available
    best_config_file = os.path.join(
        config.get("BEST_CONFIG_DIR", "best_configs"),
        f"best_config_{config.get('STUDY_NAME', 'default')}.yml",
    )

    if os.path.exists(best_config_file):
        with open(best_config_file, "r") as f:
            best_config = yaml.safe_load(f)
        print(f"Using tuned configuration from {best_config_file}")
    else:
        best_config = {}
        print("No tuned configuration found. Using default parameters.")

    # Encode labels
    label_encoder = LabelEncoder()
    df["label_enc"] = label_encoder.fit_transform(df["labels"])
    num_classes = len(label_encoder.classes_)
    print(f"Found {num_classes} classes: {label_encoder.classes_}")

    # Pre-compute embeddings for the entire dataset if needed
    if "embedding" not in df.columns:
        print("Pre-computing embeddings for all samples to ensure consistency...")
        embeddings = []
        batch_size = 32

        # Create batches for efficient processing
        texts = df["text"].tolist()
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_texts = [preprocess_text(text) for text in batch_texts]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                max_length=config.get("MAX_SEQ_LEN", 64),
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            ).to(device)

            # Generate embeddings
            with torch.no_grad():
                outputs = triplet_model(encoded["input_ids"], encoded["attention_mask"])
                embeddings.extend(outputs.cpu().numpy().tolist())

        # Store embeddings as numpy arrays not strings
        df["embedding"] = embeddings
        print(f"Generated {len(embeddings)} embeddings for dataset")
        print(f"Type of embeddings: {type(embeddings[0])}")

    # Create cross-validator
    if use_stratified:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = cv.split(df, df["label_enc"])
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = cv.split(df)

    # Track metrics across folds
    all_reports = []
    all_predictions = []
    all_true_labels = []
    all_f1_scores = []
    all_precision_scores = []
    all_recall_scores = []
    fold_misclassifications = []

    # Cross-validation loop
    for fold, (train_idx, test_idx) in enumerate(splits):
        print(f"\nFold {fold + 1}/{n_folds}")

        # Split data
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)

        # Create datasets with embeddings
        train_ds = CrossValidationDataset(df_train, device=device)
        test_ds = CrossValidationDataset(df_test, device=device)

        batch_size = config.get("BATCH_SIZE", 32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Make sure label encoding is correct
        num_classes = len(df["labels"].unique())
        print(f"Training with {num_classes} classes, {len(train_ds)} samples")

        # Create classifier
        classifier = create_classifier(embedding_dim, num_classes, best_config)
        model = BertClassifier(triplet_model, classifier).to(device)

        # Configure optimizer and criterion with lr warmup
        optimizer = get_optimizer(model, best_config)
        criterion = nn.CrossEntropyLoss()

        # Add learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

        # Enhanced training
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0

        for epoch in range(num_epochs):
            # Train
            model.train()
            total_loss = 0
            for batch_inputs, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_inputs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Validate
            val_loss, val_acc, _ = validate(model, test_loader, criterion)
            print(
                f"  Epoch {epoch + 1}/{num_epochs}: Train Loss = {total_loss / len(train_loader):.4f}, Val Acc = {val_acc:.4f}"
            )

            # Learning rate scheduling
            lr_scheduler.step(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 5:  # Early stopping
                    print(f"  Early stopping triggered after epoch {epoch + 1}")
                    break

        # Restore best model for evaluation
        if best_model_state is not None:
            model.load_state_dict(
                {k: v.to(device) for k, v in best_model_state.items()}
            )

        # Final evaluation with detailed metrics
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_inputs, batch_labels in test_loader:
                outputs = model(batch_inputs)
                _, preds = torch.max(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())

        # Store texts for misclassification analysis
        misclassified = []
        for i, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
            if true_label != pred_label:
                misclassified.append(
                    {
                        "text": df_test.iloc[i]["text"],
                        "true_label": label_encoder.inverse_transform([true_label])[0],
                        "pred_label": label_encoder.inverse_transform([pred_label])[0],
                    }
                )

        fold_misclassifications.append(misclassified)

        # Calculate metrics
        report = classification_report(
            all_labels, all_preds, target_names=label_encoder.classes_, output_dict=True
        )

        # Save results for this fold
        all_reports.append(report)
        all_predictions.extend(all_preds)
        all_true_labels.extend(all_labels)

        # Calculate aggregate scores for this fold
        fold_f1 = f1_score(all_labels, all_preds, average="macro")
        fold_precision = precision_score(all_labels, all_preds, average="macro")
        fold_recall = recall_score(all_labels, all_preds, average="macro")

        all_f1_scores.append(fold_f1)
        all_precision_scores.append(fold_precision)
        all_recall_scores.append(fold_recall)

        print(f"Fold {fold + 1} F1 Score (macro): {fold_f1:.4f}")
        print(f"Fold {fold + 1} Precision (macro): {fold_precision:.4f}")
        print(f"Fold {fold + 1} Recall (macro): {fold_recall:.4f}")

        # Print confusion matrix for this fold
        fold_cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(fold_cm)

    # Calculate aggregate metrics across all folds
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(
        f"F1 Score (macro): {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}"
    )
    print(
        f"Precision (macro): {np.mean(all_precision_scores):.4f} ± {np.std(all_precision_scores):.4f}"
    )
    print(
        f"Recall (macro): {np.mean(all_recall_scores):.4f} ± {np.std(all_recall_scores):.4f}"
    )

    # Create overall confusion matrix
    cm = confusion_matrix(all_true_labels, all_predictions)

    # Return all results for further analysis
    return {
        "reports": all_reports,
        "f1_scores": all_f1_scores,
        "precision_scores": all_precision_scores,
        "recall_scores": all_recall_scores,
        "confusion_matrix": cm,
        "misclassifications": fold_misclassifications,
        "classes": label_encoder.classes_,
    }


def plot_confusion_matrix(
    cm: np.ndarray, class_names: List[str], output_path: str = "confusion_matrix.png"
):
    """Plot a confusion matrix with proper labeling"""

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_misclassification_word_clouds(
    misclassifications: List[Dict],
    class_names: List[str],
    output_dir: str,
    use_ngrams: str = "uni",  # Changed default from "bi" to "uni"
):
    """Generate word clouds for misclassified texts per class

    Args:
        misclassifications: List of misclassified examples
        class_names: List of class names
        output_dir: Directory to save output files
        use_ngrams: Which n-grams to use ("uni", "bi", "tri", "all")
    """
    os.makedirs(output_dir, exist_ok=True)

    # Combine misclassifications across all folds
    all_misclassifications = []
    for fold_misclass in misclassifications:
        all_misclassifications.extend(fold_misclass)

    # Group by true class
    misclassified_by_class = {}
    for cls in class_names:
        misclassified_by_class[cls] = []

    for item in all_misclassifications:
        true_label = item["true_label"]
        misclassified_by_class[true_label].append(item["text"])

    # Create stop words set - use a simplified list since we don't have NLTK's STOPWORDS
    stop_words = {
        "the",
        "and",
        "to",
        "of",
        "a",
        "in",
        "that",
        "it",
        "with",
        "as",
        "for",
        "is",
        "on",
        "you",
        "this",
        "be",
        "are",
        "i",
        "at",
        "by",
        "not",
        "or",
        "have",
        "from",
        "but",
        "an",
        "they",
        "which",
        "was",
        "we",
        "their",
        "been",
        "has",
        "will",
        "all",
        "if",
        "can",
        "when",
        "so",
        "no",
        "would",
        "what",
        "about",
        "who",
        "there",
        "my",
        "your",
        "his",
        "her",
        "our",
        "its",
    }

    # Generate word clouds for each class
    for cls, texts in misclassified_by_class.items():
        if not texts:  # Skip if no misclassifications for this class
            continue

        # Combine all texts
        combined_text = " ".join(texts)

        # Create word frequency dictionaries using our custom functions
        tokens = simple_tokenize(combined_text)
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Create frequency maps for different n-gram levels
        ngram_types = []
        if use_ngrams == "all" or use_ngrams == "uni":
            ngram_types.append(("unigrams", 1))
        if use_ngrams == "all" or use_ngrams == "bi":
            ngram_types.append(("bigrams", 2))
        if use_ngrams == "all" or use_ngrams == "tri":
            ngram_types.append(("trigrams", 3))

        for ngram_name, n in ngram_types:
            if n == 1:
                # Use single words (unigrams)
                word_freq = Counter(filtered_tokens)
                suffix = "unigrams"
            else:
                # Create n-grams using custom function
                ngrams = list(generate_ngrams(filtered_tokens, n))
                word_freq = Counter([f"{' '.join(gram)}" for gram in ngrams])
                suffix = f"{ngram_name}"

            if not word_freq:
                print(f"No meaningful {ngram_name} found for class {cls}")
                continue

            # Create word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                max_words=100,
                stopwords=stop_words,
                collocations=False,  # Important: disable automatic bigrams
            ).generate_from_frequencies(word_freq)

            # Plot and save
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Misclassified {ngram_name.title()} for Class: {cls}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_dir, f"wordcloud_{cls.replace(' ', '_')}_{suffix}.png"
                )
            )
            plt.close()

    print(f"Word clouds saved to {output_dir}")


def main():
    """Main evaluation entry point"""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate BERT classifier using cross-validation"
    )
    parser.add_argument(
        "--folds", type=int, default=5, help="Number of cross-validation folds"
    )
    parser.add_argument(
        "--ngrams",
        type=str,
        default="uni",  # Changed default from "bi" to "uni"
        choices=["uni", "bi", "tri", "all"],
        help="N-grams to use in word clouds (uni, bi, tri, or all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs per fold"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override the output directory"
    )

    args = parser.parse_args()

    # Use the get_device function to select the best device
    device = get_device()
    print(f"Main process using device: {device}")

    # Load configuration
    with open("config.yml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Create evaluation output directory
    if args.output_dir:
        eval_dir = args.output_dir
    else:
        eval_dir = config.get(
            "EVALUATION_DIR", os.path.join(config["BASE_ROOT"], "evaluation")
        )

    os.makedirs(eval_dir, exist_ok=True)
    wordcloud_dir = os.path.join(eval_dir, "wordclouds")
    os.makedirs(wordcloud_dir, exist_ok=True)

    # Load dataset from CSV file
    data_file = config.get("CSV_PATH")
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return

    print(f"Loading data from {data_file}")
    df = pd.read_csv(data_file)
    print(f"Dataset loaded with {len(df)} samples")

    # Perform cross-validation with updated parameters
    n_folds = args.folds
    num_epochs = args.epochs
    print(
        f"Starting cross-validation with {n_folds} folds and {num_epochs} epochs per fold"
    )

    results = cross_validate(df, config, n_folds=n_folds, num_epochs=num_epochs)

    # Plot confusion matrix
    plot_confusion_matrix(
        results["confusion_matrix"],
        results["classes"],
        output_path=os.path.join(eval_dir, "confusion_matrix.png"),
    )

    # Generate word clouds for misclassified samples
    generate_misclassification_word_clouds(
        results["misclassifications"],
        results["classes"],
        output_dir=wordcloud_dir,
        use_ngrams=args.ngrams,  # Pass ngrams argument
    )

    # Save detailed metrics to CSV
    metrics_df = pd.DataFrame(
        {
            "Fold": list(range(1, n_folds + 1)),
            "F1 (macro)": results["f1_scores"],
            "Precision (macro)": results["precision_scores"],
            "Recall (macro)": results["recall_scores"],
        }
    )

    # Add mean and std rows
    metrics_df.loc[n_folds] = [
        "Mean",
        np.mean(results["f1_scores"]),
        np.mean(results["precision_scores"]),
        np.mean(results["recall_scores"]),
    ]
    metrics_df.loc[n_folds + 1] = [
        "Std",
        np.std(results["f1_scores"]),
        np.std(results["precision_scores"]),
        np.std(results["recall_scores"]),
    ]

    # Save metrics to CSV
    metrics_path = os.path.join(eval_dir, "cv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Cross-validation metrics saved to {metrics_path}")

    # Return results for potential further use
    return results


if __name__ == "__main__":
    main()
