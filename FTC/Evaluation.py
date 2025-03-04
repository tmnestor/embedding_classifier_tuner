#!/usr/bin/env python3
# Standard library imports
import os
import sys
import re
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Iterator
from collections import Counter

# Add the parent directory (project root) to sys.path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

# Add the current directory to the path to allow importing local modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Third-party imports
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud

# Local imports
from FTC_utils.logging_utils import tee_to_file
from FTC_utils.LoaderSetup import join_constructor, env_var_or_default_constructor
from FTC_utils.env_utils import check_environment

# Start capturing output to log file
log_file = tee_to_file("Evaluation")

# Register YAML constructors
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
yaml.add_constructor(
    "!env_var_or_default", env_var_or_default_constructor, Loader=yaml.SafeLoader
)

# Check environment and create necessary directories
check_environment()

# Import necessary components from existing modules
from FTC_utils.LoaderSetup import join_constructor
from FTC_utils.shared import load_triplet_model, validate
from FTC.BertClassification import (
    FFModel,
    DenseBlock,
    preprocess_text,
    TextDataset,  # Changed from TokenizedDataset to TextDataset
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


# Create a custom BertClassifier for Evaluation that can handle direct embeddings
class EvaluationBertClassifier(nn.Module):
    """BertClassifier that can handle both direct embeddings and tokenized inputs"""

    def __init__(self, triplet_model, classifier):
        """
        Initialize the EvaluationBertClassifier.
        
        Args:
            triplet_model: The triplet BERT model for generating embeddings
            classifier: The classification head that processes embeddings
        """
        super(EvaluationBertClassifier, self).__init__()
        self.triplet_model = triplet_model
        self.classifier = classifier

    def forward(self, inputs):
        """
        Forward pass that handles both direct embeddings and tokenized inputs.

        Args:
            inputs: Either a tensor (direct embeddings) or a dictionary with input_ids/attention_mask

        Returns:
            logits: Classification logits
        """
        # Check if inputs is already a tensor (embedding)
        if isinstance(inputs, torch.Tensor):
            # Direct embeddings - pass directly to classifier
            return self.classifier(inputs)

        # Dictionary with tokenized inputs - process through triplet model
        elif isinstance(inputs, dict):
            try:
                if "input_ids" in inputs and "attention_mask" in inputs:
                    embeddings = self.triplet_model(
                        inputs["input_ids"], inputs["attention_mask"]
                    )
                    return self.classifier(embeddings)
                else:
                    # If input_ids/attention_mask aren't available, try the first tensor
                    for key, value in inputs.items():
                        if isinstance(value, torch.Tensor):
                            return self.classifier(value)
            except Exception as e:
                # Fall back to using any tensor in the inputs
                for key, value in inputs.items():
                    if isinstance(value, torch.Tensor):
                        if value.dim() == 1:
                            # Add batch dimension if needed
                            value = value.unsqueeze(0)
                        return self.classifier(value)

            # If we get here, we couldn't find a usable tensor
            raise ValueError(f"Could not find usable tensor in inputs: {type(inputs)}")

        # Handle other types of inputs
        else:
            raise ValueError(f"Unsupported input type: {type(inputs)}")


class CrossValidationDataset(Dataset):
    """Dataset for handling text during cross-validation - using on-the-fly tokenization only"""

    def __init__(self, df, tokenizer, device=None, max_seq_len=64):
        """
        Initialize the CrossValidationDataset.
        
        Args:
            df: DataFrame containing the text data and labels
            tokenizer: Tokenizer to use for on-the-fly tokenization
            device: Torch device to use for tensors (default: None, will use CPU)
            max_seq_len: Maximum sequence length for tokenization (default: 64)
        
        Raises:
            ValueError: If tokenizer is None or if 'label_enc' column is missing from DataFrame
        """
        self.df = df
        self.device = device if device is not None else torch.device("cpu")
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Validate that we have the required tokenizer
        if tokenizer is None:
            raise ValueError(
                "CrossValidationDataset requires a tokenizer for on-the-fly processing"
            )

        # Convert labels
        if "label_enc" in df.columns:
            self.labels = torch.tensor(
                self.df["label_enc"].tolist(), dtype=torch.long, device=self.device
            )
        else:
            raise ValueError("No 'label_enc' column found in DataFrame!")

    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: The number of samples in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a specific sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
        
        Returns:
            tuple: A tuple containing (encoded_inputs, label) where encoded_inputs
                  is a dictionary of tensors and label is a tensor
        """
        # Always tokenize on the fly
        text = preprocess_text(self.df.iloc[idx]["text"])

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


def create_classifier(input_dim: int, output_dim: int, config: Dict) -> nn.Module:
    """Create a classifier based on configuration"""
    # Get parameters with good defaults
    dropout_rate = config.get("dropout_rate", 0.3)
    activation_fn = config.get("activation", "gelu")

    # Create a simpler classifier that's less prone to issues
    class SimpleClassifier(nn.Module):
        """
        A simple classifier with a single hidden layer for BERT embeddings.
        
        The classifier applies a linear transformation followed by batch normalization,
        ReLU activation, dropout, and a final linear layer to produce class logits.
        """
        
        def __init__(self, input_dim, output_dim, dropout_rate=0.3):
            """
            Initialize the SimpleClassifier.
            
            Args:
                input_dim: Dimension of input embeddings
                output_dim: Number of output classes
                dropout_rate: Dropout probability (default: 0.3)
            """
            super(SimpleClassifier, self).__init__()
            # Use a single hidden layer to avoid overfitting
            hidden_size = max(input_dim // 2, output_dim * 4)

            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size, output_dim),
            )

            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

        def forward(self, x):
            """
            Forward pass through the classifier.
            
            Args:
                x: Input tensor of shape (batch_size, input_dim)
                
            Returns:
                torch.Tensor: Output logits of shape (batch_size, output_dim)
            """
            return self.layers(x)

    # Use our simpler classifier
    return SimpleClassifier(input_dim, output_dim, dropout_rate)

    # The code below is commented out but kept for reference
    """
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
    """


def get_optimizer(model: nn.Module, config: Dict) -> torch.optim.Optimizer:
    """Create optimizer based on configuration"""
    optimizer_name = config.get("optimizer", "adam")
    lr = float(config.get("lr", 0.001))
    weight_decay = float(config.get("weight_decay", 0.0001))

    # Check if model has any trainable parameters
    if not any(p.requires_grad for p in model.parameters()):
        # Make all parameters trainable as a fallback
        for p in model.parameters():
            p.requires_grad = True

    # Create optimizer based on name
    if optimizer_name.lower() == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
        )
    elif optimizer_name.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name.lower() == "sgd":
        momentum = config.get("momentum", 0.9)
        nesterov = config.get("nesterov", True)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def cross_validate(
    df: pd.DataFrame,
    config: Dict,
    n_folds: int = 5,
    random_state: int = 42,
    use_stratified: bool = True,
    num_epochs: int = 10,
    use_trained_model: bool = False,  # Parameter to use existing model
) -> Dict:
    """
    Perform cross-validation evaluation of BERT classification model.
    
    This function implements a comprehensive k-fold cross-validation process.
    For each fold, it either loads a pre-trained model or trains a new model
    from scratch, then evaluates its performance on the test fold.
    
    Args:
        df: DataFrame containing text data and labels
        config: Dictionary containing configuration parameters
        n_folds: Number of cross-validation folds (default: 5)
        random_state: Random seed for reproducibility (default: 42)
        use_stratified: Whether to use stratified folds (default: True)
        num_epochs: Number of training epochs per fold (default: 10)
        use_trained_model: Whether to use a pre-trained model instead of training (default: False)
    
    Returns:
        Dict: Dictionary containing evaluation results with the following keys:
            - reports: List of classification reports for each fold
            - f1_scores: List of F1 scores for each fold
            - precision_scores: List of precision scores for each fold
            - recall_scores: List of recall scores for each fold
            - confusion_matrix: Confusion matrix over all folds
            - misclassifications: List of misclassified instances for each fold
            - classes: List of class names
    """
    import numpy as np

    print(f"Starting {n_folds}-fold cross-validation...")
    device = get_device()
    print(f"Using device: {device}")

    # Load the tokenizer matching the model specified in config.yml
    from transformers import AutoTokenizer
    from FTC_utils.shared import MODEL

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # We'll load a separate triplet model for each fold to avoid data leakage
    # Just get the embedding dimension now for planning
    try:
        # Temporary model just to get dimensions
        temp_triplet_model = load_triplet_model(device)
        embedding_dim = temp_triplet_model.base_model.config.hidden_size
        # Delete the model to free up memory
        del temp_triplet_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except AttributeError:
        # Fallback to a standard dimension
        embedding_dim = 768

    # Load best configuration if available
    best_config_file = os.path.join(
        config.get("BEST_CONFIG_DIR", "best_configs"),
        f"best_config_{config.get('STUDY_NAME', 'default')}.yml",
    )

    if os.path.exists(best_config_file):
        with open(best_config_file, "r") as f:
            best_config = yaml.safe_load(f)
    else:
        best_config = {}

    # If we're using a trained model, get its path
    trained_model_path = None
    if use_trained_model:
        model_dir = os.path.join(config.get("CHECKPOINT_DIR", "checkpoints"), "model")
        trained_model_path = os.path.join(model_dir, "best_model.pt")
        if not os.path.exists(trained_model_path):
            trained_model_path = None

    # Encode labels
    label_encoder = LabelEncoder()
    df["label_enc"] = label_encoder.fit_transform(df["labels"])
    num_classes = len(label_encoder.classes_)

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
        print("-" * 40)
        
        # Set a consistent seed for this fold to ensure reproducibility
        fold_seed = random_state + fold
        torch.manual_seed(fold_seed)
        np.random.seed(fold_seed)
        random.seed(fold_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(fold_seed)
        
        # Load a fresh triplet model for each fold to avoid data leakage
        print(f"Loading fresh triplet model for fold {fold + 1}")
        triplet_model = load_triplet_model(device)

        # Split data
        df_train = df.iloc[train_idx].reset_index(drop=True)
        df_test = df.iloc[test_idx].reset_index(drop=True)
        
        # Further split training data into train and validation to prevent leakage
        # Use stratified split if possible
        if len(df_train["labels"].unique()) > 1:
            train_idx_inner, val_idx_inner = train_test_split(
                range(len(df_train)), 
                test_size=0.2,  # 20% validation
                random_state=fold_seed,
                stratify=df_train["labels"]
            )
        else:
            train_idx_inner, val_idx_inner = train_test_split(
                range(len(df_train)), 
                test_size=0.2,
                random_state=fold_seed
            )
            
        df_train_inner = df_train.iloc[train_idx_inner].reset_index(drop=True)
        df_val_inner = df_train.iloc[val_idx_inner].reset_index(drop=True)
        
        print(f"Split data: {len(df_train_inner)} train, {len(df_val_inner)} validation, {len(df_test)} test samples")

        # Create datasets with on-the-fly tokenization
        train_ds = CrossValidationDataset(df_train_inner, tokenizer=tokenizer, device=device)
        val_ds = CrossValidationDataset(df_val_inner, tokenizer=tokenizer, device=device)
        test_ds = CrossValidationDataset(df_test, tokenizer=tokenizer, device=device)

        batch_size = config.get("BATCH_SIZE", 32)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        # Make sure label encoding is correct
        num_classes = len(df["labels"].unique())
        print(f"Training with {num_classes} classes, {len(train_ds)} train samples")

        # Create model: either load trained one or create a new one
        if trained_model_path and use_trained_model:
            # Load the exact model trained by BertClassification
            print(
                f"Loading trained model from {trained_model_path} for fold {fold + 1}"
            )

            try:
                # Create classifier architecture first
                classifier = create_classifier(embedding_dim, num_classes, best_config)
                model = EvaluationBertClassifier(triplet_model, classifier).to(device)

                # Load the model checkpoint
                checkpoint = torch.load(trained_model_path, map_location=device)

                # Handle different checkpoint formats
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    # Proper checkpoint dict with model_state_dict
                    model.load_state_dict(checkpoint["model_state_dict"])
                    epoch_info = checkpoint.get("epoch", "N/A")
                else:
                    # Raw state dict
                    model.load_state_dict(checkpoint)
                    epoch_info = "N/A"

                print(f"Trained model loaded (Epoch: {epoch_info})")

                # Set up criterion for evaluation
                criterion = nn.CrossEntropyLoss()

                # No training needed, just evaluate
                model.eval()
                best_model_state = model.state_dict()

            except Exception as e:
                print(f"Error loading trained model: {e}")
                print("Falling back to training a new model...")
                use_trained_model = False  # Fall back to training

        else:
            # Create classifier for training from scratch
            classifier = create_classifier(embedding_dim, num_classes, best_config)
            model = EvaluationBertClassifier(triplet_model, classifier).to(device)

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

            # Store predictions for each epoch to monitor improvement
            epoch_preds_history = []

            for epoch in range(num_epochs):
                # Train
                model.train()
                total_loss = 0
                train_preds = []
                train_true = []

                # Compute balanced class weights
                all_train_labels = []
                for _, labels in train_loader:
                    all_train_labels.extend(labels.cpu().numpy())

                # Get class weights for balanced training
                unique_classes = np.unique(all_train_labels)
                if len(unique_classes) > 1:
                    from sklearn.utils.class_weight import compute_class_weight

                    class_weights = compute_class_weight(
                        class_weight="balanced",
                        classes=unique_classes,
                        y=all_train_labels,
                    )
                    class_weights = torch.tensor(
                        class_weights, dtype=torch.float32, device=device
                    )
                    criterion = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    criterion = nn.CrossEntropyLoss()

                # Training loop
                for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
                    optimizer.zero_grad()
                    outputs = model(batch_inputs)
                    _, preds = torch.max(outputs, 1)
                    train_preds.extend(preds.cpu().numpy())
                    train_true.extend(batch_labels.cpu().numpy())
                    loss = criterion(outputs, batch_labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()

                # Calculate performance metrics
                train_acc = np.mean(np.array(train_preds) == np.array(train_true))
                train_f1 = f1_score(
                    train_true, train_preds, average="macro", zero_division=0
                )
                # Use validation set for early stopping, NOT the test set
                val_loss, val_acc, val_f1 = validate(model, val_loader, criterion)

                # Store predictions for the validation set (not test set)
                with torch.no_grad():
                    epoch_preds = []
                    for batch_inputs, _ in val_loader:
                        outputs = model(batch_inputs)
                        _, preds = torch.max(outputs, 1)
                        epoch_preds.extend(preds.cpu().numpy())
                    epoch_preds_history.append(epoch_preds)

                # Print minimal epoch summary
                print(
                    f"  Epoch {epoch + 1}/{num_epochs}: Train F1 = {train_f1:.4f}, Val F1 = {val_f1:.4f}"
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
                    print(f"    New best validation accuracy: {val_acc:.4f}")
                else:
                    patience_counter += 1
                    print(f"    No improvement for {patience_counter} epochs")
                    if patience_counter >= 5:  # Early stopping
                        print(f"    Early stopping triggered after epoch {epoch + 1}")
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
        
        # Clean up memory between folds
        print(f"Cleaning up memory for fold {fold + 1}")
        del model, optimizer, criterion
        del train_ds, val_ds, test_ds
        del train_loader, val_loader, test_loader
        del triplet_model
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Completed fold {fold + 1}/{n_folds}")
        print("-" * 40)

    # Calculate aggregate metrics across all folds
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)

    # Make sure we have numpy imported within this function's scope
    import numpy as np

    # Check if we have enough data to calculate means
    if all_f1_scores:
        print(
            f"F1 Score (macro): {np.mean(all_f1_scores):.4f} ± {np.std(all_f1_scores):.4f}"
        )
        print(
            f"Precision (macro): {np.mean(all_precision_scores):.4f} ± {np.std(all_precision_scores):.4f}"
        )
        print(
            f"Recall (macro): {np.mean(all_recall_scores):.4f} ± {np.std(all_recall_scores):.4f}"
        )
    else:
        print("No metrics were collected across folds.")

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
            # Use pathlib for safe path handling
            plt.savefig(
                str(
                    Path(output_dir) / f"wordcloud_{cls.replace(' ', '_')}_{suffix}.png"
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
        default="uni",
        choices=["uni", "bi", "tri", "all"],
        help="N-grams to use in word clouds (uni, bi, tri, or all)",
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs per fold"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Override the output directory"
    )
    parser.add_argument(
        "--use-trained-model",
        action="store_true",
        help="Use the trained model from BertClassification.py instead of training new models for each fold",
    )

    args = parser.parse_args()

    # Get device and load configuration
    device = get_device()

    # Load configuration
    if os.path.exists("config.yml"):
        config_path = "config.yml"
    else:
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(parent_dir, "config.yml")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        config = {
            "SEED": 42,
            "BASE_ROOT": "./outputs",
            "CHECKPOINT_DIR": "./outputs/checkpoints",
            "EVALUATION_DIR": "./outputs/evaluation",
        }

    # Check model path
    model_dir = os.path.join(config.get("CHECKPOINT_DIR", "checkpoints"), "model")
    model_path = os.path.join(model_dir, "best_model.pt")

    if args.use_trained_model and not os.path.exists(model_path):
        args.use_trained_model = False

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

    # First check if the train/val/test splits from TripletTraining.py exist
    train_file = config.get("TRAIN_CSV")
    val_file = config.get("VAL_CSV")
    test_file = config.get("TEST_CSV")
    
    # Try to load all three split files to use the complete dataset for cross-validation
    if all(os.path.exists(file) for file in [train_file, val_file, test_file]):
        print("Found pre-created splits from TripletTraining.py")
        print("Loading and combining all data for cross-validation...")
        
        # Load all splits and concatenate for cross-validation
        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(val_file)
        df_test = pd.read_csv(test_file)
        
        # Combine all splits for cross-validation
        df = pd.concat([df_train, df_val, df_test], ignore_index=True)
        print(f"Combined dataset created with {len(df)} samples")
        print(f"- Train: {len(df_train)} samples")
        print(f"- Val: {len(df_val)} samples")
        print(f"- Test: {len(df_test)} samples")
    else:
        print("WARNING: Pre-created splits not found. Please run TripletTraining.py first.")
        print("Falling back to loading from the main CSV file, but this may result in data leakage.")
        
        # Load dataset from CSV file
        data_file = config.get("CSV_PATH")
        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            print("Creating dummy dataset for demonstration purposes...")

            # Create a dummy dataset
            num_samples = 100
            num_classes = 3
            class_names = [f"Class_{i}" for i in range(num_classes)]

            # Create dummy texts and labels
            texts = [
                f"This is a sample text {i} for classification" for i in range(num_samples)
            ]
            labels = [random.choice(class_names) for _ in range(num_samples)]

            # Create DataFrame
            df = pd.DataFrame({"text": texts, "labels": labels})

            print(f"Created dummy dataset with {len(df)} samples and {num_classes} classes")
        else:
            print(f"Loading data from {data_file}")
            df = pd.read_csv(data_file)
            print(f"Dataset loaded with {len(df)} samples")

    # Perform cross-validation with updated parameters
    n_folds = args.folds
    num_epochs = args.epochs
    print(
        f"Starting cross-validation with {n_folds} folds and {num_epochs} epochs per fold"
    )

    # Use the same SEED for evaluation as used in BertClassification.py
    print(f"Using random seed: {config.get('SEED', 42)} for fold splits")
    random_state = config.get("SEED", 42)

    results = cross_validate(
        df,
        config,
        n_folds=n_folds,
        num_epochs=num_epochs,
        random_state=random_state,
        use_trained_model=args.use_trained_model,
    )

    # Plot confusion matrix using pathlib for safe path handling
    plot_confusion_matrix(
        results["confusion_matrix"],
        results["classes"],
        output_path=str(Path(eval_dir) / "confusion_matrix.png"),
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

    # Save metrics to CSV using pathlib for safe path handling
    metrics_path = str(Path(eval_dir) / "cv_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Cross-validation metrics saved to {metrics_path}")

    # Print summary results
    print("\n============================================================")
    print("CROSS-VALIDATION RESULTS")
    print("============================================================")
    print(
        f"F1 Score (macro): {np.mean(results['f1_scores']):.4f} ± {np.std(results['f1_scores']):.4f}"
    )
    print(
        f"Precision (macro): {np.mean(results['precision_scores']):.4f} ± {np.std(results['precision_scores']):.4f}"
    )
    print(
        f"Recall (macro): {np.mean(results['recall_scores']):.4f} ± {np.std(results['recall_scores']):.4f}"
    )
    print(f"Results saved to: {eval_dir}")

    # Print evaluation approach
    if args.use_trained_model:
        print("\nUsed pre-trained model from BertClassification.py")
    else:
        print("\nTrained new models for each cross-validation fold")

    return results


if __name__ == "__main__":
    main()
