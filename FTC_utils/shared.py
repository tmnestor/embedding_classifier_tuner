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

from FTC_utils.utils import preprocess_text

# Import the join_constructor and register it explicitly here
from FTC_utils.LoaderSetup import join_constructor

# Ensure we register the constructor for this module's yaml usage
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)

# Load configuration from one level up
config_path = os.path.join(os.path.dirname(__file__), "../config.yml")
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Use MODEL_NAME consistently throughout the codebase
MODEL_NAME = config["MODEL_NAME"]
# Get explicit model path if available
MODEL_PATH = config.get("MODEL_PATH", None)
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
    Load the trained triplet model from the path specified in config.yml.
    
    TripletTraining.py saves the model with additional metadata in a dictionary,
    so we need to extract the model_state_dict from it.
    """
    model_path = EMBEDDING_PATH
    print(f"Loading trained triplet model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ERROR: Trained model file not found at {model_path}! "
            f"You must run TripletTraining.py first to create this file."
        )
    
    # Create the base model with exactly the same architecture as used in training
    try:
        # If MODEL_PATH is explicitly set in config, use it
        if MODEL_PATH:
            print(f"Using explicit model path from config: {MODEL_PATH}")
            base_model = AutoModel.from_pretrained(MODEL_PATH, local_files_only=True).to(device)
        else:
            # Otherwise use the model name
            base_model = AutoModel.from_pretrained(MODEL_NAME).to(device)
            
        # Create and return the triplet model
        triplet_model = TripletEmbeddingModel(base_model).to(device)
        print(f"Created model architecture using: {MODEL_NAME}")
    except Exception as e:
        print(f"ERROR loading model: {type(e).__name__}: {e}")
        raise
    
    # Load the saved checkpoint with warning suppression for weights_only
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, message=".*weights_only.*")
            checkpoint = torch.load(model_path, map_location=device)
        
        # Print checkpoint keys to help debug
        print(f"Checkpoint contains keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'not a dict'}")
        
        # Extract state_dict based on format
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                # Standard format with model_state_dict key
                state_dict = checkpoint["model_state_dict"]
                print(f"Using model_state_dict from checkpoint")
            elif "state_dict" in checkpoint:
                # Alternative format with state_dict key
                state_dict = checkpoint["state_dict"]
                print(f"Using state_dict from checkpoint")
            else:
                # Assume the entire dict is the state_dict
                state_dict = checkpoint
                print(f"Using entire checkpoint as state_dict")
        else:
            # Assume the checkpoint itself is the state_dict
            state_dict = checkpoint
            print(f"Checkpoint is not a dict, using directly")
        
        # Load the state_dict into the model
        triplet_model.load_state_dict(state_dict, strict=False)
        print(f"Successfully loaded trained weights from: {model_path}")
        
        # Print metadata if available
        if isinstance(checkpoint, dict):
            metadata_keys = [k for k in checkpoint.keys() if k not in ["model_state_dict", "state_dict"]]
            if metadata_keys:
                print(f"Model also contains metadata: {metadata_keys}")
        
    except Exception as e:
        print(f"ERROR loading model weights: {e}")
        print(f"This indicates a mismatch between the current model architecture")
        print(f"and the saved weights, or a corrupted file.")
        raise RuntimeError(f"Failed to load triplet model from {model_path}")
    
    # Set model to evaluation mode
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
        # Generate random data with specified device
        embeddings = (
            torch.randn(num, embedding_dim, device=device).cpu().numpy().tolist()
        )
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


def generate_embedding_csv_data(triplet_model, device, generate_test=False):
    """Generate embedding data from raw text CSV
    
    Args:
        triplet_model: The triplet embedding model to use
        device: The device to use for computation
        generate_test: Whether to generate embeddings for test data (default: False)
            For TuneBert, we don't need test embeddings at all, saving computation
    """
    # First check if TripletTraining.py has created the train/val/test splits
    # to ensure consistency and prevent data leakage
    try:
        print("Looking for pre-created data splits from TripletTraining.py...")
        if os.path.exists(TRAIN_CSV) and os.path.exists(VAL_CSV) and os.path.exists(TEST_CSV):
            print("Found pre-created split files, checking if they contain text data...")
            # Check if these files have the 'text' column but not 'embedding' column
            # This would indicate they're the splits from TripletTraining but don't have embeddings yet
            df_train = pd.read_csv(TRAIN_CSV)
            df_val = pd.read_csv(VAL_CSV)
            df_test = pd.read_csv(TEST_CSV)
            
            if ('text' in df_train.columns and 'embedding' not in df_train.columns and
                'text' in df_val.columns and 'embedding' not in df_val.columns and
                'text' in df_test.columns and 'embedding' not in df_test.columns):
                print("Using pre-created splits from TripletTraining.py to prevent data leakage.")
                df_train["text"] = df_train["text"].apply(preprocess_text)
                df_val["text"] = df_val["text"].apply(preprocess_text)
                df_test["text"] = df_test["text"].apply(preprocess_text)
            else:
                print("Files exist but don't appear to be clean splits from TripletTraining.py.")
                raise ValueError("Existing files don't have the expected structure")
        else:
            print("Pre-created splits from TripletTraining.py not found.")
            raise FileNotFoundError("Split files do not exist")
            
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading pre-created splits: {e}")
        print("ERROR: To prevent data leakage, you must run TripletTraining.py first.")
        print("This will create consistent train/val/test splits used across the pipeline.")
        print("Run: python FTC/TripletTraining.py --epochs 10 --lr 2e-5")
        sys.exit(1)

    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    df_train["label_enc"] = label_encoder.fit_transform(df_train["labels"])
    df_val["label_enc"] = label_encoder.transform(df_val["labels"])
    
    # Do this even if not using test data since it doesn't cost much
    # It will be needed if the test set is later requested
    df_test["label_enc"] = label_encoder.transform(df_test["labels"])
    
    # Load tokenizer using MODEL_PATH if available
    try:
        # If MODEL_PATH is explicitly set in config, use it
        if MODEL_PATH:
            print(f"Using explicit model path for tokenizer: {MODEL_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
        else:
            # Otherwise use the model name
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"ERROR loading tokenizer: {type(e).__name__}: {e}")
        raise

    def compute_embeddings(texts, batch_size=32):
        """Process embeddings in batches to avoid memory issues"""
        all_embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Use tqdm for progress tracking
        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Computing embeddings"):
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
                
                # Clear CUDA cache periodically
                if torch.cuda.is_available() and i % (batch_size * 10) == 0 and i > 0:
                    torch.cuda.empty_cache()
                    
            except (RuntimeError, ValueError, IndexError) as e:
                # More specific exceptions for embedding computation issues
                print(f"Error processing batch: {e}")
                # Add zero embeddings as fallback
                embedding_dim = triplet_model.base_model.config.hidden_size
                all_embeddings.extend([([0.0] * embedding_dim)] * len(batch))

        return all_embeddings

    # Create output directory if it doesn't exist
    for path in [
        os.path.dirname(TRAIN_CSV),
        os.path.dirname(VAL_CSV),
    ]:
        os.makedirs(path, exist_ok=True)
        
    if generate_test:
        os.makedirs(os.path.dirname(TEST_CSV), exist_ok=True)

    # Compute embeddings for train and validation splits
    print(f"Computing embeddings for {len(df_train)} training samples...")
    df_train["embedding"] = compute_embeddings(df_train["text"].tolist())
    print(f"Computing embeddings for {len(df_val)} validation samples...")
    df_val["embedding"] = compute_embeddings(df_val["text"].tolist())
    
    # Save train/val files
    print("Saving embedding files...")
    df_train.to_csv(TRAIN_CSV, index=False)
    df_val.to_csv(VAL_CSV, index=False)
    
    # Only compute and save test embeddings if specifically requested
    if generate_test:
        print(f"Computing embeddings for {len(df_test)} test samples...")
        df_test["embedding"] = compute_embeddings(df_test["text"].tolist())
        df_test.to_csv(TEST_CSV, index=False)
        print("Test embeddings saved to:", TEST_CSV)
    else:
        # Save a minimal test file with structure but without embeddings
        # This prevents errors if something tries to check if the file exists
        minimal_df = df_test[["text", "labels", "label_enc"]].copy()
        minimal_df.to_csv(TEST_CSV, index=False)
        print("Minimal test structure saved (no embeddings - for reference only)")
    
    print("Embedding CSV data generation complete.")
    return True


class EmbeddingDataset(Dataset):
    """Dataset for handling precomputed embeddings loaded from a CSV file."""

    def __init__(self, csv_file, device, chunk_size=None):
        """Initialize embedding dataset with optional chunking for large files
        
        Args:
            csv_file: Path to CSV file with embeddings
            device: Torch device for tensors
            chunk_size: If provided, load data in chunks to reduce memory usage
        """
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        # Use chunked loading for very large files
        if chunk_size is not None:
            self.chunked_mode = True
            # Store file path for later loading
            self.csv_file = csv_file
            self.device = device
            self.chunk_size = chunk_size
            
            # Read only metadata (count rows and check structure)
            for chunk in pd.read_csv(csv_file, chunksize=1):
                # Verify expected columns exist
                if "embedding" not in chunk.columns or "label_enc" not in chunk.columns:
                    raise ValueError(f"CSV file missing required columns: {csv_file}")
                break
                
            # Count total rows without loading the whole file
            with open(csv_file, 'r') as f:
                # Subtract 1 for header
                self.total_rows = sum(1 for _ in f) - 1
                
            # Create index of line positions for efficient random access
            self._create_index()
        else:
            # Standard mode - load everything into memory
            self.chunked_mode = False
            self.df = pd.read_csv(csv_file)

            try:
                self.df["embedding"] = self.df["embedding"].apply(eval)
            except (SyntaxError, ValueError) as e:
                # Use the "from e" syntax to preserve the exception context
                raise ValueError(f"Error parsing embeddings in {csv_file}: {e}") from e

            try:
                self.embeddings = torch.tensor(
                    self.df["embedding"].tolist(), dtype=torch.float, device=device
                )
                self.labels = torch.tensor(
                    self.df["label_enc"].tolist(), dtype=torch.long, device=device
                )
            except Exception as e:
                # Use the "from e" syntax here too
                raise ValueError(f"Error converting embeddings to tensors: {e}") from e

    def _create_index(self):
        """Create an index of file positions for random access."""
        self.line_offsets = []
        with open(self.csv_file, 'r') as f:
            # Skip header
            header_pos = f.tell()
            f.readline()
            
            # Store each line position
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                self.line_offsets.append(pos)

    def __getitem__(self, idx):
        if self.chunked_mode:
            # Read specific line from file
            with open(self.csv_file, 'r') as f:
                f.seek(self.line_offsets[idx])
                line = f.readline().strip()
            
            # Parse CSV line
            import csv
            from io import StringIO
            reader = csv.reader(StringIO(line))
            row = next(reader)
            
            # Get embedding and label from CSV row
            # First find column indices from header
            with open(self.csv_file, 'r') as f:
                header = f.readline().strip()
                reader = csv.reader(StringIO(header))
                header_row = next(reader)
                embedding_idx = header_row.index("embedding")
                label_idx = header_row.index("label_enc")
            
            # Process embedding and label
            embedding = eval(row[embedding_idx])
            label = int(float(row[label_idx]))
            
            # Convert to tensors
            embedding_tensor = torch.tensor(embedding, dtype=torch.float, device=self.device)
            label_tensor = torch.tensor(label, dtype=torch.long, device=self.device)
            
            return embedding_tensor, label_tensor
        else:
            # Standard in-memory mode
            return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        if self.chunked_mode:
            return len(self.line_offsets)
        else:
            return len(self.labels)


def load_embedding_data(device, large_dataset=False, chunk_size=10000, load_test=False, verbose=True):
    """Load embedding data from CSV files, regenerating if needed
    
    Args:
        device: Torch device for tensors
        large_dataset: If True, use chunked loading for very large datasets
        chunk_size: Size of chunks when using chunked loading
        load_test: Whether to load the test dataset (default: False for TuneBert optimization)
        verbose: Whether to print detailed messages (default: True, set to False during hyperparameter optimization)
    """
    # Check for missing files - only check train and val files by default
    required_files = [TRAIN_CSV, VAL_CSV]
    if load_test:
        required_files.append(TEST_CSV)
        
    missing_files = []
    for csv_file in required_files:
        if not os.path.exists(csv_file):
            missing_files.append(csv_file)

    # If any files are missing, try to regenerate them
    if missing_files:
        print(f"Error: Missing embedding files: {', '.join(missing_files)}")
        print("Regenerating embedding files from source data...")
        triplet_model = load_triplet_model(device)
        success = generate_embedding_csv_data(triplet_model, device, generate_test=load_test)
        if not success:
            raise FileNotFoundError("Failed to generate embedding files")
    else:
        # Files exist, but check if they have embeddings
        try:
            # Check if train and val files have embeddings
            df_train = pd.read_csv(TRAIN_CSV)
            df_val = pd.read_csv(VAL_CSV)
            
            if 'embedding' not in df_train.columns or 'embedding' not in df_val.columns:
                print("CSV files exist but don't have embeddings. Generating them now...")
                triplet_model = load_triplet_model(device)
                success = generate_embedding_csv_data(triplet_model, device, generate_test=load_test)
                if not success:
                    raise ValueError("Failed to generate embeddings for existing files")
        except Exception as e:
            print(f"Error checking CSV files: {e}")
            raise

    # Load the embedding data
    try:
        # Verify the files have embeddings column
        for csv_file in required_files:
            df = pd.read_csv(csv_file)
            if 'embedding' not in df.columns:
                print(f"Error: {csv_file} is missing the 'embedding' column")
                print("Regenerating embeddings...")
                triplet_model = load_triplet_model(device)
                success = generate_embedding_csv_data(triplet_model, device, generate_test=load_test)
                if not success:
                    raise ValueError(f"Failed to generate embeddings for {csv_file}")
                break
        
        # Determine if files are large enough to warrant chunked loading
        # Only use chunked loading if explicitly requested or files are very large
        file_sizes = {
            "train": os.path.getsize(TRAIN_CSV),
            "val": os.path.getsize(VAL_CSV)
        }
        if load_test and os.path.exists(TEST_CSV):
            file_sizes["test"] = os.path.getsize(TEST_CSV)
        
        total_size_mb = sum(file_sizes.values()) / (1024 * 1024)
        use_chunked = large_dataset or total_size_mb > 500  # Use chunking if >500MB total
        
        if use_chunked:
            print(f"Using chunked loading for large dataset ({total_size_mb:.1f} MB)")
            chunk_size_param = chunk_size
        else:
            chunk_size_param = None
            
        # Create datasets with appropriate loading mode
        train_ds = EmbeddingDataset(TRAIN_CSV, device, chunk_size=chunk_size_param)
        val_ds = EmbeddingDataset(VAL_CSV, device, chunk_size=chunk_size_param)
        
        # Only load test dataset if specifically requested
        test_ds = None
        if load_test and os.path.exists(TEST_CSV):
            test_ds = EmbeddingDataset(TEST_CSV, device, chunk_size=chunk_size_param)

        # Get label encoder information (only read header and labels column)
        from sklearn.preprocessing import LabelEncoder
        
        # Read labels efficiently - only load the labels column
        train_labels = pd.read_csv(TRAIN_CSV, usecols=["labels"])["labels"]
        label_encoder = LabelEncoder()
        label_encoder.fit(train_labels)

        # Instead of modifying the global NUM_LABEL, return it as part of the function output
        num_label = len(label_encoder.classes_)

        # Create data loaders
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # Only create test loader if requested
        test_loader = None
        if load_test and test_ds is not None:
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        if verbose:
            print(f"Loaded embedding data successfully: {num_label} classes")
            if use_chunked:
                print(f"Using chunked loading with {chunk_size} samples per chunk")
            if not load_test:
                print("Test data not loaded (not needed for hyperparameter tuning)")
            
        return train_loader, val_loader, test_loader, label_encoder, num_label
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError, KeyError) as e:
        # More specific exceptions for data loading issues
        print(f"Error loading embedding data: {e}")
        raise


def validate(model, dataloader, criterion):
    """Validate model on validation set with standardized metrics"""
    from sklearn.metrics import f1_score, precision_score, recall_score

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
    
    # Calculate multiple metrics consistently
    val_loss = total_loss / len(dataloader)
    val_acc = (all_preds == all_labels).mean()
    
    # Calculate F1 score, handling potential edge cases
    try:
        # First try macro average (equal weight to all classes)
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        # Also calculate weighted F1 (accounts for class imbalance)
        val_f1_weighted = f1_score(all_labels, all_preds, average='weighted')
        
        # Use weighted F1 if there's significant class imbalance
        # (determined by difference between macro and weighted scores)
        if abs(val_f1_macro - val_f1_weighted) > 0.05:
            # More than 5% difference indicates significant imbalance
            print(f"Note: Using weighted F1 due to class imbalance (macro: {val_f1_macro:.4f}, weighted: {val_f1_weighted:.4f})")
            val_f1 = val_f1_weighted
        else:
            val_f1 = val_f1_macro
    except Exception as e:
        print(f"Warning: F1 score calculation failed: {e}. Using default value.")
        val_f1 = 0.0
        
    return val_loss, val_acc, val_f1
