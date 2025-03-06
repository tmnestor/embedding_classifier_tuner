# Free Text Categoriser

## Purpose

This project provides an extensible and reusable framework for training and fine-tuning a BERT-based text classifier, combining embedding learning with optimized classification. 
It's designed to be modular and easy to use, following a pipeline approach:

1. Train BERT embeddings using triplet loss for better text representation
2. Generate text embeddings on-the-fly to prevent data leakage
3. Optimize classifier architecture and hyperparameters using Optuna
4. Train a classifier on top of the pre-trained embeddings
5. Evaluate model performance through cross-validation and advanced metrics
6. Make predictions on new text data with the trained model

## Project Structure

```
App
├── PreTrainedEncoderModel
├── FTC
│   ├── BertClassification.py
│   ├── Evaluation.py
│   ├── Predict.py
│   ├── TripletTraining.py
│   ├── TuneBert.py
│   └── __init__.py
├── FTC_utils
│   ├── ConfigReader.py
│   ├── LoaderSetup.py
│   ├── config_utils.py
│   ├── conformal.py
│   ├── device_utils.py
│   ├── env_utils.py
│   ├── file_utils.py
│   ├── logging_utils.py
│   ├── shared.py
│   ├── utils.py
│   └── __init__.py
└── config.yml

BASE_ROOT
├── calibration
│   └── calibration_scores.npy
├── checkpoints
│   ├── model
│   │   └── best_model.pt
│   ├── training_metrics.npy
│   └── triplet_model.pt
├── config
│   └── best_config_bert_opt.yml
├── data
│   ├── development_data.csv
│   ├── test_predictions.csv
│   └── testing_data.csv
├── evaluation
│   ├── confusion_matrix.png
│   ├── cv_metrics.csv
│   ├── learning_curves.png
│   ├── triplet_embeddings_visualization_20250303_175857.png
│   └── wordclouds
│       ├── wordcloud_Books_unigrams.png
│       ├── wordcloud_Clothing_&_Accessories_unigrams.png
│       ├── wordcloud_Electronics_unigrams.png
│       └── wordcloud_Household_unigrams.png
└── logs

```

- `FTC/`: Core package containing the main modules
  - `BertClassification.py`: Core module for classifier training, embedding generation, and model evaluation
  - `TripletTraining.py`: Trains embeddings using triplet loss to create better text representations
  - `TuneBert.py`: Performs hyperparameter optimization using Optuna to find the best classifier architecture
  - `Evaluation.py`: Conducts robust model evaluation through cross-validation and detailed metrics analysis
  - `Predict.py`: Applies the trained model to make predictions on new text data
- `FTC_utils/`: Utility modules supporting the main functionality
  - `shared.py`: Shared functions used across multiple scripts
  - `utils.py`: General utility functions for text processing
  - `LoaderSetup.py`: YAML loading utilities with custom constructors
  - `device_utils.py`: Hardware detection for CUDA, MPS, and CPU
  - `conformal.py`: Implementation of conformal prediction for uncertainty quantification
  - `logging_utils.py`: Utilities for consistent logging across modules with output capture
  - `config_utils.py`: Standardized configuration handling
  - `env_utils.py`: Environment checking and directory creation
  - `file_utils.py`: File operations and path management
- `config.yml`: Central configuration file that controls all aspects of the pipeline

## Data Pipeline

The application follows a multi-stage data pipeline:

### 1. Embedding Learning with Triplet Loss

`TripletTraining.py` creates a specialized BERT model that learns to represent similar texts closer together and dissimilar texts farther apart in the embedding space. This improves classification performance compared to using the raw BERT embeddings. The script creates train/validation/test splits (60%/20%/20%) and saves them to CSV files to ensure consistency across the entire pipeline.

### 2. Consistent Data Splitting to Prevent Leakage

To prevent data leakage, the pipeline enforces consistent data splits across all stages:
- `TripletTraining.py` creates and saves train/val/test splits to specific CSV files
- `BertClassification.py` uses these exact same splits rather than creating its own
- This ensures the validation and test data used in BertClassification weren't seen during triplet model training
- The system strictly enforces this split consistency by requiring `TripletTraining.py` to run first

### 3. On-the-fly Embedding Generation

The trained triplet model is used to generate embeddings on-the-fly for each dataset split, ensuring data integrity. This approach has several advantages:
- Prevents data leakage between splits
- Maintains proper separation between training, validation, and test data
- Ensures consistency in embedding generation across all model components
- Simplifies the pipeline by avoiding storage of intermediate embeddings

### 4. Hyperparameter Optimization

`TuneBert.py` uses Optuna to systematically search for the best classifier architecture and training parameters, including:
- Number of hidden layers and their sizes
- Dropout rates
- Activation functions
- Optimizer choice and parameters

### 5. Classifier Training

`BertClassification.py` uses the triplet model to generate embeddings on-the-fly and trains a classifier using either:
- The optimized architecture from TuneBert.py
- A default architecture if no tuning has been performed

### 6. Model Evaluation

`Evaluation.py` provides robust assessment of classifier performance through:
- K-fold cross-validation (configurable number of folds)
- Comprehensive metrics (accuracy, F1, precision, recall)
- Confusion matrices for error analysis
- Word clouds of frequently misclassified terms (unigrams, bigrams, trigrams)
- Detailed metrics reports saved as CSV files

### 7. Model Application

`Predict.py` loads the trained model and makes predictions on new text data, adding predicted labels to the output.

In this module, conformal prediction provides statistically guaranteed prediction sets rather than
  just point predictions. The ConformalPredictor class calibrates on training data to compute
  nonconformity scores (1 - confidence), then determines a threshold at the specified significance level
  (default 0.1). When making predictions, it includes all classes in the prediction set where the
  nonconformity score is below this threshold, ensuring that with 90% probability, the true class is
  contained within the set. This approach quantifies prediction uncertainty by producing prediction sets
  that vary in size based on the model's confidence - more confident predictions yield smaller sets, while
   uncertain predictions yield larger sets. The implementation in Predict.py automatically falls back to
  calibration if needed, saves calibration data for reuse, and includes the prediction sets alongside
  traditional point predictions, allowing users to make decisions with a statistical guarantee on error
  rates.

## Requirements

- Python 3.10+
- PyTorch
- Transformers (Hugging Face)
- Optuna
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- PyYAML
- wordcloud
- tqdm

## Configuration System

The project uses a sophisticated configuration system based on a central `config.yml` file that employs YAML anchors, references, and custom tags to create a maintainable and flexible setup:

### Directory Structure

All output paths are defined relative to a single `BASE_ROOT` directory, which is set via an environment variable with a fallback default. This ensures consistent directory organization and makes the project easily portable across different environments:

```yaml
# BASE_ROOT is set from FTC_OUTPUT_PATH environment variable or uses default
BASE_ROOT: &BASE_ROOT !env_var_or_default [FTC_OUTPUT_PATH, "/default/path/bert_outputs"]

# All other directories reference the BASE_ROOT
DATA_ROOT: &DATA_ROOT !join [ *BASE_ROOT, "/data" ]
CKPT_ROOT: &CKPT_ROOT !join [ *BASE_ROOT, "/checkpoints" ]
BEST_CONFIG_DIR: !join [ *BASE_ROOT, "/config" ]
LEARNING_CURVES_DIR: !join [ *BASE_ROOT, "/evaluation" ]
CALIBRATION_DIR: !join [ *BASE_ROOT, "/calibration" ]
EXPLANATION_DIR: !join [ *BASE_ROOT, "/explanations" ]
```

Before running any scripts, set the environment variable to specify your output directory:

```bash
# Linux/macOS
export FTC_OUTPUT_PATH="/path/to/your/output/directory"

```

If not set, the system will use the default path `/default/path/bert_outputs` and display a warning.

### Path Joining Functionality

The configuration uses a custom YAML tag `!join` that concatenates paths, allowing for clean, readable path definitions:

```yaml
# Example of path joining
EMBEDDING_PATH: !join [ *CKPT_ROOT, "/triplet_model.pt" ]
```

This function is implemented in `FTC_utils/LoaderSetup.py` and automatically registered when the configuration is loaded.

### Configuration Sections

The configuration is organized into logical sections using YAML anchors and references:

#### Model and Training Parameters

```yaml
base: &base
  SEED: 42                                # Random seed for reproducibility
  NUM_EPOCHS: 10                          # Number of training epochs
  BATCH_SIZE: 64                          # Batch size for training
  MAX_SEQ_LEN: 64                         # Maximum sequence length for tokenization
  MODEL_PATH: "./all-mpnet-base-v2"       # Base transformer model
  MODEL_NAME: "all-mpnet-base-v2"         # Name used for loading the model
  DROP_OUT: 0.3                           # Dropout rate for classifier
  ACT_FN: "silu"                          # Activation function (silu, gelu, relu, etc.)
  LEARNING_RATE: 2e-5                     # Base learning rate
  PATIENCE: 10                            # Early stopping patience (epochs)
  LR_PATIENCE: 5                          # Learning rate reduction patience
  MIN_LR: 1e-6                            # Minimum learning rate
  STUDY_NAME: "bert_opt"                  # Name for hyperparameter optimization study
```

#### File Paths

```yaml
paths: &paths
  CSV_PATH: !join [ *DATA_ROOT, "/development_data.csv" ]           # Main dataset
  TRAIN_CSV: !join [ *DATA_ROOT, "/train.csv" ]              # Training set with embeddings
  VAL_CSV: !join [ *DATA_ROOT, "/val.csv" ]                  # Validation set with embeddings
  TEST_CSV: !join [ *DATA_ROOT, "/test.csv" ]                # Test set with embeddings
  CHECKPOINT_DIR: *CKPT_ROOT                                 # Directory for model checkpoints
  EMBEDDING_PATH: !join [ *CKPT_ROOT, "/triplet_model.pt" ]  # Triplet model path
```

### Configuration Merging

The sections are merged at the end of the file using YAML merge keys:

```yaml
<<: *base    # Merge in the base configuration
<<: *paths   # Merge in the paths configuration
```

### Usage in Code

The configuration is loaded in each module using the standard YAML loader with the custom constructor:

```python
import yaml
from FTC_utils.LoaderSetup import join_constructor

# Register the YAML constructor
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)

# Load the configuration
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
```

### Overriding Configuration

For specific runs, command-line arguments can override configuration values:

```bash
# Override the number of epochs and learning rate
python TripletTraining.py --epochs 20 --lr 1e-5
```

This flexible configuration system provides several benefits:
- Single source of truth for all settings
- Clean path management using relative paths from a base directory
- Logical organization of related settings
- Easy portability across different environments
- Ability to override settings via command line when needed

## Usage

### 1. Train the Triplet Embedding Model

```bash
python FTC/TripletTraining.py --epochs 10 --lr 2e-5
```
This generates a model that learns to represent text in a semantically meaningful vector space.

### 2. Tune Classifier Hyperparameters (Optional)

```bash
python FTC/TuneBert.py --trials 30
```
This uses Optuna to find the optimal classifier architecture and hyperparameters, saving results to a YAML file.

### 3. Train the Classifier

```bash
python FTC/BertClassification.py
```
Trains the classifier using either the tuned architecture or default settings, providing detailed performance metrics and learning curves.

### 4. Evaluate Model Performance

```bash
python FTC/Evaluation.py --folds 5 --epochs 10 --ngrams all
```
Runs cross-validation to assess model robustness, generates confusion matrices, and creates word clouds of misclassified terms. The `--ngrams` parameter supports "uni" (default), "bi", "tri", or "all" to generate different n-gram word clouds.

### 5. Make Predictions

```bash
python FTC/Predict.py [--input PATH] [--output PATH] [--significance 0.1]
```
Loads the trained model and makes predictions on new data. Features include:
- Conformal prediction sets with statistical guarantees
- Confidence scores for each prediction
- Prediction explanations using SHAP, LIME, or attention visualization
- Decision support reports with recommended actions
- CSV output with detailed prediction information

#### Command Line Arguments
- `--input`: Path to input CSV file (defaults to test file in config)
- `--output`: Path to output predictions CSV file
- `--significance`: Significance level for conformal prediction (default: 0.1)
- `--batch-size`: Batch size for prediction (default: 32)
- `--large-file`: Process input file in chunks for very large datasets
- `--chunk-size`: Number of rows to process at once when using --large-file

#### Output Files
- Predictions CSV with all prediction information
- Decision reports in YAML format with recommended actions
- Explanation visualizations showing which words influenced predictions
- All outputs saved to directories specified in config.yml

#### Output Format
The prediction output includes:
- `predicted_label`: Most likely class for each text
- `confidence_score`: Confidence level (0-1) for the prediction
- `prediction_set`: Set of possible classes with statistical guarantees
- `prediction_set_size`: Number of classes in the prediction set

#### Decision Support
When using `--decision-support`, comprehensive reports are generated with:
- Primary prediction with confidence assessment
- Supporting evidence highlighting influential words
- Alternative predictions with supporting evidence
- Uncertainty assessment (low, medium, high)
- Recommended actions based on confidence and uncertainty

#### Programmatic Usage
The prediction functionality can also be used programmatically:
```python
from Predict import load_model, predict_with_explanation_and_confidence

# Load model components
model, tokenizer, label_encoder, conformal_predictor = load_model()

# Make predictions with explanations
texts = ["Text to classify"]
results = predict_with_explanation_and_confidence(
    model, tokenizer, texts, label_encoder,
    conformal_predictor, method="shap"
)
```


## Model Architecture

The system uses a two-part architecture:

1. **Embedding Model**: A transformer-based model that converts text to vectors
2. **Classifier**: A configurable feedforward network that maps embeddings to classes

The classifier architecture can be:
- Automatically tuned with different numbers of layers, widths, and activation functions
- Manually specified with a predefined structure

## Performance Visualization

Training and validation metrics are automatically plotted and saved, including:
- Loss curves
- F1 score progression
- Confusion matrices
- Word clouds of misclassified terms (unigrams, bigrams, or trigrams)
- Detailed summaries in console output

## Device Support

The code automatically selects the best available device:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- CPU as fallback

## Notes

- **IMPORTANT:** Always run TripletTraining.py first before BertClassification.py to prevent data leakage
- The pipeline is designed to maintain proper isolation between training, validation, and test sets
- Embeddings are generated on-the-fly to avoid data leakage, never stored in intermediate CSV files
- Always use models loaded from paths in config.yml, never download them directly from HuggingFace
- To improve performance, adjust `NUM_TRIALS` in `TuneBert.py` for more thorough hyperparameter search
- For robust evaluation, increase folds in `Evaluation.py` (e.g., `--folds 10`)
- For detailed error analysis, use `--ngrams all` to see patterns at different n-gram levels
- For processing large datasets, use batching options (see "Large Dataset Handling" section below)

## Large Dataset Handling

The system includes advanced batching features for handling production-scale datasets:

### Batched Embedding Generation

Process large datasets efficiently with memory-optimized batching:

```bash
python BertClassification.py --large-dataset --chunk-size 5000
```

This processes data in chunks to avoid memory issues, with automatic progress tracking and memory management.

### Memory-Efficient Training

Train triplet models on larger datasets with gradient accumulation:

```bash
python TripletTraining.py --batch-size 16 --gradient-accumulation 4
```

This creates an effective batch size of 64 while only using memory for 16 samples at a time.

### Chunked Prediction for Large Files

Process prediction files too large to fit in memory:

```bash
python Predict.py --large-file --chunk-size 10000
```

This streams through the input file in manageable chunks, ensuring efficient processing regardless of file size.

### Automatic Memory Management

All modules include:
- Periodic CUDA cache clearing to prevent memory leaks
- Progress tracking with ETA for long-running operations
- Efficient tensor management to minimize memory usage
- Automatic batch size selection based on available hardware

## Future Extensions

Possible future extensions for this project include:

- Adding a test suite for the prediction and evaluation functionality
- Implementing additional evaluation metrics and visualizations
- Creating a web UI for interactive model exploration
- Adding Model Explainability features
- Adding visual dashboards for browsing and comparing explanations