# BERT Embedding Classifier Tuner

## Purpose

This project provides a streamlined framework for training and fine-tuning a BERT-based text classifier, combining embedding learning with optimized classification. It's designed to be modular and easy to use, following a pipeline approach:

1. Train BERT embeddings using triplet loss for better text representation
2. Generate and save text embeddings to avoid recomputation during tuning
3. Optimize classifier architecture and hyperparameters using Optuna
4. Train a classifier on top of the pre-trained embeddings
5. Evaluate model performance through cross-validation and advanced metrics
6. Make predictions on new text data with the trained model

## Project Structure

- `BertClassification.py`: Core module for classifier training, embedding generation, and model evaluation
- `TripletTraining.py`: Trains embeddings using triplet loss to create better text representations
- `TuneBert.py`: Performs hyperparameter optimization using Optuna to find the best classifier architecture
- `Evaluation.py`: Conducts robust model evaluation through cross-validation and detailed metrics analysis
- `Predict.py`: Applies the trained model to make predictions on new text data
- `config.yml`: Central configuration file that controls all aspects of the pipeline
- `utils/`: Utility modules supporting the main functionality
  - `shared.py`: Shared functions used across multiple scripts
  - `utils.py`: General utility functions for text processing and device handling
  - `LoaderSetup.py`: YAML loading utilities
  - `device_utils.py`: Hardware detection and optimization utilities

## Data Pipeline

The application follows a multi-stage data pipeline:

### 1. Embedding Learning with Triplet Loss

`TripletTraining.py` creates a specialized BERT model that learns to represent similar texts closer together and dissimilar texts farther apart in the embedding space. This improves classification performance compared to using the raw BERT embeddings.

### 2. Embedding Generation and Storage

The trained triplet model is used to generate embeddings for all text data, which are saved to CSV files. This separation of embedding generation from classifier training has two advantages:
- Avoids repeatedly computing expensive BERT embeddings
- Allows focused experimentation on classifier architectures

### 3. Hyperparameter Optimization

`TuneBert.py` uses Optuna to systematically search for the best classifier architecture and training parameters, including:
- Number of hidden layers and their sizes
- Dropout rates
- Activation functions
- Optimizer choice and parameters

### 4. Classifier Training

`BertClassification.py` loads the pre-computed embeddings and trains a classifier using either:
- The optimized architecture from TuneBert.py
- A default architecture if no tuning has been performed

### 5. Model Evaluation

`Evaluation.py` provides robust assessment of classifier performance through:
- K-fold cross-validation (configurable number of folds)
- Comprehensive metrics (accuracy, F1, precision, recall)
- Confusion matrices for error analysis
- Word clouds of frequently misclassified terms
- Detailed metrics reports saved as CSV files

### 6. Model Application

`Predict.py` loads the trained model and makes predictions on new text data, adding predicted labels to the output.

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- Optuna
- scikit-learn
- pandas
- matplotlib
- seaborn
- PyYAML
- wordcloud

```bash
pip install torch transformers optuna scikit-learn pandas matplotlib seaborn pyyaml tqdm wordcloud
```

## Configuration

The project uses a central `config.yml` file with YAML path joining functionality to manage settings:

```yaml
BASE_ROOT: "/your/output/path"  # Base directory for all outputs
DATA_ROOT: "/path/to/data"      # Data directory
MODEL: "all-mpnet-base-v2"      # Base transformer model
# Additional parameters for training, batch size, etc.
```

## Usage

### 1. Train the Triplet Embedding Model

```bash
python TripletTraining.py --epochs 10 --lr 2e-5
```
This generates a model that learns to represent text in a semantically meaningful vector space.

### 2. Tune Classifier Hyperparameters (Optional)

```bash
python TuneBert.py
```
This uses Optuna to find the optimal classifier architecture and hyperparameters, saving results to a YAML file.

### 3. Train the Classifier

```bash
python BertClassification.py
```
Trains the classifier using either the tuned architecture or default settings, providing detailed performance metrics and learning curves.

### 4. Evaluate Model Performance

```bash
python Evaluation.py --folds 5 --epochs 10 --bigrams
```
Runs cross-validation to assess model robustness, generates confusion matrices, and creates word clouds to visualize frequently misclassified terms.

### 5. Make Predictions

```bash
python Predict.py
```
Loads the trained model and makes predictions on new data.

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
- Word clouds of misclassified terms
- Detailed summaries in console output

## Device Support

The code automatically selects the best available device:
- CUDA for NVIDIA GPUs
- MPS for Apple Silicon
- CPU as fallback

## Notes

- Always train embeddings before classifier training
- The embedding files can be reused across multiple experiments
- To improve performance, adjust `NUM_TRIALS` in `TuneBert.py` for more thorough hyperparameter search
- For robust evaluation, increase folds in `Evaluation.py` (e.g., `--folds 10`)