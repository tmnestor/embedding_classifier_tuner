# Quick Start Guide

## Purpose

This guide provides a step-by-step process to quickly get started with the BERT Embedding Classifier Tuner. It covers essential setup, data preparation, model training, and evaluation with minimal configuration.

## Prerequisites

Before starting, ensure you have:

1. Python 3.7+ installed
2. PyTorch installed (with CUDA/MPS support if available)
3. 5GB+ of free disk space
4. A CSV file with text data and labels

## Data Preparation

Prepare your dataset with these columns:
- `text`: The input text to classify
- `labels`: The category/class for each text

Example CSV format:
```csv
text,labels
"This product exceeded my expectations!",Positive
"The delivery was delayed by two days.",Negative
"It works as described in the listing.",Neutral
```

## Step 1: Configuration

Create or modify `config.yml` to point to your data:

```yaml
# Set using FTC_OUTPUT_PATH environment variable or default location
BASE_ROOT: &BASE_ROOT !env_var_or_default [FTC_OUTPUT_PATH, "/path/to/outputs"]
DATA_ROOT: &DATA_ROOT !join [ *BASE_ROOT, "/data" ]
CSV_PATH: "/path/to/your/dataset.csv"  # Your input data
MODEL: "all-mpnet-base-v2"  # Base transformer model
```

The configuration uses YAML path joining for convenience:
```yaml
CHECKPOINT_DIR: !join [ *BASE_ROOT, "/checkpoints" ]
```

## Step 2: Train the Embedding Model

Run the triplet training to create better text embeddings:

```bash
python FTC/TripletTraining.py --epochs 10
```

This creates a transformer model that maps similar texts closer together in the embedding space, improving classification performance. The model is saved to:
```
{CHECKPOINT_DIR}/triplet_model.pt
```

Output example:
```
Epoch 1/10, Loss: 0.4876
Epoch 2/10, Loss: 0.3215
...
Epoch 10/10, Loss: 0.0897
Triplet model saved.
```

## Step 3: Optimize Classifier Architecture (Optional)

Find the optimal classifier architecture and hyperparameters:

```bash
python FTC/TuneBert.py
```

This uses Optuna to search for the best classifier configuration, balancing model complexity with performance. Results are saved to:
```
{BEST_CONFIG_DIR}/best_config_{STUDY_NAME}.yml
```

Example output:
```
Trial 7/20: Complete - Best validation accuracy: 0.9234
...
Best trial:
Validation Accuracy: 0.9452
Best hyperparameters: 
    dropout_rate: 0.2546
    activation: gelu
    n_hidden: 2
    hidden_units_0: 512
    hidden_units_1: 256
    optimizer: adamw
    lr: 0.00025
```

## Step 4: Train the Classifier

Train the final classifier using the optimized architecture:

```bash
python FTC/BertClassification.py
```

This creates embeddings for your text data (stored as CSVs) and trains a classifier with early stopping and learning rate scheduling. The trained model is saved to:
```
{CHECKPOINT_DIR}/model/best_model.pt
```

Example output:
```
Epoch 1/10
Learning Rate: 2.5e-04
Training Loss: 0.5421, Train F1: 0.7854
Validation Loss: 0.2134, Val Accuracy: 92.45%, Val F1: 0.9246
...
Test Performance:
  - Loss: 0.1956
  - Accuracy: 95.12%
  - F1 Score: 0.9513
```

## Step 5: Evaluate the Model

Evaluate model performance using cross-validation:

```bash
python FTC/Evaluation.py --folds 5 --epochs 10
```

This tests the model on different data splits to ensure robustness and identifies potential weaknesses. Results are saved to:
```
{EVALUATION_DIR}/cv_metrics.csv
{EVALUATION_DIR}/confusion_matrix.png
{EVALUATION_DIR}/wordclouds/*.png
```

Example output:
```
CROSS-VALIDATION RESULTS
============================================================
F1 Score (macro): 0.9421 ± 0.0087
Precision (macro): 0.9437 ± 0.0075
Recall (macro): 0.9408 ± 0.0093
```

## Step 6: Apply the Model

Make predictions on new data:

```bash
python FTC/Predict.py
```

This adds predictions to your test file in a new column called `predicted_label`.

## Command Line Arguments

Most scripts support additional arguments to customize their behavior:

### TripletTraining.py
```bash
python FTC/TripletTraining.py --epochs 5 --lr 1e-5
```

### TuneBert.py
```bash
python FTC/TuneBert.py --trials 30
```

### Evaluation.py
```bash
python FTC/Evaluation.py --folds 3 --epochs 5 --ngrams bi
```

### Predict.py
```bash
python FTC/Predict.py --input /path/to/data.csv --output /path/to/results.csv
```

## Abbreviated Workflow

For quick experimentation, you can use this simplified approach:

```bash
# Step 1: Configure with default settings
cp config.template.yml config.yml
# Edit config.yml to point to your data

# Step 2: Train embeddings with fewer epochs
python FTC/TripletTraining.py --epochs 5

# Step 3: Skip tuning, use default architecture
python FTC/BertClassification.py

# Step 4: Quick evaluation
python FTC/Evaluation.py --folds 3 --epochs 5
```

This streamlined approach can complete in under an hour on modern hardware.

## Troubleshooting Common Issues

### Out of Memory Errors

Reduce batch size in `config.yml`:
```yaml
BATCH_SIZE: 16  # Try lower values if needed
```

### Poor Model Performance

1. Ensure class balance in your dataset
2. Try longer training for the triplet model:
   ```bash
   python FTC/TripletTraining.py --epochs 20
   ```
3. Increase samples per class (aim for 100+ examples)

### Slow Training

1. Enable GPU/MPS acceleration
2. Set an environment variable for output paths:
   ```bash
   export FTC_OUTPUT_PATH="/your/desired/output/path"
   ```
3. Use batched processing for large datasets:
   ```bash
   python FTC/BertClassification.py --large-dataset --chunk-size 5000
   ```
4. Optimize memory usage with gradient accumulation:
   ```bash
   python FTC/TripletTraining.py --batch-size 16 --gradient-accumulation 4
   ```

## Next Steps

After completing this quick start:

1. **Experiment with different models**:
   ```yaml
   MODEL: "sentence-transformers/all-MiniLM-L6-v2"  # Faster, smaller model
   ```

2. **Try different text preprocessing** techniques in `preprocess_text()`

3. **Create a simple deployment** script to use your model in production

4. **Analyze misclassifications** using the word clouds from Evaluation.py
