# Understanding the Predict.py Module

## Overview

The `Predict.py` module provides functionality for making predictions using the trained BERT embedding classifier. This module leverages the model trained through `TripletTraining.py` and tuned with `BertClassification.py` to classify text inputs.

## Key Features

- Makes predictions on new text data using the trained model
- Supports both individual predictions and batch processing
- Provides confidence scores for predictions
- Implements conformal prediction for uncertainty quantification
- Handles large files by processing in chunks to avoid memory issues

## Command Line Usage

```bash
python FTC/Predict.py [--input PATH] [--output PATH] [--significance 0.1]
```

### Arguments

- `--input`: Path to input CSV file containing text to classify
- `--output`: Path to save prediction results (CSV)
- `--use-test-split`: Use the test split created by TripletTraining.py
- `--verbose`: Show detailed model information
- `--config`: Path to specific config file (overrides default)
- `--significance`: Significance level for conformal prediction (default: 0.1)
- `--batch-size`: Batch size for prediction (default: 32)
- `--large-file`: Process input file in chunks for very large datasets
- `--chunk-size`: Number of rows to process at once when using --large-file (default: 5000)

## Core Components

### Model Loading

The module loads several key components:
- Tokenizer from the specified model
- Triplet embedding model (trained by TripletTraining.py)
- Classifier model with the optimal configuration
- Label encoder to map numeric predictions to class labels

### Prediction Workflow

1. **Input Processing**: Input text is preprocessed and tokenized
2. **Feature Extraction**: The triplet model extracts embeddings from the processed text
3. **Classification**: The classifier predicts labels from these embeddings
4. **Uncertainty Quantification**: Conformal prediction provides prediction sets with statistical guarantees

### Conformal Prediction

The module implements conformal prediction to provide prediction sets with guaranteed error rates:
- Each prediction includes a set of possible classes
- The prediction set's size adapts to the difficulty of each sample
- Guarantees that the true label is in the prediction set with probability (1-significance)

### Large File Handling

For large datasets, the module implements:
- Chunk-based processing to manage memory usage
- Progress reporting during lengthy operations
- Memory cleanup to prevent out-of-memory errors

## Output Format

The prediction output includes:
- Original text data
- Predicted label (highest confidence class)
- Confidence score
- Prediction set (all possible classes within significance level)
- Prediction set size (indicates prediction uncertainty)

## Example Usage

```python
# Programmatic usage
from FTC.Predict import load_model, predict_with_probs

# Load model components
model, tokenizer, label_encoder = load_model()

# Make predictions
texts = ["This is a sample text to classify"]
predictions, probabilities = predict_with_probs(model, tokenizer, texts, label_encoder)
```

## Integration with Other Modules

This module relies on components defined in:
- `BertClassification.py`: For model architecture and text preprocessing
- `FTC_utils/conformal.py`: For uncertainty quantification
- `FTC_utils/device_utils.py`: For hardware detection
- `FTC_utils/file_utils.py`: For safe loading of models and configuration

## Advanced Usage

### Custom Input Files

```bash
python FTC/Predict.py --input /path/to/custom/input.csv --output /path/to/results.csv
```

### Processing Very Large Files

For datasets too large to fit in memory, use the `--large-file` flag with chunked processing:

```bash
python FTC/Predict.py --input /path/to/large-dataset.csv --output /path/to/results.csv --large-file --chunk-size 10000
```

### Using Conformal Prediction

```python
from FTC.Predict import load_model, predict_with_confidence
from FTC_utils.conformal import ConformalPredictor

# Load model components
model, tokenizer, label_encoder = load_model()

# Prepare conformal predictor
conformal_predictor = ConformalPredictor(significance=0.1)
conformal_predictor.load_calibration("path/to/calibration.npy")

# Make predictions
texts = ["Text to classify"]
results = predict_with_confidence(
    model, tokenizer, texts, label_encoder,
    conformal_predictor
)
```

## Performance Considerations

1. **Batch Processing**: The module processes data in batches (default: 32) to optimize throughput
2. **GPU Acceleration**: Automatically uses available hardware acceleration
3. **Memory Management**: Periodically clears CUDA cache to prevent memory leaks during long-running operations
4. **Error Handling**: Graceful handling of edge cases and input validation