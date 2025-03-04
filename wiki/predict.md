# Predict Module

## Purpose

The Predict module provides a streamlined interface for applying your trained BERT classifier to new text data. It handles the end-to-end process of preparing input text, generating embeddings, and returning predictions with minimal setup required.

## Technical Background

This module integrates the trained models from previous steps into a cohesive pipeline:

1. **Text preprocessing**: Cleans and normalizes input using the same pipeline as training
2. **Embedding generation**: Applies the triplet-trained BERT model to create text embeddings
3. **Classification**: Uses the tuned classifier to predict the most likely class

The module reads the same configuration as other components, ensuring consistency throughout the pipeline.

![Prediction Pipeline](uploads/prediction_pipeline.png)

## Implementation

The module centers around these key components:

1. **PredictionDataset**: Handles on-the-fly tokenization for new text data
2. **Model loading**: Reconstructs the exact architecture used during training
3. **Prediction pipeline**: Processes batches efficiently with proper error handling

```python
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
        # Return as tuple (inputs, dummy_label) to match TextDataset format
        return {k: v.squeeze(0) for k, v in encoded.items()}, 0
```

## Usage

### Command Line

```bash
python Predict.py
```

This loads and processes the test data file specified in your config file.

### Programmatic Usage

```python
from Predict import load_model, predict_labels

# Load model components
model, tokenizer, label_encoder = load_model()

# Make predictions on new texts
texts = ["This is a sample text", "Another example to classify"]
predictions = predict_labels(model, tokenizer, texts, label_encoder)
print(predictions)  # ['Class A', 'Class B']
```

## Model Loading Process

The module carefully reconstructs the exact model architecture used during training:

1. Loads the base triplet embedding model
2. Reads tuning parameters from the best configuration file
3. Reconstructs the classifier with identical architecture and hyperparameters
4. Loads the trained weights from the checkpoint

This ensures perfect consistency between training and inference.

## Configuration and Paths

The module reads the following paths from `config.yml`:

| Path | Description |
|------|-------------|
| `CHECKPOINT_DIR` | Directory containing model checkpoints |
| `EMBEDDING_PATH` | Path to the triplet model weights |
| `DATA_PATH` | Directory containing data files |
| `TEST_FILE` | Path to the test file to process |
| `BEST_CONFIG_DIR` | Directory containing tuned hyperparameters |

## Model Verification

The module performs and displays detailed model verification to ensure proper loading:

```
MODEL VERIFICATION
================================================================================
Pretrained Model: all-mpnet-base-v2
Architecture:
  - Embedding Model: TripletEmbeddingModel
  - Classifier: Custom Sequential Architecture
    - Input Dimension: 768
    - Output Classes: 4
    - Hidden Layers: 1
      - Layer 1: 721 units
Parameters:
  - Dropout Rate: 0.244
  - Activation Function: gelu
  - Optimizer: rmsprop
  - Learning Rate: 0.00029
  - Momentum: 0.467
  - Alpha: 0.915
  - Weight Decay: 0.000076
```

## Output Format

The module adds several columns to the input CSV file:
- `predicted_label`: The most likely class prediction
- `confidence_score`: Confidence score for the prediction (between 0 and 1)
- `prediction_set`: List of possible classes within the conformal prediction set
- `prediction_set_size`: Number of classes in the prediction set

Example output:
```
text,labels,predicted_label,confidence_score,prediction_set,prediction_set_size
"This is an example text",,"Electronics",0.92,["Electronics"],1
"Another sample to classify",,"Clothing",0.75,["Clothing","Electronics"],2
```

## Conformal Prediction Sets

The module uses **conformal prediction** to provide uncertainty estimates for each prediction:

- A prediction set contains all classes that could be correct with the given significance level
- The significance level (default: 0.1) controls the size of prediction sets
- Smaller prediction sets indicate more confident predictions
- A prediction set size of 1 means the model is highly confident in its prediction

This approach offers more reliable error guarantees than standard confidence scores alone.

### Programmatic Usage

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

# Access results
for i, text in enumerate(texts):
    print(f"Text: {text}")
    print(f"Prediction: {results['predicted_label'][i]}")
    print(f"Confidence: {results['confidence_score'][i]:.2f}")
    print(f"Prediction set: {results['prediction_set'][i]}")
```

## Conformal Prediction Details

### Technical Background

Conformal prediction provides statistically valid uncertainty quantification:

- Creates prediction sets that contain the true label with a guaranteed probability (1-significance)
- Gives more reliable uncertainty measures than standard softmax probabilities
- Adapts automatically to the difficulty of each prediction

### Calibration Process

The conformal predictor needs calibration data to make valid predictions:

- Uses a portion of the training data for calibration
- The calibration data is saved to `{CALIBRATION_DIR}/calibration_scores.npy`
- Once calibrated, the same calibration file can be reused for future predictions

### Significance Level

Control prediction set size with the significance parameter:
```bash
python FTC/Predict.py --significance 0.1
```

Lower significance (e.g., 0.05) means higher confidence but larger prediction sets.
Higher significance (e.g., 0.2) means smaller prediction sets with lower confidence.

## Performance Considerations

1. **Batch Processing**: The module processes data in batches (default: 32) to optimize throughput
2. **GPU/MPS Acceleration**: Automatically uses available hardware acceleration
3. **Memory Usage**: Optimized to avoid storing duplicate tensors or unnecessary copies
4. **Error Handling**: Graceful handling of edge cases like empty strings or malformed inputs
5. **Large File Support**: Processes files of any size through chunked streaming
6. **Memory Management**: Periodically clears CUDA cache to prevent memory leaks during long-running operations

## Best Practices

1. **Input Formatting**: Ensure text data is in the same format as training data
2. **Batch Size**: Adjust batch size based on available memory and text length
3. **Device Selection**: For large datasets, use GPU/MPS when available
4. **Pre-processing**: Apply the same cleaning steps used during training

## Advanced Usage

### Custom Input Files

```bash
python FTC/Predict.py --input /path/to/custom/input.csv --output /path/to/results.csv
```

### Controlling Batch Size

```bash
python FTC/Predict.py --batch-size 64
```

### Processing Very Large Files

For datasets too large to fit in memory, use the `--large-file` flag with chunked processing:

```bash
python FTC/Predict.py --input /path/to/large-dataset.csv --output /path/to/results.csv --large-file --chunk-size 10000
```

This processes the file in manageable chunks, with these benefits:
- Constant memory usage regardless of file size
- Progress tracking with percentage completion
- Periodic memory cleanup
- Automatic error recovery for problematic records

### Accessing Raw Probabilities

```python
# Get prediction results
results = predict_with_confidence(model, tokenizer, texts, label_encoder, conformal_predictor)

# Access and use raw probabilities
probabilities = results["probabilities"]
print(f"Label: {results['predicted_label'][0]}, Confidence: {results['confidence_score'][0]:.2f}")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label}: {probabilities[0][i]:.4f}")
```

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy
- matplotlib
- seaborn