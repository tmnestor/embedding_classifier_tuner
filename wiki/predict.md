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
        return {k: v.squeeze(0) for k, v in encoded.items()}
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

The module adds a new column called `predicted_label` to the input CSV file, preserving all original data while adding predictions.

Example output:
```
text,labels,predicted_label
"This is an example text",,"Electronics"
"Another sample to classify",,"Clothing & Accessories"
```

## Performance Considerations

1. **Batch Processing**: The module processes data in batches (default: 32) to optimize throughput
2. **GPU/MPS Acceleration**: Automatically uses available hardware acceleration
3. **Memory Usage**: Optimized to avoid storing duplicate tensors or unnecessary copies
4. **Error Handling**: Graceful handling of edge cases like empty strings or malformed inputs

## Best Practices

1. **Input Formatting**: Ensure text data is in the same format as training data
2. **Batch Size**: Adjust batch size based on available memory and text length
3. **Device Selection**: For large datasets, use GPU/MPS when available
4. **Pre-processing**: Apply the same cleaning steps used during training

## Advanced Usage

### Custom Input Files

```bash
python Predict.py --input /path/to/custom/input.csv --output /path/to/results.csv
```

### Controlling Batch Size

```bash
python Predict.py --batch-size 64
```

### Getting Prediction Probabilities

```python
from Predict import predict_with_probs

texts = ["Text to classify"]
labels, probabilities = predict_with_probs(model, tokenizer, texts, label_encoder)
print(f"Label: {labels[0]}, Confidence: {max(probabilities[0]):.2f}")
```

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- pandas
- numpy