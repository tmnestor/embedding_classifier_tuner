# BertClassification Module

## Purpose

The BertClassification module trains a classifier on top of pre-computed BERT embeddings. It serves as the central training component of the pipeline, applying the optimized architecture from TuneBert and the enhanced embeddings from TripletTraining.

## Technical Background

This module implements a modular classification approach with:

1. **Embedding Generation**: Using the triplet-trained BERT model to produce text embeddings
2. **Classification Layer**: A configurable feedforward network that maps embeddings to classes
3. **Training Pipeline**: Efficient training with early stopping and learning rate scheduling

The module works with pre-computed embeddings stored in CSV files, which dramatically speeds up training compared to on-the-fly embedding generation.

![Classification Architecture](uploads/classification_architecture.png)

## Implementation

The module consists of these key components:

1. **BertClassifier**: Combines embedding model and classifier
2. **FFModel/DenseBlock**: Configurable classification layers
3. **Training Logic**: Handles model training, validation, and checkpointing

```python
# BertClassifier contains both embedding and classification components
class BertClassifier(nn.Module):
    def __init__(self, base_model, classifier):
        super(BertClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = classifier

    def forward(self, inputs):
        # Handle different input types (embeddings or tokenized inputs)
        # ...
        logits = self.classifier(normalized_output)
        return logits
```

## Usage

### Command Line

```bash
python BertClassification.py
```

### Workflow

1. The module first checks if embeddings need regeneration:
   ```python
   if embeddings_are_outdated():
       generate_embedding_csv_data(triplet_model, DEVICE)
   ```

2. It then loads the best configuration (if available) or uses defaults:
   ```python
   if os.path.exists(best_config_file):
       with open(best_config_file, "r") as f:
           best_config = yaml.safe_load(f)
   ```

3. Creates and trains the classifier:
   ```python
   model = BertClassifier(triplet_model, classifier).to(DEVICE)
   metrics = train_model(model, train_loader, val_loader, optimizer, criterion, scheduler)
   ```

## Output

The module produces:

1. A trained model saved at:
   ```
   {CHECKPOINT_DIR}/model/best_model.pt
   ```

2. Learning curve plots showing training and validation metrics:
   ```
   {LEARNING_CURVES_DIR}/learning_curves.png
   ```

3. Training metrics saved for analysis:
   ```
   {CHECKPOINT_DIR}/training_metrics.npy
   ```

## Performance Optimization

BertClassification employs several techniques to improve training:

1. **Learning Rate Scheduling**: Reduces learning rate when performance plateaus
2. **Early Stopping**: Terminates training when no improvements are seen
3. **Pre-computed Embeddings**: Avoids redundant BERT computation
4. **Device Optimization**: Automatically uses GPU acceleration when available

## Best Practices

1. **Data Preparation**: Ensure your data CSV has properly formatted text and labels
2. **Hardware**: Use GPU/MPS acceleration when available for significantly faster training
3. **Model Selection**: Run TuneBert first to find the optimal classifier architecture
4. **Checkpointing**: The best model is automatically saved based on validation accuracy

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- NumPy
- Pandas
- Matplotlib
- Seaborn
