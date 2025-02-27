# TripletTraining Module

## Purpose

The TripletTraining module improves text embedding quality through contrastive learning. It trains a BERT model to represent similar texts closer together and dissimilar texts farther apart in the embedding space, resulting in more discriminative features for classification.

## Technical Background

### Triplet Loss

The module implements triplet loss, which uses triplets of samples:
- **Anchor**: A reference text sample
- **Positive**: A different text from the same class
- **Negative**: A text from a different class

The loss function minimizes the distance between anchor-positive pairs while maximizing the distance between anchor-negative pairs, subject to a margin:

```
L = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

Where `d` is the Euclidean distance in embedding space.

![Triplet Loss Visualization](uploads/triplet_loss.png)

## Implementation

The module consists of:

1. **TripletDataset**: Creates triplets of anchor, positive, and negative samples
2. **TripletEmbeddingModel**: Encodes text and produces normalized embedding vectors
3. **Training Logic**: Trains the model using triplets and saves the learned embeddings

```python
def train_triplet(epochs=5, lr=2e-5):
    dataset = TripletDataset(CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    model = TripletEmbeddingModel(base_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    
    # Training loop...
    
    # Save the learned model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "triplet_model.pt"))
```

## Usage

### Command Line

```bash
python TripletTraining.py --epochs 10 --lr 2e-5
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 2e-5 | Learning rate |

## Output

The module produces a trained embedding model saved to:
```
{CHECKPOINT_DIR}/triplet_model.pt
```

This model is used by subsequent modules to generate enhanced text embeddings.

## Performance Impact

Training with triplet loss typically improves classification performance by:
- Increasing F1 score by 3-5%
- Improving handling of edge cases and ambiguous examples
- Creating more separable feature spaces for downstream classification

## Best Practices

1. **Data Quality**: Ensure your training dataset has a good balance of classes
2. **Epochs**: 5-15 epochs is typically sufficient
3. **Learning Rate**: 1e-5 to 5e-5 works well for most applications
4. **Margin**: The default margin of 1.0 works well but can be tuned

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- pandas
