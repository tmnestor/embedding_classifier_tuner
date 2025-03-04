# TripletTraining Module

## Purpose

The TripletTraining module improves text embedding quality through contrastive learning. It trains a BERT model to represent similar texts closer together and dissimilar texts farther apart in the embedding space, resulting in more discriminative features for classification. The module now includes visualization capabilities to inspect embedding quality.

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

### Embedding Visualization

The module uses t-SNE (t-Distributed Stochastic Neighbor Embedding) to visualize the high-dimensional embedding space in 2D. This provides a visual representation of how well the model separates different classes:

![Embedding Visualization](uploads/triplet_embeddings_visualization.png)

The visualization shows the embedding space before and after triplet training, clearly demonstrating how the training process clusters similar texts together and separates different classes.

## Implementation

The module consists of:

1. **TripletDataset**: Creates triplets of anchor, positive, and negative samples
2. **TripletEmbeddingModel**: Encodes text and produces normalized embedding vectors
3. **Training Logic**: Trains the model using triplets and saves the learned embeddings
4. **Visualization**: Generates t-SNE visualizations of embedding spaces before and after training

```python
def train_triplet(epochs=5, lr=2e-5, visualize=False):
    dataset = TripletDataset(CSV_PATH)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Generate "before" embeddings if visualization requested
    if visualize:
        before_embeddings, labels, label_texts = generate_embeddings(df, base_model, tokenizer, device)
    
    model = TripletEmbeddingModel(base_model).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=MARGIN, p=2)
    
    # Training loop...
    
    # Save the learned model
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "triplet_model.pt"))
    
    # Generate and visualize "after" embeddings if requested
    if visualize:
        after_embeddings, _, _ = generate_embeddings(df, model, tokenizer, device)
        visualize_embeddings(before_embeddings, after_embeddings, labels, label_texts, config)
```

## Usage

### Command Line

```bash
python TripletTraining.py --epochs 10 --lr 2e-5 --visualize
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 10 | Number of training epochs |
| `--lr` | 2e-5 | Learning rate |
| `--visualize` | False | Generate embeddings visualization |

## Output

The module produces:

1. A trained embedding model saved to:
   ```
   {CHECKPOINT_DIR}/triplet_model.pt
   ```

2. If visualization is enabled, a comparison image of embeddings before and after training:
   ```
   {LEARNING_CURVES_DIR}/triplet_embeddings_visualization_{timestamp}.png
   ```

The trained model is used by subsequent modules to generate enhanced text embeddings.

## Performance Impact

Training with triplet loss typically improves classification performance by:
- Increasing F1 score by 3-5%
- Improving handling of edge cases and ambiguous examples
- Creating more separable feature spaces for downstream classification

The visualization capability helps validate embedding quality by showing:
- Class separation in the embedding space
- Potential overlap between classes
- Identification of outliers or misclassified examples

## Best Practices

1. **Data Quality**: Ensure your training dataset has a good balance of classes
2. **Epochs**: 5-15 epochs is typically sufficient
3. **Learning Rate**: 1e-5 to 5e-5 works well for most applications
4. **Margin**: The default margin of 1.0 works well but can be tuned
5. **Visualization**: Use the `--visualize` flag to inspect embedding quality, especially for new datasets

## Demo Script

A demonstration script is available to quickly experiment with triplet training:

```bash
python demos/triplet_training_demo.py --epochs 5 --visualize --sample-data
```

The demo includes a sample dataset and shows the full triplet training process, making it easy to understand and experiment with the module.

## Dependencies

- PyTorch
- Transformers (Hugging Face)
- pandas
- matplotlib
- scikit-learn (for t-SNE visualization)
