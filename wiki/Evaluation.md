# Evaluation Module

## Purpose

The Evaluation module provides comprehensive assessment of your trained classifier's performance through cross-validation, detailed metrics analysis, and error visualization. It helps you understand model performance across different data partitions and identify specific weaknesses in classification.

## Technical Features

### Cross-Validation Implementation

The module implements k-fold cross-validation with these key features:

- **Stratified Sampling**: Maintains class distribution across folds
- **Pre-computed Embeddings**: Generates and stores embeddings once for efficiency
- **Early Stopping**: Uses patience-based stopping to avoid overfitting
- **Learning Rate Scheduling**: Reduces learning rate when performance plateaus

### Performance Metrics

The module calculates and reports multiple performance metrics:

- **Macro F1 Score**: Balanced measure across all classes regardless of class size
- **Precision**: Measures classifier exactness (TP / (TP + FP))
- **Recall**: Measures classifier completeness (TP / (TP + FN))
- **Per-fold Statistics**: Tracks performance variation across data splits
- **Mean and Standard Deviation**: Provides robust overall performance estimates

### Error Analysis Tools

The module includes sophisticated error analysis capabilities:

- **Confusion Matrix**: Visual representation of classification errors
- **N-gram Word Clouds**: Visual analysis of misclassified text patterns
  - **Unigram Analysis**: Shows single words frequently appearing in misclassified examples
  - **Bigram Analysis**: Highlights common two-word phrases in misclassifications  
  - **Trigram Analysis**: Reveals three-word patterns associated with errors

## Implementation Details

### CrossValidationDataset

This custom dataset handles both text and pre-computed embeddings:

```python
class CrossValidationDataset(Dataset):
    def __init__(self, df, tokenizer=None, device=None, max_seq_len=64):
        # Handles both embedding columns and on-the-fly tokenization
        # Can process various embedding formats (lists, numpy arrays, strings)
```

### Key Functions

The module is organized around these functions:

- **cross_validate()**: Performs k-fold cross-validation with detailed tracking
- **plot_confusion_matrix()**: Generates confusion matrix visualizations
- **generate_misclassification_word_clouds()**: Creates word clouds from misclassified texts
- **create_classifier()**: Builds classifier models with optimized architectures

## Usage

### Basic Usage

```bash
python Evaluation.py
```

This will perform 5-fold cross-validation with the default settings and generate unigram word clouds.

### Advanced Usage Options

```bash
python Evaluation.py --folds 10 --epochs 15 --ngrams all --output-dir /custom/path
```

### Command-Line Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--folds` | 5 | Number of cross-validation folds |
| `--epochs` | 10 | Number of training epochs per fold |
| `--ngrams` | "uni" | N-grams to use in word clouds ("uni", "bi", "tri", or "all") |
| `--output-dir` | None | Optional custom output directory |

## Outputs

The module generates these output files:

1. **Confusion Matrix**: `{EVALUATION_DIR}/confusion_matrix.png`
   - Visual heatmap showing prediction errors between classes
   
2. **Metrics CSV**: `{EVALUATION_DIR}/cv_metrics.csv`
   - Detailed metrics for each fold
   - Mean and standard deviation statistics
   
3. **N-gram Word Clouds**: Based on the `--ngrams` parameter
   - Unigrams: `{EVALUATION_DIR}/wordclouds/wordcloud_{CLASS_NAME}_unigrams.png`
   - Bigrams: `{EVALUATION_DIR}/wordclouds/wordcloud_{CLASS_NAME}_bigrams.png`
   - Trigrams: `{EVALUATION_DIR}/wordclouds/wordcloud_{CLASS_NAME}_trigrams.png`

## Metrics Format

The CSV output provides these metrics for each fold:

```
Fold,F1 (macro),Precision (macro),Recall (macro)
1,0.932,0.940,0.925
2,0.945,0.952,0.939
...
Mean,0.938,0.945,0.932
Std,0.008,0.007,0.009
```

## Word Cloud Analysis

The N-gram word clouds provide valuable insights into model errors:

- **Unigram Clouds**: Show individual words that appear frequently in misclassified examples
  - Helps identify problematic concepts or terms
  
- **Bigram Clouds**: Reveal common word pairs in misclassifications
  - Highlights contextual patterns causing confusion
  
- **Trigram Clouds**: Display three-word sequences leading to errors
  - Shows more complex linguistic patterns

To generate all N-gram types:

```bash
python Evaluation.py --ngrams all
```

## Best Practices

1. **Fold Count**: Start with 5 folds, increase to 10 for more reliable estimates
2. **N-gram Selection**: Use "all" for comprehensive error analysis
3. **Hardware**: Run on GPU/MPS to significantly speed up evaluation
4. **Analysis**: Compare classification performance across different folds to assess stability
5. **Word Clouds**: Look for patterns in misclassifications to identify model blindspots

## Performance Considerations

The module includes optimizations to improve performance:

- Pre-computes embeddings once for the entire dataset
- Saves model states to CPU during training to reduce memory usage
- Uses early stopping to avoid unnecessary computation
- Automatically uses available hardware acceleration (CUDA, MPS)