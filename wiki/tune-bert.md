# TuneBert Module

## Purpose

The TuneBert module systematically optimizes the classifier architecture and training hyperparameters using Bayesian optimization. This helps find the best-performing model architecture for your specific dataset without manual trial and error.

## Technical Background

### Hyperparameter Optimization

TuneBert uses [Optuna](https://optuna.org/), a powerful hyperparameter optimization framework that:
- Efficiently explores the hyperparameter space
- Uses pruning to terminate unpromising trials early
- Leverages previous trial results to guide future trials

The module optimizes:
1. **Architecture Parameters**: Number of layers, layer widths, activation functions
2. **Training Parameters**: Learning rate, weight decay, optimizer choice
3. **Regularization Parameters**: Dropout rates

![Optuna Optimization Visualization](uploads/optuna_visualization.png)

## Implementation

The optimization is structured around Optuna's objective function approach:

```python
def objective(trial, device):
    # 1. Sample hyperparameters from search space
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    activation_fn = trial.suggest_categorical("activation", ["gelu", "relu", "silu"])
    n_hidden = trial.suggest_int("n_hidden", 1, 3)
    
    # 2. Build and train model
    classifier = TunableFFModel(input_dim, output_dim, trial)
    model = BERTClassifierTuned(classifier).to(device)
    
    # 3. Return validation accuracy as objective value
    return best_val_acc
```

## Usage

### Command Line

```bash
python TuneBert.py
```

### Configuration

Tuning parameters are controlled in the `TuneBert.py` script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_TRIALS` | 20 | Number of hyperparameter combinations to try |
| `EPOCHS` | 5 | Number of epochs for each trial |

## Output

The module saves the best hyperparameters to:
```
{BEST_CONFIG_DIR}/best_config_{STUDY_NAME}.yml
```

This YAML file is automatically used by BertClassification when training the final model.

## Understanding Results

After optimization, TuneBert will output:
- Best validation accuracy achieved
- Best hyperparameter configuration
- Path to the saved configuration file

Example output:
```
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
    weight_decay: 0.000015
Saved best configuration to /users/output/config/best_config_bert_opt.yml
```

## Best Practices

1. **Number of Trials**: Start with 20 trials for quick exploration, increase to 50+ for thorough optimization
2. **Compute Resources**: Use GPU acceleration if available to speed up trials
3. **Trial Duration**: Set `EPOCHS` low (3-5) during tuning to save time
4. **Search Space**: Adjust the search ranges in the objective function if needed

## Dependencies

- Optuna
- PyTorch
- PyYAML
