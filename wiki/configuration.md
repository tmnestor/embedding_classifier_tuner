# Configuration System

## Purpose

The configuration system provides a central, flexible, and maintainable way to manage all settings across the entire framework. It uses YAML with advanced features like anchors, references, and custom tags to create a clean, DRY (Don't Repeat Yourself) configuration structure.

## Configuration Design

The configuration is organized into logical sections:

### 1. Directory Structure

The configuration defines a directory structure with all paths relative to a single `BASE_ROOT` directory:

```yaml
# BASE_ROOT is set from FTC_OUTPUT_PATH environment variable or uses default
BASE_ROOT: &BASE_ROOT !env_var_or_default [FTC_OUTPUT_PATH, "/users/username/bert_outputs"]

# Define all directory paths using the BASE_ROOT anchor
DATA_ROOT: &DATA_ROOT !join [ *BASE_ROOT, "/data" ]
CKPT_ROOT: &CKPT_ROOT !join [ *BASE_ROOT, "/checkpoints" ]
BEST_CONFIG_DIR: !join [ *BASE_ROOT, "/config" ]
LEARNING_CURVES_DIR: !join [ *BASE_ROOT, "/evaluation" ]
CALIBRATION_DIR: &CALIBRATION_DIR !join [ *BASE_ROOT, "/calibration" ]
```

This structure ensures that:
- All outputs are organized in a single location
- The entire system can be moved to a new location by changing just `BASE_ROOT`
- Directory relationships are maintained consistently

### 2. Model and Training Parameters

Basic model parameters and training settings are defined in a separate section:

```yaml
base: &base
  SEED: 42                       # Random seed for reproducibility
  NUM_EPOCHS: 10                 # Default number of training epochs
  BATCH_SIZE: 64                 # Batch size for training
  MAX_SEQ_LEN: 64                # Maximum sequence length for tokenization
  MODEL: "all-mpnet-base-v2"     # Base transformer model
  MODEL_NAME: "all-mpnet-base-v2"  # Name used for loading the model
  DROP_OUT: 0.3                  # Dropout rate for classifier
  ACT_FN: "silu"                 # Activation function
  LEARNING_RATE: 2e-5            # Default learning rate
  PATIENCE: 10                   # Early stopping patience (epochs)
  LR_PATIENCE: 5                 # Learning rate reduction patience
  MIN_LR: 1e-6                   # Minimum learning rate
  STUDY_NAME: "bert_opt"         # Name for hyperparameter optimization study
```

### 3. File Paths

File paths are defined relative to the directories:

```yaml
paths: &paths
  CSV_PATH: !join [ *DATA_ROOT, "/ecommerce.csv" ]  # Main dataset
  TRAIN_CSV: !join [ *DATA_ROOT, "/train.csv" ]     # Training embeddings
  VAL_CSV: !join [ *DATA_ROOT, "/val.csv" ]         # Validation embeddings
  TEST_CSV: !join [ *DATA_ROOT, "/test.csv" ]       # Test embeddings
  CHECKPOINT_DIR: *CKPT_ROOT                        # Model checkpoint directory
  EMBEDDING_PATH: !join [ *CKPT_ROOT, "/triplet_model.pt" ]  # Triplet model
  DATA_PATH: *DATA_ROOT                             # Data directory
  CALIBRATION_PATH: !join [ *CALIBRATION_DIR, "/calibration_scores.npy" ]  # Calibration data
```

### 4. Combined Configuration

The sections are merged at the end:

```yaml
<<: *base   # Merge in base settings
<<: *paths  # Merge in path settings
```

## Advanced Features

### Path Joining

The custom `!join` tag allows clean path concatenation:

```yaml
EMBEDDING_PATH: !join [ *CKPT_ROOT, "/triplet_model.pt" ]
```

This tag is implemented in `FTC_utils/LoaderSetup.py`:

```python
def join_constructor(loader, node):
    # Joins the list of strings into one string.
    seq = loader.construct_sequence(node)
    return "".join(seq)

# Register the constructor
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)
```

### YAML Anchors and References

Anchors (`&name`) define reusable values, and references (`*name`) use them:

```yaml
BASE_ROOT: &BASE_ROOT "/users/username/bert_outputs"  # Define anchor
DATA_ROOT: &DATA_ROOT !join [ *BASE_ROOT, "/data" ]   # Reference anchor
```

This prevents duplication and ensures consistency.

## Using the Configuration

### Loading in Python

Load the configuration in Python code:

```python
import yaml
from FTC_utils.LoaderSetup import join_constructor

# Register the constructor for path joining
yaml.add_constructor("!join", join_constructor, Loader=yaml.SafeLoader)

# Load configuration
with open("config.yml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Access configuration values
data_file = config["CSV_PATH"]
model_name = config["MODEL"]
batch_size = config["BATCH_SIZE"]
```

### Command-Line Overrides

Override configuration values using command-line arguments:

```python
parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
args = parser.parse_args()

# Override config value
num_epochs = args.epochs  # Override config["NUM_EPOCHS"]
```

## Modifying the Configuration

### Adding New Parameters

Add new parameters by extending the appropriate section:

```yaml
base: &base
  # Existing parameters...
  NEW_PARAMETER: value  # Add your new parameter here
```

### Adding New Directories

Add new directories by extending the directory structure:

```yaml
NEW_DIR: &NEW_DIR !join [ *BASE_ROOT, "/new_directory" ]
```

### Environment-Specific Configuration

For environment-specific settings, consider creating separate config files:

```bash
config_dev.yml
config_prod.yml
```

And load them based on an environment variable:

```python
env = os.environ.get("ENV", "dev")
config_file = f"config_{env}.yml"
```

## Best Practices

1. **Centralization**: Keep all settings in the config file, not hardcoded in modules
2. **Documentation**: Document each parameter with comments
3. **Default Values**: Use reasonable defaults for all parameters
4. **Grouping**: Group related parameters in logical sections
5. **Validation**: Validate critical parameters when loading the configuration
6. **Overrides**: Allow command-line overrides for key parameters

## Dependencies

- PyYAML
- FTC_utils.LoaderSetup (for path joining functionality)
- FTC_utils.env_utils (for environment variable handling)