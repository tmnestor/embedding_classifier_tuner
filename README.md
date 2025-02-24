# Minimal BERT Classifier Tuner

## Purpose

This project provides a streamlined framework for training and fine-tuning a BERT-based text classifier. It's designed to be modular and easy to use, allowing you to:

1.  Train BERT embeddings using triplet loss.
2.  Train a classifier on top of the pre-trained BERT embeddings.
3.  Retrain the classifier with learning curve visualization.
4.  Predict class labels for new text data.

## Project Structure

-   `BERT_classification.py`: Contains the core BERT classifier model, data loading, training, validation, and evaluation functions.
-   `triplet_training.py`: Implements the training of BERT embeddings using triplet loss.
-   `tune_bert.py`: Uses Optuna to perform hyperparameter tuning for the classifier.
-   `main.py`: Provides a command-line interface to orchestrate the different stages of the project.
-   `./data/csv/`: Contains example BBC text dataset in CSV format.
-   `./checkpoints/`: Stores model checkpoints, training metrics, and other artifacts.

## Requirements

-   Python 3.7+
-   PyTorch
-   Transformers
-   Optuna
-   Scikit-learn
-   Pandas
-   Seaborn
-   Matplotlib
-   PyYAML

To install the necessary packages, run:

```bash
pip install torch transformers optuna scikit-learn pandas seaborn matplotlib pyyaml
```

## Usage

1. **Train Triplet Embedding Model**
   - Run `triplet_training.py` to generate and save the triplet model and embeddings.
   ```bash
   python triplet_training.py --epochs <num_epochs> --lr <learning_rate>
   ```

2. **Hyperparameter Tuning**
   - Run `tune_bert.py` to perform hyperparameter tuning. The best configuration is saved to the BEST_CONFIG_DIR.
   ```bash
   python tune_bert.py
   ```

3. **Train and Evaluate Classifier**
   - Run `BERT_classification.py` to train the classifier using the triplet model embeddings and tuned hyperparameters.
   ```bash
   python BERT_classification.py
   ```

4. **Run Predictions**
   - Run `predict.py` to load the best classifier checkpoint and add predictions to your test CSV.
   ```bash
   python predict.py
   ```

## Hyperparameter Tuning

```bash
python tune_bert.py
```

This script will run Optuna to find the best hyperparameters and save them to a YAML file named best_config_<study_name>.yml. The study_name is defined as a constant in both tune_bert.py and BERT_classification.py.

## Notes

Ensure that you train the BERT embeddings first before training or retraining the classifier.
The study_name argument should match the name used when running tune_bert.py to tune the hyperparameters.
The best configuration is loaded from best_config_<study_name>.yml in the project directory.
For the predict subcommand, the input CSV file will be modified with a new column containing the predicted labels.

# Data Pipeline

The application follows a multi-stage data pipeline to prepare and use embeddings for classifier training:

## Loading the Triplet Model

The `load_triplet_model` function (defined in `BERT_classification.py`) is responsible for:
- **Model Initialization:** It loads a pre-trained transformer model (using the model name provided in the configuration).
- **Triplet Embedding Wrapping:** The transformer is wrapped inside a `TripletEmbeddingModel`, which computes embeddings by applying mean-pooling over the last hidden states and normalizing the result.
- **Parameter Loading:** Finally, the function loads the saved model parameters from the file specified by the `EMBEDDING_PATH` in the config, setting the model into evaluation mode for inference steps.

This function ensures that the same model architecture and weights are used consistently for generating embeddings across different modules.

## Loading Embedding Data

The `load_embedding_data` function (also in `BERT_classification.py`) implements the following steps:
- **CSV Reading:** It reads CSV files that contain precomputed embeddings along with their corresponding labels. These CSVs are generated by the `generate_embedding_csv_data` function.
- **Data Conversion:** The stored string representation of embeddings is converted back into numerical lists (using `ast.literal_eval`) and then to PyTorch tensors.
- **Dataset Creation:** Custom `EmbeddingDataset` objects are created for train, validation, and test splits.
- **DataLoader Preparation:** These datasets are wrapped in `DataLoader` objects to handle batching and shuffling during training and evaluation of the classifier.

This pipeline avoids recomputation of embeddings, allowing for efficient classifier training since the heavy computation of transformer embeddings is done only once.

## Overall Workflow

1. **Embedding Training:** The `triplet_training.py` script trains a triplet model using raw CSV data after text preprocessing.
2. **Embedding Generation:** The trained triplet model is then used by `generate_embedding_csv_data` to compute and save normalized embeddings into CSV files.
3. **Data Loading for Classification:** Finally, `load_embedding_data` reads these CSVs, creates PyTorch DataLoaders, and supplies the precomputed embeddings for classifier training and evaluation.

This modular design separates the computationally intensive embedding generation from the classifier training, making the overall process efficient and scalable.