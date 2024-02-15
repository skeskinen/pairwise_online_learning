# TASK AGNOSTIC CONTINUAL LEARNING WITH PAIRWISE LAYER ARCHITECTURE

This repository contains the code for implementing and running experiments related to the continual learning model using the Pairwise Interaction Layer as described in our paper.

## Installation

Developed on Python 3.11

First, clone the repository and then install the required dependencies:

pip install -r requirements.txt

## Running the Experiments

To run the main experiment pipeline, execute:

python main.py

### Configuration Options

Inside `main.py`, you can optionally set the following variables to configure the experiment run:

- `HYPERPARAMETER_SWEEP = False`: Set to `True` to use Weights & Biases (wandb) for hyperparameter search.
- `LOG_TO_WANDB = False`: Set to `True` to log the experiment runs to Weights & Biases. Make sure you have set up your wandb account, set the username in main.py and are logged in.

### Reproducing Paper Results

To reproduce the results from the paper, use the provided JSON files with the best hyperparameters. For example, to run the experiment for the MLP-700-1 model on the Split MNIST dataset with Streaming-MAS, use:

python main.py repro_configs/mlp-700-1|250000_s-mas/split_mnist.json

### Local Hyperparameter Search

For quick, local hyperparameter searches, you can modify the section in `main.py` around line 85:

```python
exp_hyperparams = get_experiment_hyperparams(config)
# Uncomment the next line to train with specific hyperparameters
# return train_with_hyperparams({**exp_hyperparams, **config})
# Uncomment and modify the following for a local hyperparameter sweep
return train_trial_sweep(config, {
    'learning_rate': [2.5e-5, 5e-5, 1e-4, 2e-4, 4e-4],
    'wta_p': [0.1, 0.15],
})
```