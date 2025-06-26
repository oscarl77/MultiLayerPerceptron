import random

import numpy as np
from sympy.printing.pytorch import torch

from src.mlp_utils.loss_fns import CrossEntropyLoss
from src.model import MultiLayerPerceptron
from src.mlp_utils.optimisers import SGDOptimiser

from src.utils.config_loader import load_config
from src.utils.parameter_loader import load_sgd_hyperparams, load_training_hyperparams
from src.utils.logger import log_experiment
from src.utils.visualisations import plot_losses
from src.data_tools.process_dataset import get_preprocessed_datasets
from src.scripts.training_scripts import train_one_epoch, validate_one_epoch


def train():
    """
    Training environment setup to train the model.
    """
    config = load_config()
    set_seed(config)

    # Load in preprocessed data
    x_train, y_train, x_val, y_val = get_preprocessed_datasets(mode="TRAIN")

    # Retrieve all hyperparameters
    batch_size, epochs = load_training_hyperparams()
    learning_rate, momentum = load_sgd_hyperparams()

    # Initialise SGD optimiser with momentum and Cross Entropy Loss
    sgd_optimiser = SGDOptimiser(learning_rate, momentum)
    cross_entropy_loss = CrossEntropyLoss()

    # Initialise model
    model = MultiLayerPerceptron()

    # Log training and validation losses
    train_losses, val_losses = [], []

    # Define a patience threshold for early stopping
    THRESHOLD = 3
    patience = THRESHOLD

    # Run training script for the defined number of epochs
    for epoch in range(epochs):
        set_seed(config)
        train_loss = train_one_epoch(model, sgd_optimiser, cross_entropy_loss, x_train, y_train, batch_size)
        train_losses.append(train_loss)
        val_loss = validate_one_epoch(model, cross_entropy_loss, x_val, y_val, batch_size)
        val_losses.append(val_loss)

        if (epoch + 1) % 1 == 0:
            print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

        if val_loss <= train_loss:
            patience = THRESHOLD
        else:
            patience -= 1

        if patience == 0:
            print(f"No improvement observed over {epoch} epochs, early stopping.")
            break

    fig = plot_losses(train_losses, val_losses)

    # Log experiment details if enabled
    enable_logging = config["LOGGING_SETTINGS"]["ENABLE_LOGGING"]
    if enable_logging == "True":
        log_experiment(model, fig)


def set_seed(config):
    """Set fixed random seeds throughout project for training reproducibility."""
    seed = config["TRAINING_CONFIG"]["RANDOM_SEED"]
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    train()