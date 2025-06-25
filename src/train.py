import random

import numpy as np

from src.mlp_utils.loss_fns import CrossEntropyLoss
from src.model import MultiLayerPerceptron
from src.model2 import MultiLayerPerceptron2
from src.mlp_utils.optimisers import SGDOptimiser

from src.utils.config_loader import load_config
from src.utils.parameter_loader import load_sgd_hyperparams, load_training_hyperparams, load_model_hyperparams
from src.utils.logger import log_experiment
from src.utils.visualisations import plot_losses
from src.data_tools.process_dataset import get_preprocessed_datasets
from src.scripts.training_scripts import train_one_epoch, validate_one_epoch, train_one_epoch2, validate_one_epoch2


def train():
    config = load_config()
    set_seed(config)

    # Load in preprocessed data
    x_train, y_train, x_val, y_val = get_preprocessed_datasets(mode="TRAIN")
    (n, d) = x_train.shape
    (n, c) = y_train.shape

    # Retrieve all hyperparameters
    hidden_layers, hidden_activation, output_activation, init_func = load_model_hyperparams()
    batch_size, epochs = load_training_hyperparams()
    learning_rate, momentum = load_sgd_hyperparams()

    # Initialise SGD optimiser with momentum
    sgd_optimiser1 = SGDOptimiser(learning_rate, momentum)
    sgd_optimiser2 = SGDOptimiser(learning_rate, momentum)
    cross_entropy_loss = CrossEntropyLoss()

    # Initialise Multi Layer Perceptron
    model = MultiLayerPerceptron(input_dim=d, output_dim=c, hidden_layers_config=hidden_layers,
                                 hidden_activation_fn=hidden_activation, output_activation_fn=output_activation,
                                 weight_init_strategy=init_func)
    set_seed(config)
    model2 = MultiLayerPerceptron2()

    # Run training script for the defined number of epochs
    # Log training and validation losses
    train_losses, val_losses = [], []

    # Define a patience threshold for early stopping
    patience = 2

    for epoch in range(epochs):
        #loss = train_one_epoch(model, sgd_optimiser1, x_train, y_train, batch_size)
        set_seed(config)
        #train_loss = train_one_epoch(model, sgd_optimiser1, x_train, y_train, batch_size)
        train_loss = train_one_epoch2(model2, sgd_optimiser2, cross_entropy_loss, x_train, y_train, batch_size)
        train_losses.append(train_loss)
        #val_loss = validate_one_epoch(model, x_val, y_val, batch_size)
        val_loss = validate_one_epoch2(model2, cross_entropy_loss, x_val, y_val, batch_size)
        val_losses.append(val_loss)

        #if (epoch + 1) % 2 == 0:
        print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

        if val_loss <= train_loss:
            patience = 3
        else:
            patience -= 1

        if patience == 0:
            print(f"No improvement observed over {epoch} epochs, early stopping.")
            break

    fig = plot_losses(train_losses, val_losses)

    #params1 = model.get_parameters()
    #params2 = model2.get_parameters()

    #print(f"OLD: {params1['W1'][0][0]}")
    #print(f"NEW: {params2['W1'][0][0]}")

    enable_logging = config["LOGGING_SETTINGS"]["ENABLE_LOGGING"]
    if enable_logging == "True":
        log_experiment(model2, fig)


def set_seed(config):
    """Set fixed random seeds throughout project for training reproducibility."""
    seed = config["TRAINING_CONFIG"]["RANDOM_SEED"]
    random.seed(seed)
    np.random.seed(seed)


if __name__ == '__main__':
    train()