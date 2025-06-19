import random
import numpy as np

from src.model import MultiLayerPerceptron
from src.mlp_utils.optimisers import SGDOptimiser

from src.data_tools.process_dataset import get_preprocessed_datasets
from src.utils.config_loader import load_config
from src.utils.parameter_loader import *
from src.scripts.training_scripts import train_one_epoch, validate_one_epoch


def train():
    config = load_config()
    set_seed(config)

    # Load in preprocessed data
    x_train, y_train, x_val, y_val = get_preprocessed_datasets(mode="TRAIN")
    (n, d) = x_train.shape
    (n, c) = y_train.shape

    # Retrieve all hyperparameters
    hidden_layers, hidden_activation, output_activation, init_func = load_model_hyperparams(config)
    batch_size, epochs = load_training_hyperparams(config)
    learning_rate, momentum = load_sgd_hyperparams(config)

    # Initialise SGD optimiser with momentum
    sgd_optimiser = SGDOptimiser(learning_rate, momentum)

    # Initialise Multi Layer Perceptron
    model = MultiLayerPerceptron(input_dim=d, output_dim=c, hidden_layers_config=hidden_layers,
                                 hidden_activation_fn=hidden_activation, output_activation_fn=output_activation,
                                 weight_init_strategy=init_func)

    # Run training script for the defined number of epochs
    for epoch in range(epochs):
        train_acc, train_loss = train_one_epoch(model, sgd_optimiser, x_train, y_train, batch_size)
        val_acc, val_loss = validate_one_epoch(model, x_val, y_val, batch_size)

        print(f"Epoch: {epoch + 1}, Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

def set_seed(config):
    """
    Set fixed random seeds throughout project for training reproducibility.
    :param config: dict containing all project configurations
    """
    seed = config["TRAINING_PARAMS"]["RANDOM_SEED"]
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    train()