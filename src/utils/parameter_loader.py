import os
import pickle

from src.utils.config_loader import load_config

config = load_config()

def load_training_hyperparams():
    """Load training hyperparams from config file"""
    params = config["TRAINING_CONFIG"]
    batch_size = params["BATCH_SIZE"]
    epochs = params["EPOCHS"]
    return batch_size, epochs

def load_sgd_hyperparams():
    """Load SGD hyperparams from config file"""
    params = config["OPTIMISER_CONFIG"]
    learning_rate = params["LEARNING_RATE"]
    momentum = params["MOMENTUM"]
    return learning_rate, momentum

def load_trained_parameters(name=None):
    """Load trained parameters dict from experiment specified in config file"""
    if name:
        NAME = name
    else:
        NAME = config["TEST_SETTINGS"]["EXPERIMENT_ID"]
    BASE_DIR = config["LOGGING_SETTINGS"]["BASE_DIR"]
    DESTINATION_DIR = os.path.join(BASE_DIR, NAME)
    parameters_filename = 'model.pkl'
    parameters_filepath = os.path.join(str(DESTINATION_DIR), parameters_filename)
    try:
        with open(parameters_filepath, "rb") as f:
            loaded_params = pickle.load(f)
            return loaded_params
    except FileNotFoundError as e:
        print(f"Error loading parameters from {DESTINATION_DIR}: {e}")

