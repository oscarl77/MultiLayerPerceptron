from src.mlp_utils.activations import ACTIVATIONS
from src.mlp_utils.initialisers import INIT_STRATEGIES

def load_model_hyperparams(config):
    """Load model hyperparams from config file"""
    params = config["MODEL_PARAMS"]
    hidden_layers = params["HIDDEN_LAYERS"]
    init_func = INIT_STRATEGIES[params["INIT_STRATEGY"]]
    hidden_activation = ACTIVATIONS[params["HIDDEN_ACTIVATION"]]
    output_activation = ACTIVATIONS[params["OUTPUT_ACTIVATION"]]
    return hidden_layers, hidden_activation, output_activation, init_func

def load_training_hyperparams(config):
    """Load training hyperparams from config file"""
    params = config["TRAINING_PARAMS"]
    batch_size = params["BATCH_SIZE"]
    epochs = params["EPOCHS"]
    return batch_size, epochs

def load_sgd_hyperparams(config):
    """Load SGD hyperparams from config file"""
    params = config["OPTIMISER_PARAMS"]
    learning_rate = params["LEARNING_RATE"]
    momentum = params["MOMENTUM"]
    return learning_rate, momentum


