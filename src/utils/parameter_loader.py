from src.utils.config_loader import load_config
from src.activations import ACTIVATIONS
from src.initialisers import INIT_STRATEGIES

def load_model_hyperparams():
    """Load """
    config = load_config()
    params = config["HYPERPARAMETERS"]
    hidden_layers = params["HIDDEN_LAYERS"]
    init_func = INIT_STRATEGIES[params["INIT_STRATEGY"]]
    hidden_activation = ACTIVATIONS[params["HIDDEN_ACTIVATION"]]
    output_activation = ACTIVATIONS[params["OUTPUT_ACTIVATION"]]
    return hidden_layers, hidden_activation, output_activation, init_func

