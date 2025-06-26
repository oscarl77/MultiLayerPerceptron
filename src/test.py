import random
import numpy as np

from src.model  import MultiLayerPerceptron

from src.utils.config_loader import load_config
from src.data_tools.process_dataset import get_preprocessed_datasets
from src.utils.parameter_loader import load_trained_parameters
from src.scripts.test_scripts import test_model

def test():
    """
    Testing environment to test the model.
    """
    config = load_config()
    set_seed(config)

    EXPERIMENT_ID = config["TEST_SETTINGS"]["EXPERIMENT_ID"]

    x_test, y_test = get_preprocessed_datasets(mode="TEST")


    batch_size = config["TRAINING_CONFIG"]["BATCH_SIZE"]

    model = MultiLayerPerceptron()

    trained_parameters = load_trained_parameters(EXPERIMENT_ID)

    model.set_parameters(trained_parameters)

    test_accuracy = test_model(model, x_test, y_test, batch_size)

    print(f"""
    Model ID: {EXPERIMENT_ID}
    Dataset: MNIST
    Test accuracy: {test_accuracy:.2f}%
        """)

def set_seed(config):
    """
    Set fixed random seeds throughout project for training reproducibility.
    :param config: dict containing all project configurations
    """
    seed = config["TRAINING_CONFIG"]["RANDOM_SEED"]
    random.seed(seed)
    np.random.seed(seed)

if __name__ == '__main__':
    test()