import random
import numpy as np

from src.model import MultiLayerPerceptron
from src.model2  import MultiLayerPerceptron2

from src.utils.config_loader import load_config
from src.data_tools.process_dataset import get_preprocessed_datasets
from src.utils.parameter_loader import load_model_hyperparams, load_trained_parameters
from src.scripts.test_scripts import test_model


def test():
    config = load_config()
    set_seed(config)

    EXPERIMENT_ID = config["TEST_SETTINGS"]["EXPERIMENT_ID"]

    x_test, y_test = get_preprocessed_datasets(mode="TEST")
    (n, d) = x_test.shape
    (n, c) = y_test.shape

    batch_size = config["TRAINING_CONFIG"]["BATCH_SIZE"]
    hidden_layers, hidden_activation, output_activation, init_func = load_model_hyperparams()

    model1 = MultiLayerPerceptron(input_dim=d, output_dim=c, hidden_layers_config=hidden_layers,
                                 hidden_activation_fn=hidden_activation, output_activation_fn=output_activation,
                                 weight_init_strategy=init_func)

    model2 = MultiLayerPerceptron2()

    trained_parameters1 = load_trained_parameters("vOLD")
    trained_parameters2 = load_trained_parameters("vNEW")

    # Todo: fix backpropagation, as params from 2nd layer onwards are incorrect

    model1.set_parameters(trained_parameters1)
    model2.set_parameters(trained_parameters2)

    test_accuracy1 = test_model(model1, x_test, y_test, batch_size)
    test_accuracy2 = test_model(model2, x_test, y_test, batch_size)

    print(f"""
    Model ID: vOLD
    Dataset: MNIST
    Test accuracy: {test_accuracy1:.2f}%
    """)

    print(f"""
    Model ID: vNEW
    Dataset: MNIST
    Test accuracy: {test_accuracy2:.2f}%
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