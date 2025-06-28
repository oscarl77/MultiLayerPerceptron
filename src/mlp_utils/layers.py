import numpy as np

from src.mlp_utils.initialisers import INIT_STRATEGIES
from src.utils.config_loader import load_config

class DenseLayer:

    def __init__(self, input_dim, output_dim, activation_func):
        init_func = self._weight_init()
        self.weights = init_func(input_dim, output_dim)
        self.biases = np.zeros(output_dim)
        self.activation_func = activation_func
        self.Z = None
        self.A = None

        self.A_prev = None
        self.dL_dW = None
        self.dL_db = None

    def get_params(self):
        return self.weights, self.biases

    def set_params(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, X):
        self.A_prev = X
        self.Z = X @ self.weights + self.biases
        self.A = self.activation_func.forward(self.Z)
        return self.A

    def backward(self, dL_dA):
        batch_size = dL_dA.shape[0]
        dL_dA = self.activation_func.backward(dL_dA)

        self.dL_dW = self.A_prev.T @ dL_dA / batch_size
        self.dL_db = np.sum(dL_dA, axis=0, keepdims=True) / batch_size

        dL_dA_prev = dL_dA @ self.weights.T
        return dL_dA_prev, self.dL_dW, self.dL_db

    @staticmethod
    def _weight_init():
        config = load_config()
        strategy = config["INIT_CONFIG"]
        return INIT_STRATEGIES[strategy]

class DropoutLayer:

    def __init__(self, rate):
        """
        Initialise dropout layer.
        :param rate: The dropout rate, the probability of an input value being
        set to 0.
        """
        if not (0.0 <= rate <= 1.0):
            raise ValueError("Rate must be between 0.0 and 1.0")
        self.rate = rate
        self.mask = None
        self.enabled = True

    @staticmethod
    def get_params():
        """
        Get method to maintain consistency with DenseLayer.
        Dropout layers do not have parameters to retrieve.
        """
        return None, None

    @staticmethod
    def set_params(weights, biases):
        """
        Set method to maintain consistency with DenseLayer.
        Dropout layers do not have parameters to set.
        """
        pass

    def forward(self, X):
        """
        Compute the forward pass for the current layer.
        :param X: Input data from previous layer.
        :return: The output data, i.e. the input with the mask applied if
        dropout is enabled, otherwise return the input data.
        """
        if not self.enabled:
            return X

        self.mask = (np.random.rand(*X.shape) > self.rate) / (1.0 - self.rate)

        return X * self.mask

    def backward(self, dL_dA):
        """
        Compute the backward pass for the current layer.
        :param dL_dA: Gradient of the loss w.r.t. the output of the current layer.
        :return: Tuple containing:
                    - dL_dA: Gradient of loss w.r.t. output of the current layer
                    with applied mask if enabled, otherwise the original gradient.
                    - dL_dW: None, as a dropout layer has no parameters.
                    - dL_db: None, as a dropout layer has no parameters.
        """
        if not self.enabled:
            return dL_dA, None, None

        dL_dA_prev = dL_dA * self.mask

        return dL_dA_prev, None, None


