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

    def backward(self, dL_dA, layer_idx):
        batch_size = dL_dA.shape[0]
        dL_dA = self.activation_func.backward(dL_dA)
        self.dL_dW = self.A_prev.T @ dL_dA / batch_size
        self.dL_db = np.sum(dL_dA, axis=0, keepdims=True) / batch_size
        dL_dA_prev = dL_dA @ self.weights.T
        return dL_dA_prev, self.dL_dW, self.dL_db

    @staticmethod
    def _weight_init():
        config = load_config()
        params = config["MODEL_CONFIG"]
        return INIT_STRATEGIES[params["INIT_STRATEGY"]]
