import numpy as np

from src.mlp_utils.initialisers import INIT_STRATEGIES
from src.utils.config_loader import load_config

class DenseLayer:

    def __init__(self, input_dim, output_dim, activation_func):
        init_func = self._weight_init()
        self.output_dim = output_dim
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
        self.training = True

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
        if not self.training:
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
        if not self.training:
            return dL_dA, None, None

        dL_dA_prev = dL_dA * self.mask

        return dL_dA_prev, None, None

class BatchNormLayer:

    def __init__(self, input_dim, momentum=0.9):
        """Initialise batch normalization layer.
        :param input_dim: Input data from previous layer.
        :param momentum: Momentum for batch normalization.
        """

        self.gamma = np.ones((1, input_dim)) # scale
        self.beta = np.zeros((1, input_dim)) # shift

        self.moving_mean = np.zeros((1, input_dim))
        self.moving_variance = np.ones((1, input_dim))

        self.momentum = momentum
        self.epsilon = 1e-8
        self.training = True

        self.X_normalised = None
        self.batch_mean = None
        self.batch_variance = None
        self.X = None

    def get_params(self):
        """Return the learnable parameters for the current layer."""
        return self.gamma, self.beta

    def set_params(self, gamma, beta):
        """Set the learnable parameters for the current layer."""
        self.gamma = gamma
        self.beta = beta

    def forward(self, X):
        """
        Compute the forward pass for the current layer.
        :param X: Input data from the previous layer.
        :return: The normalised, scaled and shifted input data.
        """
        self.X = X

        if self.training:
            # Calculate mean and variance for current batch
            self.batch_mean = np.mean(X, axis=0, keepdims=True)
            self.batch_variance = np.var(X, axis=0, keepdims=True)

            # Update moving averages
            self.moving_mean = self.momentum * self.moving_mean + (1 - self.momentum) * self.batch_mean
            self.moving_variance = self.momentum * self.moving_variance + (1 - self.momentum) * self.batch_variance

            # Normalise batch
            self.X_normalised = (X - self.batch_mean) / np.sqrt(self.batch_variance + self.epsilon)

        else:
            # Normalise batch using learned moving averages and variance
            self.X_normalised = (X - self.moving_mean) / np.sqrt(self.moving_variance + self.epsilon)

        # Scale and shift the normalised batch
        X_scaled_shifted = self.gamma * self.X_normalised + self.beta
        return X_scaled_shifted

    def backward(self, dL_dA):
        """
        Compute the backward pass for the current layer.
        :param dL_dA: Gradient of the loss w.r.t. the output of the current layer.
        :return: Tuple containing:
                    - dL_dX: Gradient of loss w.r.t. the input.
                    - dL_dgamma: Gradient of the loss w.r.t. gamma.
                    - dL_dbeta: Gradient of the loss w.r.t. beta.
        """
        inverse_std_dev = 1. / np.sqrt(self.batch_variance + self.epsilon)

        # Compute gradients for gamma and beta
        dL_dgamma = np.sum(dL_dA * self.X_normalised, axis=0, keepdims=True)
        dL_dbeta = np.sum(dL_dA, axis=1, keepdims=True)

        # Compute gradient of loss w.r.t. the normalised input.
        dL_dX_normalised = dL_dA * self.gamma
        # Compute gradient of the loss w.r.t. the variance.
        dL_dvariance = self._compute_dL_dvariance(dL_dX_normalised, inverse_std_dev)
        # Compute the gradient of the loss w.r.t. the mean
        dL_dmean = self._compute_dL_dmean(dL_dX_normalised, inverse_std_dev, dL_dvariance)
        # Compute the gradient of the loss w.r.t. the input.
        dL_dX = self._compute_dL_dX(dL_dX_normalised, inverse_std_dev, dL_dvariance, dL_dmean)

        return dL_dX, dL_dgamma, dL_dbeta

    def _compute_dL_dvariance(self, dL_dX_norm, inverse_std_dev):
        """
        Compute dL_dvariance = (dL/dX_normalised) * (dX_normalised/dvariance)
        :param dL_dX_norm: Gradient of the loss w.r.t. the normalised input.
        :param inverse_std_dev: Inverse formula for standard deviation.
        :return: dL_dvariance: Gradient of the loss w.r.t. the variance.
        """
        x_minus_mean = self.X - self.batch_mean
        dX_norm_dvariance = x_minus_mean * (-0.5) * (inverse_std_dev ** 3)
        dL_dvariance = np.sum(dL_dX_norm * dX_norm_dvariance, axis=0, keepdims=True)
        return dL_dvariance

    def _compute_dL_dmean(self, dL_dX_norm, inverse_std_dev, dL_dvariance):
        """
        Compute dL_dmean = (dL/dX_normalised) * (dX_normalised/dmean)
        (Note that the mean affects the loss via two 'paths'.)
        :param dL_dX_norm: Gradient of the loss w.r.t. the normalised input.
        :param inverse_std_dev: Inverse formula for standard deviation.
        :param dL_dvariance: Gradient of the loss w.r.t. the variance.
        :return: dL_dmean: Gradient of the loss w.r.t. the mean.
        """

        # First path is direct from X_norm to mean
        # (dL / dX_norm) * (dX_norm / dmean)
        dX_norm_dmean = -inverse_std_dev
        dL_dmean_path1 = np.sum(dL_dX_norm * dX_norm_dmean, axis=0, keepdims=True)

        # Second path is indirect from variance to mean
        # (dL / dvariance) * (dvariance / dmean)
        dvariance_dmean = np.mean(-2.0 * (self.X - self.batch_mean), axis=0, keepdims=True)
        dL_dmean_path2 = dL_dvariance * dvariance_dmean

        dL_dmean = dL_dmean_path1 + dL_dmean_path2
        return dL_dmean

    def _compute_dL_dX(self, dL_dX_norm, inverse_std_dev, dL_dmean, dL_dvariance):
        """
        Compute dL_dX = (dL/dX_normalised) * (dX_normalised/dX)
        :param dL_dX_norm: Gradient of the loss w.r.t. the normalised input.
        :param inverse_std_dev: Inverse formula for standard deviation.
        :param dL_dmean: Gradient of the loss w.r.t. the mean.
        :param dL_dvariance: Gradient of the loss w.r.t. the variance.
        :return: dL_dX: Gradient of the loss w.r.t. the input.
        """
        batch_size = dL_dX_norm.shape[0]

        dL_dX_path1 = dL_dX_norm * inverse_std_dev
        dL_dX_path2 = dL_dvariance * (2.0/batch_size) * (self.X - self.batch_mean)
        dL_dX_path3 = dL_dmean / batch_size

        dL_dX = dL_dX_path1 + dL_dX_path2 + dL_dX_path3
        return dL_dX