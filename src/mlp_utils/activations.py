import numpy as np

class ReLU:

    def __init__(self):
        self.Z = None

    def forward(self, Z):
        """
        ReLU activation function.
        :param Z: Pre-activation vector
        :return: THe output vector of the activation function.
        """
        self.Z = Z
        return np.maximum(0, Z)

    def backward(self, dL_dA):
        """
        Derivative of the ReLU activation function.
        :param dL_dA: Derivative of the loss w.r.t. the ReLU output.
        :return: Derivative of the loss w.r.t. the pre-activation.
        """
        dA_dZ = np.where(self.Z > 0, 1, 0)
        return dL_dA * dA_dZ


class Softmax:

    @staticmethod
    def forward(Z):
        """
        Softmax activation function.
        :param Z: Pre-activation vector.
        :return: The output vector of the activation function.
        """
        Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z_shifted)
        sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
        A = exp_Z / sum_exp_Z
        return A

    @staticmethod
    def backward(dL_dA):
        """
        Derivative of Softmax activation function.
        :param dL_dA: Derivative of loss w.r.t. the Softmax output.
        """
        return dL_dA

def _ReLU(Z):
    """
    ReLU activation function
    :param Z: Pre-activation vector
    :return: The output vector of the activation function
    """
    return np.maximum(0, Z)

def d_ReLU(dA, Z):
    """Derivative of ReLU activation function"""
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def softmax(Z):
    """
    Softmax activation function
    :param Z: Pre-activation vector
    :return: The output vector of the activation function.
    """
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    A = exp_Z / sum_exp_Z
    return A

ACTIVATIONS = {
    'RELU': ReLU,
    'ReLU': [_ReLU, d_ReLU],
    'SOFTMAX': Softmax,
    'Softmax': softmax
}
