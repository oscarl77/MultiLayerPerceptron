import numpy as np

class ReLU:

    def __init__(self):
        self.Z = None

    def forward(self, Z):
        """
        ReLU activation function.
        :param Z: Pre-activation vector
        :return: A: The output vector of the activation function.
        """
        self.Z = Z
        A = np.maximum(0, Z)
        return A

    def backward(self, dL_dA):
        """
        Derivative of the ReLU activation function.
        :param dL_dA: Derivative of the loss w.r.t. the ReLU output.
        :return: Derivative of the loss w.r.t. the pre-activation.
        """
        dA_dZ = np.where(self.Z > 0, 1, 0)
        dL_dZ = dL_dA * dA_dZ
        return dL_dZ


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

ACTIVATIONS = {
    'RELU': ReLU,
    'SOFTMAX': Softmax,
}
