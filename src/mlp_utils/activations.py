import numpy as np

def ReLU(Z):
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
    'ReLU': [ReLU, d_ReLU],
    'Softmax': softmax
}
