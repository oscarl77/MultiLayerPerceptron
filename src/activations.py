import numpy as np

def ReLU(Z):
    """ReLU activation function"""
    return np.maximum(0, Z)

def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_shifted)
    sum_exp_Z = np.sum(exp_Z, axis=1, keepdims=True)
    A = exp_Z / sum_exp_Z
    return A

ACTIVATIONS = {
    'ReLU': ReLU,
    'Softmax': softmax
}