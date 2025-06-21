import numpy as np

def init_random_uniform(input_dim, output_dim, scale=0.1):
    """
    Initialise weights using random uniform distribution.
    :param input_dim: Dimension of input vector.
    :param output_dim: Dimension of output vector.
    :param scale: Range of values to initialise weights from.
    :return: Array of shape (input_dim, output_dim) of initialised weights.
    """
    return np.random.uniform(low=-scale, high=scale, size=(input_dim, output_dim))

def init_xavier_uniform(input_dim, output_dim):
    """
    Initialise weights using xavier uniform distribution.
    :param input_dim: Dimension of input vector.
    :param output_dim: Dimension of output vector.
    :return: Array of shape (input_dim, output_dim) of initialised weights.
    """
    limit = np.sqrt(6. / (input_dim + output_dim))
    return np.random.uniform(low=-limit, high=limit, size=(input_dim, output_dim))

def init_kaiming(input_dim, output_dim):
    """
    Initialise weights using kaiming (He) uniform distribution.
    :param input_dim: Dimension of input vector.
    :param output_dim: Dimension of output vector.
    :return: Array of shape (input_dim, output_dim) of initialised weights.
    """
    return np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)

INIT_STRATEGIES = {
    'random_uniform': init_random_uniform,
    'xavier_uniform': init_xavier_uniform,
    'kaiming': init_kaiming,
}