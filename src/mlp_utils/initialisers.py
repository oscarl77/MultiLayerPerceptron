import numpy as np

def init_random_uniform(input_dim, output_dim, scale=0.1):
    return np.random.uniform(low=-scale, high=scale, size=(input_dim, output_dim))

def init_xavier_uniform(input_dim, output_dim):
    limit = np.sqrt(6. / (input_dim + output_dim))
    return np.random.uniform(low=-limit, high=limit, size=(input_dim, output_dim))

def init_kaiming(input_dim, output_dim):
    return np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)

INIT_STRATEGIES = {
    'random_uniform': init_random_uniform,
    'xavier_uniform': init_xavier_uniform,
    'kaiming': init_kaiming,
}