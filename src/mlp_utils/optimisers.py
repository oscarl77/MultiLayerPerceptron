import numpy as np

class SGDOptimiser:
    """
    Stochastic Gradient Descent Optimiser with momentum
    """

    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}

    def step(self, parameters: dict, gradients: dict):
        """
        Update model's parameters using SGD with momentum
        :param parameters: dict of model's current weights and biases.
        :param gradients: dict of gradients for each weight and bias.
        """
        if not self.velocities:
            for param in parameters.keys():
                self.velocities[param] = np.zeros_like(parameters[param])

        for param in parameters.keys():
            self.velocities[param] = self.momentum * self.velocities[param] + gradients[param]
            parameters[param] = parameters[param] - self.learning_rate * self.velocities[param]

