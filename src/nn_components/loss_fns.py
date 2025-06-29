import numpy as np

class CrossEntropyLoss:
    """
    Cross-entropy loss
    """

    def __init__(self):
        self.predictions = None
        self.labels = None

    def forward(self, predictions, labels):
        self.predictions = predictions
        self.labels = labels
        n = predictions.shape[0]

        # Clip our predicted values to prevent the chance of our loss function
        # computing log(0) as this would result in an error.
        epsilon = 1e-15
        predictions = np.clip(predictions, epsilon, 1 - epsilon)
        loss_per_sample = labels * np.log(predictions)
        return -np.sum(loss_per_sample) / n

    def backward(self):
        dL_dAL = self.predictions - self.labels
        return dL_dAL