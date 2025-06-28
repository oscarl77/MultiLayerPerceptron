import numpy as np

class CrossEntropyLoss:
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

def cross_entropy_loss(y_pred, y_true):
    n = y_pred.shape[0]
    epsilon = 1e-15
    # Clip our predicted values to prevent the chance of our loss function
    # computing log(0) as this would result in an error.
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss_per_sample = y_true * np.log(y_pred_clipped)
    return -np.sum(loss_per_sample) / n