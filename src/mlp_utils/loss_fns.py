import numpy as np

def cross_entropy_loss(y_pred, y_true):
    n = y_pred.shape[0]
    epsilon = 1e-10
    # Clip our predicted values to prevent the chance of our loss function
    # computing log(0) as this would result in an error.
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    loss_per_sample = y_true * np.log(y_pred_clipped)
    return -np.sum(loss_per_sample) / n