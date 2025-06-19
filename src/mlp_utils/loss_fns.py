import numpy as np

def cross_entropy_loss(y_pred, y_true):
    n = y_pred.shape[0]
    loss_per_sample = y_true * np.log(y_pred)
    return -np.sum(loss_per_sample) / n