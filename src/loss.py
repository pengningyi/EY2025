import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    return np.where(np.abs(error) < delta, 
                    0.5 * error**2, 
                    delta * (np.abs(error) - 0.5 * delta)).mean()

def custom_weighted_rmse(y_true, y_pred, weight_factor=2.0):
    errors = y_true - y_pred
    return np.sqrt(((errors ** 2) * weight_factor).mean())