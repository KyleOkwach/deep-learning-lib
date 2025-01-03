import numpy as np

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_prime(y_true, y_pred):
    # Derivative of the loss function with respect to the prediction (dL/dy_pred)
    return 2 * (y_pred - y_true) / y_true.size