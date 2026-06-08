import numpy as np

def macro_mean_squared_error(y_true, y_pred):
    """
    Compute the macro-averaged mean squared error.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_outputs)
        True values.
    y_pred : array-like of shape (n_samples, n_outputs)
        Predicted values.

    Returns
    -------
    float
        Macro-averaged MSE over all outputs.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Compute MSE for each output
    mse_per_output = np.mean((y_true - y_pred) ** 2, axis=0)

    # Return macro-average
    return np.mean(mse_per_output)
