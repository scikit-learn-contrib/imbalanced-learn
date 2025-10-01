import numpy as np
from imblearn.metrics._regression import macro_mean_squared_error

def test_macro_mse():
    y_true = np.array([[1, 2], [3, 4]])
    y_pred = np.array([[1, 1], [4, 4]])
    
    result = macro_mean_squared_error(y_true, y_pred)
    expected = np.mean([np.mean([0,1]), np.mean([1,0])])  # manual calculation
    assert np.isclose(result, expected)
