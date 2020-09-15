"""Test the module ROSE."""
# Authors: Andrea Lorenzon <andrelorenzon@gmail.com>
# License: MIT

import numpy as np

from imblearn.over_sampling import ROSE


def test_rose():

    """Check ROSE use"""

    RND_SEED = 0

    X = data = np.array([
    [1., 1., 1., 0],
    [2., 2., 2., 0],
    [3., 3., 3., 0],
    [0.9, 0.9, 0.9, 0],
    [1.8, 1.8, 1.8, 0],
    [2.7, 2.7, 2.7, 0],
    [1.1, 1.1, 1.1, 0],
    [2.2, 2.2, 2.2, 0],
    [3.3, 3.3, 3.3, 0]])
    Y = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3] )
    X_res, y_res = ROSE(random_state=RND_SEED).fit_resample(X, Y)