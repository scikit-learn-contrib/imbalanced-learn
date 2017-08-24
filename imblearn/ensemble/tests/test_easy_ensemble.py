"""Test the module easy ensemble."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal

from imblearn.ensemble import EasyEnsemble

iris = load_iris()

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.5220963, 0.11349303], [0.59091459, 0.40692742],
              [1.10915364, 0.05718352], [0.22039505, 0.26469445],
              [1.35269503, 0.44812421], [0.85117925, 1.0185556],
              [-2.10724436, 0.70263997], [-0.23627356, 0.30254174],
              [-1.23195149, 0.15427291], [-0.58539673, 0.62515052]])
Y = np.array([1, 2, 2, 2, 1, 0, 1, 1, 1, 0])


def test_ee_init():
    # Define a ratio
    ratio = 1.
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)

    assert ee.ratio == ratio
    assert ee.replacement is False
    assert ee.n_subsets == 10
    assert ee.random_state == RND_SEED


def test_fit_sample_auto():
    # Define the ratio parameter
    ratio = 'auto'

    # Create the sampling object
    ee = EasyEnsemble(
        ratio=ratio, random_state=RND_SEED, return_indices=True, n_subsets=3)

    # Get the different subset
    X_resampled, y_resampled, idx_under = ee.fit_sample(X, Y)

    X_gt = np.array([[[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [1.35269503, 0.44812421], [0.5220963, 0.11349303],
                      [1.10915364, 0.05718352], [0.22039505, 0.26469445]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [-2.10724436, 0.70263997],
                      [0.22039505, 0.26469445], [1.10915364, 0.05718352]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [0.5220963, 0.11349303],
                      [1.10915364, 0.05718352], [0.59091459, 0.40692742]]])
    y_gt = np.array([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2],
                     [0, 0, 1, 1, 2, 2]])
    idx_gt = np.array([[5, 9, 4, 0, 2, 3], [5, 9, 8, 6, 3, 2],
                       [5, 9, 8, 0, 2, 1]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_fit_sample_half():
    # Define the ratio parameter
    ratio = 0.6

    # Create the sampling object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED, n_subsets=3)

    # Get the different subset
    X_resampled, y_resampled = ee.fit_sample(X, Y)

    X_gt = np.array([[[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [1.35269503, 0.44812421], [0.5220963, 0.11349303],
                      [-2.10724436, 0.70263997], [1.10915364, 0.05718352],
                      [0.22039505, 0.26469445], [0.59091459, 0.40692742]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [-2.10724436, 0.70263997],
                      [0.5220963, 0.11349303], [0.22039505, 0.26469445],
                      [1.10915364, 0.05718352], [0.59091459, 0.40692742]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [0.5220963, 0.11349303],
                      [1.35269503, 0.44812421], [1.10915364, 0.05718352],
                      [0.59091459, 0.40692742], [0.22039505, 0.26469445]]])
    y_gt = np.array([[0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 1, 1, 1, 2, 2, 2],
                     [0, 0, 1, 1, 1, 2, 2, 2]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_random_state_none():
    # Define the ratio parameter
    ratio = 'auto'

    # Create the sampling object
    ee = EasyEnsemble(ratio=ratio, random_state=None)

    # Get the different subset
    X_resampled, y_resampled = ee.fit_sample(X, Y)
