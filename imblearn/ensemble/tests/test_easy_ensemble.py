"""Test the module easy ensemble."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier
from sklearn.utils.testing import assert_array_equal
from sklearn.model_selection import train_test_split

from imblearn.ensemble import EasyEnsemble
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.datasets import make_imbalance

iris = load_iris()

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.5220963, 0.11349303], [0.59091459, 0.40692742], [
    1.10915364, 0.05718352
], [0.22039505, 0.26469445], [1.35269503, 0.44812421], [0.85117925, 1.0185556],
              [-2.10724436, 0.70263997], [-0.23627356, 0.30254174],
              [-1.23195149, 0.15427291], [-0.58539673, 0.62515052]])
Y = np.array([1, 2, 2, 2, 1, 0, 1, 1, 1, 0])


def test_ee_init():
    # Define a sampling_strategy
    sampling_strategy = 1.
    ee = EasyEnsemble(
        sampling_strategy=sampling_strategy, random_state=RND_SEED)

    assert ee.sampling_strategy == sampling_strategy
    assert ee.replacement is False
    assert ee.n_subsets == 10
    assert ee.random_state == RND_SEED


def test_fit_sample_auto():
    # Define the sampling_strategy parameter
    sampling_strategy = 'auto'

    # Create the sampling object
    ee = EasyEnsemble(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        return_indices=True,
        n_subsets=3)

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
    # Define the sampling_strategy parameter
    sampling_strategy = {0: 2, 1: 3, 2: 3}

    # Create the sampling object
    ee = EasyEnsemble(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        n_subsets=3)

    # Get the different subset
    X_resampled, y_resampled = ee.fit_sample(X, Y)

    X_gt = np.array([[[-0.58539673, 0.62515052], [0.85117925, 1.0185556],
                      [1.35269503, 0.44812421], [-1.23195149, 0.15427291],
                      [0.5220963, 0.11349303], [1.10915364, 0.05718352],
                      [0.59091459, 0.40692742], [0.22039505, 0.26469445]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [1.35269503, 0.44812421], [-2.10724436, 0.70263997],
                      [-1.23195149, 0.15427291], [0.59091459, 0.40692742],
                      [0.22039505, 0.26469445], [1.10915364, 0.05718352]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [0.5220963, 0.11349303],
                      [1.35269503, 0.44812421], [1.10915364, 0.05718352],
                      [0.59091459, 0.40692742], [0.22039505, 0.26469445]]])
    y_gt = np.array([[0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 1, 1, 1, 2, 2, 2],
                     [0, 0, 1, 1, 1, 2, 2, 2]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_random_state_none():
    # Define the sampling_strategy parameter
    sampling_strategy = 'auto'

    # Create the sampling object
    ee = EasyEnsemble(sampling_strategy=sampling_strategy, random_state=None)

    # Get the different subset
    X_resampled, y_resampled = ee.fit_sample(X, Y)


@pytest.mark.parametrize("n_estimators", [10, 20, 30])
@pytest.mark.parametrize("adaboost_estimator", [
    AdaBoostClassifier(n_estimators=50),
    AdaBoostClassifier(n_estimators=100)])
def test_easy_ensemble_classifier(n_estimators, adaboost_estimator):
    # Check classification for various parameter settings.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20,
                           1: 25,
                           2: 50},
        random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    eec = EasyEnsembleClassifier(n_estimators=n_estimators,
                                 adaboost_estimator=adaboost_estimator,
                                 n_jobs=-1,
                                 random_state=RND_SEED)
    eec.fit(X_train, y_train).score(X_test, y_test)
    assert len(eec.estimators_) == n_estimators
    for est in eec.estimators_:
        assert (len(est.named_steps['classifier']) ==
                adaboost_estimator.n_estimators)
