"""Test the module SMOTENC."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

import pytest

import numpy as np

from imblearn.over_sampling import SMOTENC


def test_smote_nc():
    X = np.empty((30, 4), dtype=object)
    X[:, :2] = np.random.randn(30, 2)
    X[:, 2] = np.random.choice(['a', 'b', 'c'], size=30).astype(object)
    X[:, 3] = np.random.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    smote = SMOTENC(random_state=0, categorical_features=[2, 3])
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(X_resampled)
    # X_resampled, y_resampled = smote.fit_resample(sparse.csr_matrix(X), Y)
