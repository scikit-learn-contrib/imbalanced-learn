"""Test the module SMOTENC."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

from __future__ import print_function

import numpy as np
from scipy import sparse
from pytest import raises
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.utils.testing import assert_allclose, assert_array_equal
from sklearn.utils.testing import ignore_warnings

from imblearn.over_sampling import SMOTENC

RND_SEED = 0
# Assume original data-set had 2 nominal features:
# one with 3 possible values and one with 2.
# Therefore, after one-hot encoding, values in columns 2-4
# represent first nominal feature and values in columns 5-6
# represent second nominal feature.
X = np.array([[0.11622591, -0.0317206, 0, 0, 1],
              [0.77481731, 0.60935141, 0, 1, 0],
              [1.25192108, -0.22367336, 1, 0, 0],
              [0.53366841, -0.30312976, 1, 0, 0],
              [1.52091956, -0.49283504, 2, 1, 0],
              [-0.28162401, -2.10400981, 0, 0, 1],
              [0.83680821, 1.72827342, 0, 0, 1],
              [0.3084254, 0.33299982, 0, 1, 0],
              [0.70472253, -0.73309052, 1, 0, 0],
              [0.28893132, -0.38761769, 1, 0, 0],
              [1.15514042, 0.0129463, 0, 1, 0],
              [0.88407872, 0.35454207, 0, 0, 1],
              [1.31301027, -0.92648734, 0, 0, 1],
              [-1.11515198, -0.93689695, 0, 1, 0],
              [-0.18410027, -0.45194484, 1, 0, 0],
              [0.9281014, 0.53085498, 1, 0, 0],
              [-0.14374509, 0.27370049, 2, 1, 0],
              [-0.41635887, -0.38299653, 0, 0, 1],
              [0.08711622, 0.93259929, 0, 0, 1],
              [1.70580611, -0.11219234, 2, 1, 0]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4
X_indices = [2, 3, 4]


def test_smote_nc():
    smote = SMOTENC(random_state=RND_SEED, categorical_features=X_indices)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_resampled, y_resampled = smote.fit_resample(sparse.csr_matrix(X), Y)
