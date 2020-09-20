"""Test the module ROSE."""
# Authors: Andrea Lorenzon <andrelorenzon@gmail.com>
# License: MIT

import numpy as np

from imblearn.over_sampling import ROSE
from sklearn.datasets import make_spd_matrix as SymPosDef
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal


def test_rose():

    """Check ROSE use"""

    RND_SEED = 0

    nCols = 3
    ns = [50000, 50000, 75000]
    # generate covariance matrices
    cov1 = SymPosDef(nCols)
    cov2 = SymPosDef(nCols)
    cov3 = SymPosDef(nCols)

    # generate data blobs
    cl1 = np.array(np.random.multivariate_normal([1, 1, 1],
                                                 cov=cov1,
                                                 size=ns[0]))
    cl2 = np.array(np.random.multivariate_normal([7, 7, 7],
                                                 cov=cov2,
                                                 size=ns[1]))
    cl3 = np.array(np.random.multivariate_normal([2, 9, 9],
                                                 cov=cov3,
                                                 size=ns[2]))
    # assemble dataset
    X = np.vstack((cl1, cl2, cl3))
    y = np.hstack((np.array([1] * ns[0]),
                   np.array([2] * ns[1]),
                   np.array([3] * ns[2])))

    r = ROSE(random_state=RND_SEED)
    res, lab = r.fit_resample(X, y)

    # compute and check similarity of covariance matrices
    res_cov1 = np.cov(res[lab == 1], rowvar=False)
    res_cov2 = np.cov(res[lab == 2], rowvar=False)
    res_cov3 = np.cov(res[lab == 3], rowvar=False)

    assert res_cov1.shape == cov1.shape
    assert res_cov2.shape == cov2.shape
    assert res_cov3.shape == cov3.shape


def test_rose_resampler():
    """Test ROSE resampled data matches expectation."""

    RND_SEED = 0
    X = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
        ]
    )
    Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    R_TOL = 1e-4

    rose = ROSE(random_state=RND_SEED)
    X_resampled, y_resampled = rose.fit_resample(X, Y)

    X_gt = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
            [1.39400832, 0.94383454],
            [2.67881738, -0.36918919],
            [1.80801323, -0.96629007],
            [0.06244814, 0.07625536],
        ]
    )

    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
         0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)
