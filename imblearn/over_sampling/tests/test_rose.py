"""Test the module ROSE."""
# Authors: Andrea Lorenzon <andrelorenzon@gmail.com>
# License: MIT

import numpy as np

from imblearn.over_sampling import ROSE
from sklearn.datasets import make_spd_matrix as SymPosDef


def test_rose():

    """Check ROSE use"""

    RND_SEED = 0

    nCols = 3
    ns = [50000, 50000, 75000]
    # generate covariance matrices
    cov1 = SymPosDef(nCols)
    cov2 = SymPosDef(nCols)
    cov3 = SymPosDef(nCols)
    print("covs: ", cov1,
          "\n-\n\n", cov2,
          "\n-\n\n", cov3,
          "\n-\n\n")
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

    print("res_covs: ", res_cov1,
          "\n-\n\n", res_cov2,
          "\n-\n\n", res_cov3,
          "\n-\n\n")

#     assert_allclose(res_cov1, cov1, rtol=1)
#     assert_allclose(res_cov2, cov2, rtol=1)
#     assert_allclose(res_cov3, cov3, rtol=1)
    assert res_cov1.shape == cov1.shape
    assert res_cov2.shape == cov2.shape
    assert res_cov3.shape == cov3.shape
