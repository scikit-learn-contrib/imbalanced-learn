"""Test the module repeated edited nearest neighbour."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_equal, assert_raises, assert_warns)
from sklearn.utils.estimator_checks import check_estimator
from sklearn.neighbors import NearestNeighbors

from imblearn.under_sampling import AllKNN

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[-0.12840393, 0.66446571], [1.32319756, -0.13181616],
              [0.04296502, -0.37981873], [0.83631853, 0.18569783],
              [1.02956816, 0.36061601], [1.12202806, 0.33811558],
              [-0.53171468, -0.53735182], [1.3381556, 0.35956356],
              [-0.35946678, 0.72510189], [1.32326943, 0.28393874],
              [2.94290565, -0.13986434], [0.28294738, -1.00125525],
              [0.34218094, -0.58781961], [-0.88864036, -0.33782387],
              [-1.10146139, 0.91782682], [-0.7969716, -0.50493969],
              [0.73489726, 0.43915195], [0.2096964, -0.61814058],
              [-0.28479268, 0.70459548], [1.84864913, 0.14729596],
              [1.59068979, -0.96622933], [0.73418199, -0.02222847],
              [0.50307437, 0.498805], [0.84929742, 0.41042894],
              [0.62649535, 0.46600596], [0.79270821, -0.41386668],
              [1.16606871, -0.25641059], [1.57356906, 0.30390519],
              [1.0304995, -0.16955962], [1.67314371, 0.19231498],
              [0.98382284, 0.37184502], [0.48921682, -1.38504507],
              [-0.46226554, -0.50481004], [-0.03918551, -0.68540745],
              [0.24991051, -1.00864997], [0.80541964, -0.34465185],
              [0.1732627, -1.61323172], [0.69804044, 0.44810796],
              [-0.5506368, -0.42072426], [-0.34474418, 0.21969797]])
Y = np.array([
    1, 2, 2, 2, 1, 1, 0, 2, 1, 1, 1, 2, 2, 0, 1, 2, 1, 2, 1, 1, 2, 2, 1, 1, 1,
    2, 2, 2, 2, 1, 1, 2, 0, 2, 2, 2, 2, 1, 2, 0
])
R_TOL = 1e-4


def test_allknn_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(AllKNN)


def test_allknn_init():
    """Test the initialisation of the object"""

    # Define a ratio
    allknn = AllKNN(random_state=RND_SEED)

    assert_equal(allknn.n_neighbors, 3)
    assert_equal(allknn.kind_sel, 'all')
    assert_equal(allknn.n_jobs, -1)
    assert_equal(allknn.random_state, RND_SEED)


def test_allknn_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    allknn = AllKNN(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, allknn.fit, X, y_single_class)


def test_allknn_fit():
    """Test the fitting method"""

    # Create the object
    allknn = AllKNN(random_state=RND_SEED)
    # Fit the data
    allknn.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(allknn.min_c_, 0)
    assert_equal(allknn.maj_c_, 2)
    assert_equal(allknn.stats_c_[0], 4)
    assert_equal(allknn.stats_c_[1], 16)
    assert_equal(allknn.stats_c_[2], 20)


def test_allknn_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    allknn = AllKNN(random_state=RND_SEED)
    assert_raises(RuntimeError, allknn.sample, X, Y)


def test_allknn_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    allknn = AllKNN(random_state=RND_SEED)
    X_resampled, y_resampled = allknn.fit_sample(X, Y)

    X_gt = np.array([[-0.53171468, -0.53735182], [-0.88864036, -0.33782387],
                     [-0.46226554, -0.50481004], [-0.34474418, 0.21969797],
                     [1.02956816, 0.36061601], [1.12202806, 0.33811558],
                     [-1.10146139, 0.91782682], [0.73489726, 0.43915195],
                     [0.50307437, 0.498805], [0.84929742, 0.41042894],
                     [0.62649535, 0.46600596], [0.98382284, 0.37184502],
                     [0.69804044, 0.44810796], [0.04296502, -0.37981873],
                     [0.28294738, -1.00125525], [0.34218094, -0.58781961],
                     [0.2096964, -0.61814058], [1.59068979, -0.96622933],
                     [0.73418199, -0.02222847], [0.79270821, -0.41386668],
                     [1.16606871, -0.25641059], [1.0304995, -0.16955962],
                     [0.48921682, -1.38504507], [-0.03918551, -0.68540745],
                     [0.24991051, -1.00864997], [0.80541964, -0.34465185],
                     [0.1732627, -1.61323172]])
    y_gt = np.array([
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_allclose(y_resampled, y_gt, rtol=R_TOL)


def test_allknn_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    allknn = AllKNN(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = allknn.fit_sample(X, Y)

    X_gt = np.array([[-0.53171468, -0.53735182], [-0.88864036, -0.33782387],
                     [-0.46226554, -0.50481004], [-0.34474418, 0.21969797],
                     [1.02956816, 0.36061601], [1.12202806, 0.33811558],
                     [-1.10146139, 0.91782682], [0.73489726, 0.43915195],
                     [0.50307437, 0.498805], [0.84929742, 0.41042894],
                     [0.62649535, 0.46600596], [0.98382284, 0.37184502],
                     [0.69804044, 0.44810796], [0.04296502, -0.37981873],
                     [0.28294738, -1.00125525], [0.34218094, -0.58781961],
                     [0.2096964, -0.61814058], [1.59068979, -0.96622933],
                     [0.73418199, -0.02222847], [0.79270821, -0.41386668],
                     [1.16606871, -0.25641059], [1.0304995, -0.16955962],
                     [0.48921682, -1.38504507], [-0.03918551, -0.68540745],
                     [0.24991051, -1.00864997], [0.80541964, -0.34465185],
                     [0.1732627, -1.61323172]])
    y_gt = np.array([
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2
    ])
    idx_gt = np.array([
        6, 13, 32, 39, 4, 5, 14, 16, 22, 23, 24, 30, 37, 2, 11, 12, 17, 20, 21,
        25, 26, 28, 31, 33, 34, 35, 36
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_allclose(y_resampled, y_gt, rtol=R_TOL)
    assert_allclose(idx_under, idx_gt, rtol=R_TOL)


def test_allknn_fit_sample_mode():
    """Test the fit sample routine using the mode as selection"""

    # Resample the data
    allknn = AllKNN(random_state=RND_SEED, kind_sel='mode')
    X_resampled, y_resampled = allknn.fit_sample(X, Y)

    X_gt = np.array([[-0.53171468, -0.53735182], [-0.88864036, -0.33782387],
                     [-0.46226554, -0.50481004], [-0.34474418, 0.21969797],
                     [-0.12840393, 0.66446571], [1.02956816, 0.36061601],
                     [1.12202806, 0.33811558], [-0.35946678, 0.72510189],
                     [-1.10146139, 0.91782682], [0.73489726, 0.43915195],
                     [-0.28479268, 0.70459548], [0.50307437, 0.498805],
                     [0.84929742, 0.41042894], [0.62649535, 0.46600596],
                     [0.98382284, 0.37184502], [0.69804044, 0.44810796],
                     [1.32319756, -0.13181616], [0.04296502, -0.37981873],
                     [0.28294738, -1.00125525], [0.34218094, -0.58781961],
                     [0.2096964, -0.61814058], [1.59068979, -0.96622933],
                     [0.73418199, -0.02222847], [0.79270821, -0.41386668],
                     [1.16606871, -0.25641059], [1.0304995, -0.16955962],
                     [0.48921682, -1.38504507], [-0.03918551, -0.68540745],
                     [0.24991051, -1.00864997], [0.80541964, -0.34465185],
                     [0.1732627, -1.61323172]])
    y_gt = np.array([
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2
    ])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_allknn_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    allknn = AllKNN(random_state=RND_SEED)
    allknn.fit(X, Y)
    assert_raises(RuntimeError, allknn.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_continuous_error():
    """Test either if an error is raised when the target are continuous
    type"""

    # continuous case
    y = np.linspace(0, 1, 40)
    ann = AllKNN(random_state=RND_SEED)
    assert_warns(UserWarning, ann.fit, X, y)


def test_allknn_fit_sample_with_nn_object():
    """Test the fit sample routine using a NN object"""

    # Resample the data
    nn = NearestNeighbors(n_neighbors=4)
    allknn = AllKNN(n_neighbors=nn, random_state=RND_SEED, kind_sel='mode')
    X_resampled, y_resampled = allknn.fit_sample(X, Y)

    X_gt = np.array([[-0.53171468, -0.53735182], [-0.88864036, -0.33782387],
                     [-0.46226554, -0.50481004], [-0.34474418, 0.21969797],
                     [-0.12840393, 0.66446571], [1.02956816, 0.36061601],
                     [1.12202806, 0.33811558], [-0.35946678, 0.72510189],
                     [-1.10146139, 0.91782682], [0.73489726, 0.43915195],
                     [-0.28479268, 0.70459548], [0.50307437, 0.498805],
                     [0.84929742, 0.41042894], [0.62649535, 0.46600596],
                     [0.98382284, 0.37184502], [0.69804044, 0.44810796],
                     [1.32319756, -0.13181616], [0.04296502, -0.37981873],
                     [0.28294738, -1.00125525], [0.34218094, -0.58781961],
                     [0.2096964, -0.61814058], [1.59068979, -0.96622933],
                     [0.73418199, -0.02222847], [0.79270821, -0.41386668],
                     [1.16606871, -0.25641059], [1.0304995, -0.16955962],
                     [0.48921682, -1.38504507], [-0.03918551, -0.68540745],
                     [0.24991051, -1.00864997], [0.80541964, -0.34465185],
                     [0.1732627, -1.61323172]])
    y_gt = np.array([
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2
    ])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_alknn_not_good_object():
    """Test either if an error is raised while a wrong type of NN is given"""

    nn = 'rnd'
    allknn = AllKNN(n_neighbors=nn, random_state=RND_SEED, kind_sel='mode')
    assert_raises(ValueError, allknn.fit_sample, X, Y)
