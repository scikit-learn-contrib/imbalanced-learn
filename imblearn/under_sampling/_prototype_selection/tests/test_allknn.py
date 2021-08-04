"""Test the module repeated edited nearest neighbour."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
import numpy as np

from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification

from imblearn.under_sampling import AllKNN

X = np.array(
    [
        [-0.12840393, 0.66446571],
        [1.32319756, -0.13181616],
        [0.04296502, -0.37981873],
        [0.83631853, 0.18569783],
        [1.02956816, 0.36061601],
        [1.12202806, 0.33811558],
        [-0.53171468, -0.53735182],
        [1.3381556, 0.35956356],
        [-0.35946678, 0.72510189],
        [1.32326943, 0.28393874],
        [2.94290565, -0.13986434],
        [0.28294738, -1.00125525],
        [0.34218094, -0.58781961],
        [-0.88864036, -0.33782387],
        [-1.10146139, 0.91782682],
        [-0.7969716, -0.50493969],
        [0.73489726, 0.43915195],
        [0.2096964, -0.61814058],
        [-0.28479268, 0.70459548],
        [1.84864913, 0.14729596],
        [1.59068979, -0.96622933],
        [0.73418199, -0.02222847],
        [0.50307437, 0.498805],
        [0.84929742, 0.41042894],
        [0.62649535, 0.46600596],
        [0.79270821, -0.41386668],
        [1.16606871, -0.25641059],
        [1.57356906, 0.30390519],
        [1.0304995, -0.16955962],
        [1.67314371, 0.19231498],
        [0.98382284, 0.37184502],
        [0.48921682, -1.38504507],
        [-0.46226554, -0.50481004],
        [-0.03918551, -0.68540745],
        [0.24991051, -1.00864997],
        [0.80541964, -0.34465185],
        [0.1732627, -1.61323172],
        [0.69804044, 0.44810796],
        [-0.5506368, -0.42072426],
        [-0.34474418, 0.21969797],
    ]
)
Y = np.array(
    [
        1,
        2,
        2,
        2,
        1,
        1,
        0,
        2,
        1,
        1,
        1,
        2,
        2,
        0,
        1,
        2,
        1,
        2,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        1,
        2,
        0,
        2,
        2,
        2,
        2,
        1,
        2,
        0,
    ]
)
R_TOL = 1e-4


def test_allknn_fit_resample():
    allknn = AllKNN()
    X_resampled, y_resampled = allknn.fit_resample(X, Y)

    X_gt = np.array(
        [
            [-0.53171468, -0.53735182],
            [-0.88864036, -0.33782387],
            [-0.46226554, -0.50481004],
            [-0.34474418, 0.21969797],
            [1.02956816, 0.36061601],
            [1.12202806, 0.33811558],
            [0.73489726, 0.43915195],
            [0.50307437, 0.498805],
            [0.84929742, 0.41042894],
            [0.62649535, 0.46600596],
            [0.98382284, 0.37184502],
            [0.69804044, 0.44810796],
        ]
    )
    y_gt = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_allclose(y_resampled, y_gt, rtol=R_TOL)


def test_all_knn_allow_minority():
    X, y = make_classification(
        n_samples=10000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.2, 0.3, 0.5],
        class_sep=0.4,
        random_state=0,
    )

    allknn = AllKNN(allow_minority=True)
    X_res_1, y_res_1 = allknn.fit_resample(X, y)
    allknn = AllKNN()
    X_res_2, y_res_2 = allknn.fit_resample(X, y)
    assert len(y_res_1) < len(y_res_2)


def test_allknn_fit_resample_mode():
    allknn = AllKNN(kind_sel="mode")
    X_resampled, y_resampled = allknn.fit_resample(X, Y)

    X_gt = np.array(
        [
            [-0.53171468, -0.53735182],
            [-0.88864036, -0.33782387],
            [-0.46226554, -0.50481004],
            [-0.34474418, 0.21969797],
            [-0.12840393, 0.66446571],
            [1.02956816, 0.36061601],
            [1.12202806, 0.33811558],
            [-0.35946678, 0.72510189],
            [1.32326943, 0.28393874],
            [2.94290565, -0.13986434],
            [0.73489726, 0.43915195],
            [-0.28479268, 0.70459548],
            [1.84864913, 0.14729596],
            [0.50307437, 0.498805],
            [0.84929742, 0.41042894],
            [0.62649535, 0.46600596],
            [1.67314371, 0.19231498],
            [0.98382284, 0.37184502],
            [0.69804044, 0.44810796]
        ]
    )
    y_gt = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_allknn_fit_resample_with_nn_object():
    nn = NearestNeighbors(n_neighbors=3)
    allknn = AllKNN(n_neighbors=nn, kind_sel="mode")
    X_resampled, y_resampled = allknn.fit_resample(X, Y)

    X_gt = np.array(
        [
            [-0.53171468, -0.53735182],
            [-0.88864036, -0.33782387],
            [-0.46226554, -0.50481004],
            [-0.34474418, 0.21969797],
            [-0.12840393, 0.66446571],
            [1.02956816, 0.36061601],
            [1.12202806, 0.33811558],
            [-0.35946678, 0.72510189],
            [1.32326943, 0.28393874],
            [2.94290565, -0.13986434],
            [0.73489726, 0.43915195],
            [-0.28479268, 0.70459548],
            [1.84864913, 0.14729596],
            [0.50307437, 0.498805],
            [0.84929742, 0.41042894],
            [0.62649535, 0.46600596],
            [1.67314371, 0.19231498],
            [0.98382284, 0.37184502],
            [0.69804044, 0.44810796]
        ]
    )
    y_gt = np.array(
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_alknn_not_good_object():
    nn = "rnd"
    allknn = AllKNN(n_neighbors=nn, kind_sel="mode")
    with pytest.raises(ValueError):
        allknn.fit_resample(X, Y)
