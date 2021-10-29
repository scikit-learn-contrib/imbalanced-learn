"""Test the module nearmiss."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
import numpy as np

from sklearn.utils._testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors

from imblearn.under_sampling import NearMiss

X = np.array(
    [
        [1.17737838, -0.2002118],
        [0.4960075, 0.86130762],
        [-0.05903827, 0.10947647],
        [0.91464286, 1.61369212],
        [-0.54619583, 1.73009918],
        [-0.60413357, 0.24628718],
        [0.45713638, 1.31069295],
        [-0.04032409, 3.01186964],
        [0.03142011, 0.12323596],
        [0.50701028, -0.17636928],
        [-0.80809175, -1.09917302],
        [-0.20497017, -0.26630228],
        [0.99272351, -0.11631728],
        [-1.95581933, 0.69609604],
        [1.15157493, -1.2981518],
    ]
)
Y = np.array([1, 2, 1, 0, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2])

VERSION_NEARMISS = (1, 2, 3)


@pytest.mark.parametrize(
    "nearmiss_params, err_msg",
    [
        ({"version": 1000}, "must be 1, 2 or 3"),
        ({"version": 1, "n_neighbors": "rnd"}, "NearestNeighbors object or int"),
        (
            {
                "version": 3,
                "n_neighbors": NearestNeighbors(n_neighbors=3),
                "n_neighbors_ver3": "rnd",
            },
            "NearestNeighbors object or int",
        ),
    ],
)
def test_nearmiss_error(nearmiss_params, err_msg):
    nm = NearMiss(**nearmiss_params)
    with pytest.raises(ValueError, match=err_msg):
        nm.fit_resample(X, Y)


def test_nm_fit_resample_auto():
    sampling_strategy = "auto"
    X_gt = [
        np.array(
            [
                [0.91464286, 1.61369212],
                [-0.80809175, -1.09917302],
                [-0.20497017, -0.26630228],
                [-0.05903827, 0.10947647],
                [0.03142011, 0.12323596],
                [-0.60413357, 0.24628718],
                [0.50701028, -0.17636928],
                [0.4960075, 0.86130762],
                [0.45713638, 1.31069295],
            ]
        ),
        np.array(
            [
                [0.91464286, 1.61369212],
                [-0.80809175, -1.09917302],
                [-0.20497017, -0.26630228],
                [-0.05903827, 0.10947647],
                [0.03142011, 0.12323596],
                [-0.60413357, 0.24628718],
                [0.50701028, -0.17636928],
                [0.4960075, 0.86130762],
                [0.45713638, 1.31069295],
            ]
        ),
        np.array(
            [
                [0.91464286, 1.61369212],
                [-0.80809175, -1.09917302],
                [-0.20497017, -0.26630228],
                [1.17737838, -0.2002118],
                [-0.60413357, 0.24628718],
                [0.03142011, 0.12323596],
                [1.15157493, -1.2981518],
                [-0.54619583, 1.73009918],
                [0.99272351, -0.11631728],
            ]
        ),
    ]
    y_gt = [
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
    ]
    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(sampling_strategy=sampling_strategy, version=version)
        X_resampled, y_resampled = nm.fit_resample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])


def test_nm_fit_resample_float_sampling_strategy():
    sampling_strategy = {0: 3, 1: 4, 2: 4}
    X_gt = [
        np.array(
            [
                [-0.20497017, -0.26630228],
                [-0.80809175, -1.09917302],
                [0.91464286, 1.61369212],
                [-0.05903827, 0.10947647],
                [0.03142011, 0.12323596],
                [-0.60413357, 0.24628718],
                [1.17737838, -0.2002118],
                [0.50701028, -0.17636928],
                [0.4960075, 0.86130762],
                [0.45713638, 1.31069295],
                [0.99272351, -0.11631728],
            ]
        ),
        np.array(
            [
                [-0.20497017, -0.26630228],
                [-0.80809175, -1.09917302],
                [0.91464286, 1.61369212],
                [-0.05903827, 0.10947647],
                [0.03142011, 0.12323596],
                [-0.60413357, 0.24628718],
                [1.17737838, -0.2002118],
                [0.50701028, -0.17636928],
                [0.4960075, 0.86130762],
                [0.45713638, 1.31069295],
                [0.99272351, -0.11631728],
            ]
        ),
        np.array(
            [
                [0.91464286, 1.61369212],
                [-0.80809175, -1.09917302],
                [-0.20497017, -0.26630228],
                [1.17737838, -0.2002118],
                [-0.60413357, 0.24628718],
                [0.03142011, 0.12323596],
                [-0.05903827, 0.10947647],
                [1.15157493, -1.2981518],
                [-0.54619583, 1.73009918],
                [0.99272351, -0.11631728],
                [0.45713638, 1.31069295],
            ]
        ),
    ]
    y_gt = [
        np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
        np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
        np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    ]

    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(sampling_strategy=sampling_strategy, version=version)
        X_resampled, y_resampled = nm.fit_resample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])


def test_nm_fit_resample_nn_obj():
    sampling_strategy = "auto"
    nn = NearestNeighbors(n_neighbors=3)
    X_gt = [
        np.array(
            [
                [0.91464286, 1.61369212],
                [-0.80809175, -1.09917302],
                [-0.20497017, -0.26630228],
                [-0.05903827, 0.10947647],
                [0.03142011, 0.12323596],
                [-0.60413357, 0.24628718],
                [0.50701028, -0.17636928],
                [0.4960075, 0.86130762],
                [0.45713638, 1.31069295],
            ]
        ),
        np.array(
            [
                [0.91464286, 1.61369212],
                [-0.80809175, -1.09917302],
                [-0.20497017, -0.26630228],
                [-0.05903827, 0.10947647],
                [0.03142011, 0.12323596],
                [-0.60413357, 0.24628718],
                [0.50701028, -0.17636928],
                [0.4960075, 0.86130762],
                [0.45713638, 1.31069295],
            ]
        ),
        np.array(
            [
                [0.91464286, 1.61369212],
                [-0.80809175, -1.09917302],
                [-0.20497017, -0.26630228],
                [1.17737838, -0.2002118],
                [-0.60413357, 0.24628718],
                [0.03142011, 0.12323596],
                [1.15157493, -1.2981518],
                [-0.54619583, 1.73009918],
                [0.99272351, -0.11631728],
            ]
        ),
    ]
    y_gt = [
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
        np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
    ]
    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(
            sampling_strategy=sampling_strategy,
            version=version,
            n_neighbors=nn,
        )
        X_resampled, y_resampled = nm.fit_resample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])
