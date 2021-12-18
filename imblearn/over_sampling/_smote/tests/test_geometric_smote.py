"""
Test the geometric_smote module.
"""

from collections import Counter

import pytest
import numpy as np
from numpy.linalg import norm
from sklearn.utils import check_random_state
from sklearn.datasets import make_classification

from ..geometric import _make_geometric_sample, GeometricSMOTE, SELECTION_STRATEGY

RND_SEED = 0
RANDOM_STATE = check_random_state(RND_SEED)
CENTERS = [
    RANDOM_STATE.random_sample((2,)),
    2.6 * RANDOM_STATE.random_sample((4,)),
    3.2 * RANDOM_STATE.random_sample((10,)),
    -0.5 * RANDOM_STATE.random_sample((1,)),
]
SURFACE_POINTS = [
    RANDOM_STATE.random_sample((2,)),
    5.2 * RANDOM_STATE.random_sample((4,)),
    -3.5 * RANDOM_STATE.random_sample((10,)),
    -10.9 * RANDOM_STATE.random_sample((1,)),
]
TRUNCATION_FACTORS = [-1.0, -0.5, 0.0, 0.5, 1.0]
DEFORMATION_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0]


@pytest.mark.parametrize(
    "center,surface_point",
    [
        (CENTERS[0], SURFACE_POINTS[0]),
        (CENTERS[1], SURFACE_POINTS[1]),
        (CENTERS[2], SURFACE_POINTS[2]),
        (CENTERS[3], SURFACE_POINTS[3]),
    ],
)
def test_make_geometric_sample_hypersphere(center, surface_point):
    """Test the generation of points inside a hypersphere."""
    point = _make_geometric_sample(center, surface_point, 0.0, 0.0, RANDOM_STATE)
    rel_point = point - center
    rel_surface_point = surface_point - center
    np.testing.assert_array_less(0.0, norm(rel_surface_point) - norm(rel_point))


@pytest.mark.parametrize(
    "surface_point,deformation_factor",
    [
        (np.array([1.0, 0.0]), 0.0),
        (2.6 * np.array([0.0, 1.0]), 0.25),
        (3.2 * np.array([0.0, 1.0, 0.0, 0.0]), 0.50),
        (0.5 * np.array([0.0, 0.0, 1.0]), 0.75),
        (6.7 * np.array([0.0, 0.0, 1.0, 0.0, 0.0]), 1.0),
    ],
)
def test_make_geometric_sample_half_hypersphere(surface_point, deformation_factor):
    """Test the generation of points inside a hypersphere."""
    center = np.zeros(surface_point.shape)
    point = _make_geometric_sample(
        center, surface_point, 1.0, deformation_factor, RANDOM_STATE
    )
    np.testing.assert_array_less(0.0, norm(surface_point) - norm(point))
    np.testing.assert_array_less(0.0, np.dot(point, surface_point))


@pytest.mark.parametrize(
    "center,surface_point,truncation_factor",
    [
        (center, surface_point, truncation_factor)
        for center, surface_point in zip(CENTERS, SURFACE_POINTS)
        for truncation_factor in TRUNCATION_FACTORS
    ],
)
def test_make_geometric_sample_line_segment(center, surface_point, truncation_factor):
    """Test the generation of points on a line segment."""
    point = _make_geometric_sample(
        center, surface_point, truncation_factor, 1.0, RANDOM_STATE
    )
    rel_point = point - center
    rel_surface_point = surface_point - center
    dot_product = np.dot(rel_point, rel_surface_point)
    norms_product = norm(rel_point) * norm(rel_surface_point)
    np.testing.assert_array_less(0.0, norm(rel_surface_point) - norm(rel_point))
    dot_product = (
        np.abs(dot_product) if truncation_factor == 0.0 else (-1) * dot_product
    )
    np.testing.assert_allclose(np.abs(dot_product) / norms_product, 1.0)


def test_gsmote_default_init():
    """Test the intialization with default parameters."""
    gsmote = GeometricSMOTE()
    assert gsmote.sampling_strategy == "auto"
    assert gsmote.random_state is None
    assert gsmote.truncation_factor == 1.0
    assert gsmote.deformation_factor == 0.0
    assert gsmote.selection_strategy == "combined"
    assert gsmote.k_neighbors == 5
    assert gsmote.n_jobs == 1


def test_gsmote_fit():
    """Test fit method."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(
        random_state=RND_SEED, n_samples=n_samples, weights=weights
    )
    gsmote = GeometricSMOTE(random_state=RANDOM_STATE).fit(X, y)
    assert gsmote.sampling_strategy_ == {1: 40}


def test_gsmote_invalid_selection_strategy():
    """Test invalid selection strategy."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(
        random_state=RND_SEED, n_samples=n_samples, weights=weights
    )
    gsmote = GeometricSMOTE(random_state=RANDOM_STATE, selection_strategy="Minority")
    with pytest.raises(ValueError):
        gsmote.fit_resample(X, y)


@pytest.mark.parametrize("selection_strategy", ["combined", "minority", "majority"])
def test_gsmote_nn(selection_strategy):
    """Test nearest neighbors object."""
    n_samples, weights = 200, [0.6, 0.4]
    X, y = make_classification(
        random_state=RND_SEED, n_samples=n_samples, weights=weights
    )
    gsmote = GeometricSMOTE(
        random_state=RANDOM_STATE, selection_strategy=selection_strategy
    )
    _ = gsmote.fit_resample(X, y)
    if selection_strategy in ("minority", "combined"):
        assert gsmote.nns_pos_.n_neighbors == gsmote.k_neighbors + 1
    if selection_strategy in ("majority", "combined"):
        assert gsmote.nn_neg_.n_neighbors == 1


@pytest.mark.parametrize(
    "selection_strategy, truncation_factor, deformation_factor",
    [
        (selection_strategy, truncation_factor, deformation_factor)
        for selection_strategy in SELECTION_STRATEGY
        for truncation_factor in TRUNCATION_FACTORS
        for deformation_factor in DEFORMATION_FACTORS
    ],
)
def test_gsmote_fit_resample_binary(
    selection_strategy, truncation_factor, deformation_factor
):
    """Test fit and sample for binary class case."""
    n_maj, n_min, step, min_coor, max_coor = 12, 5, 0.5, 0.0, 8.5
    X = np.repeat(np.arange(min_coor, max_coor, step), 2).reshape(-1, 2)
    y = np.concatenate([np.repeat(0, n_maj), np.repeat(1, n_min)])
    radius = np.sqrt(0.5) * step
    k_neighbors = 1
    gsmote = GeometricSMOTE(
        "auto",
        RANDOM_STATE,
        truncation_factor,
        deformation_factor,
        selection_strategy,
        k_neighbors,
    )
    X_resampled, y_resampled = gsmote.fit_resample(X, y)
    assert gsmote.sampling_strategy_ == {1: (n_maj - n_min)}
    assert y_resampled.sum() == n_maj
    np.testing.assert_array_less(X[n_maj - 1] - radius, X_resampled[n_maj + n_min])


@pytest.mark.parametrize(
    "selection_strategy, truncation_factor, deformation_factor",
    [
        (selection_strategy, truncation_factor, deformation_factor)
        for selection_strategy in SELECTION_STRATEGY
        for truncation_factor in TRUNCATION_FACTORS
        for deformation_factor in DEFORMATION_FACTORS
    ],
)
def test_gsmote_fit_resample_multiclass(
    selection_strategy, truncation_factor, deformation_factor
):
    """Test fit and sample for multiclass case."""
    n_samples, weights = 100, [0.75, 0.15, 0.10]
    X, y = make_classification(
        random_state=RND_SEED,
        n_samples=n_samples,
        weights=weights,
        n_classes=3,
        n_informative=5,
    )
    k_neighbors, majority_label = 1, 0
    gsmote = GeometricSMOTE(
        "auto",
        RANDOM_STATE,
        truncation_factor,
        deformation_factor,
        selection_strategy,
        k_neighbors,
    )
    _, y_resampled = gsmote.fit_resample(X, y)
    assert majority_label not in gsmote.sampling_strategy_.keys()
    np.testing.assert_array_equal(np.unique(y), np.unique(y_resampled))
    assert len(set(Counter(y_resampled).values())) == 1
