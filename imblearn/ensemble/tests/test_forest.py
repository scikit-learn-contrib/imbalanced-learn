import pytest

import numpy as np

from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal

from imblearn.ensemble import BalancedRandomForestClassifier


@pytest.fixture
def imbalanced_dataset():
    return make_classification(
        n_samples=10000,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_repeated=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.01, 0.05, 0.94],
        class_sep=0.8,
        random_state=0,
    )


@pytest.mark.parametrize(
    "forest_params, err_msg",
    [
        ({"n_estimators": "whatever"}, "n_estimators must be an integer"),
        ({"n_estimators": -100}, "n_estimators must be greater than zero"),
        (
            {"bootstrap": False, "oob_score": True},
            "Out of bag estimation only",
        ),
    ],
)
def test_balanced_random_forest_error(imbalanced_dataset, forest_params, err_msg):
    brf = BalancedRandomForestClassifier(**forest_params)
    with pytest.raises(ValueError, match=err_msg):
        brf.fit(*imbalanced_dataset)


def test_balanced_random_forest_error_warning_warm_start(imbalanced_dataset):
    brf = BalancedRandomForestClassifier(n_estimators=5)
    brf.fit(*imbalanced_dataset)

    with pytest.raises(ValueError, match="must be larger or equal to"):
        brf.set_params(warm_start=True, n_estimators=2)
        brf.fit(*imbalanced_dataset)

    brf.set_params(n_estimators=10)
    brf.fit(*imbalanced_dataset)

    with pytest.warns(UserWarning, match="Warm-start fitting without"):
        brf.fit(*imbalanced_dataset)


def test_balanced_random_forest(imbalanced_dataset):
    n_estimators = 10
    brf = BalancedRandomForestClassifier(n_estimators=n_estimators, random_state=0)
    brf.fit(*imbalanced_dataset)

    assert len(brf.samplers_) == n_estimators
    assert len(brf.estimators_) == n_estimators
    assert len(brf.pipelines_) == n_estimators
    assert len(brf.feature_importances_) == imbalanced_dataset[0].shape[1]


def test_balanced_random_forest_attributes(imbalanced_dataset):
    X, y = imbalanced_dataset
    n_estimators = 10
    brf = BalancedRandomForestClassifier(n_estimators=n_estimators, random_state=0)
    brf.fit(X, y)

    for idx in range(n_estimators):
        X_res, y_res = brf.samplers_[idx].fit_resample(X, y)
        X_res_2, y_res_2 = (
            brf.pipelines_[idx].named_steps["randomundersampler"].fit_resample(X, y)
        )
        assert_allclose(X_res, X_res_2)
        assert_array_equal(y_res, y_res_2)

        y_pred = brf.estimators_[idx].fit(X_res, y_res).predict(X)
        y_pred_2 = brf.pipelines_[idx].fit(X, y).predict(X)
        assert_array_equal(y_pred, y_pred_2)

        y_pred = brf.estimators_[idx].fit(X_res, y_res).predict_proba(X)
        y_pred_2 = brf.pipelines_[idx].fit(X, y).predict_proba(X)
        assert_array_equal(y_pred, y_pred_2)


def test_balanced_random_forest_sample_weight(imbalanced_dataset):
    rng = np.random.RandomState(42)
    X, y = imbalanced_dataset
    sample_weight = rng.rand(y.shape[0])
    brf = BalancedRandomForestClassifier(n_estimators=5, random_state=0)
    brf.fit(X, y, sample_weight)


@pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores")
def test_balanced_random_forest_oob(imbalanced_dataset):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, stratify=y
    )
    est = BalancedRandomForestClassifier(
        oob_score=True,
        random_state=0,
        n_estimators=1000,
        min_samples_leaf=2,
    )

    est.fit(X_train, y_train)
    test_score = est.score(X_test, y_test)

    assert abs(test_score - est.oob_score_) < 0.1

    # Check warning if not enough estimators
    est = BalancedRandomForestClassifier(
        oob_score=True, random_state=0, n_estimators=1, bootstrap=True
    )
    with pytest.warns(UserWarning) and np.errstate(divide="ignore", invalid="ignore"):
        est.fit(X, y)


def test_balanced_random_forest_grid_search(imbalanced_dataset):
    brf = BalancedRandomForestClassifier()
    grid = GridSearchCV(brf, {"n_estimators": (1, 2), "max_depth": (1, 2)}, cv=3)
    grid.fit(*imbalanced_dataset)


def test_little_tree_with_small_max_samples():
    rng = np.random.RandomState(1)

    X = rng.randn(10000, 2)
    y = rng.randn(10000) > 0

    # First fit with no restriction on max samples
    est1 = BalancedRandomForestClassifier(
        n_estimators=1,
        random_state=rng,
        max_samples=None,
    )

    # Second fit with max samples restricted to just 2
    est2 = BalancedRandomForestClassifier(
        n_estimators=1,
        random_state=rng,
        max_samples=2,
    )

    est1.fit(X, y)
    est2.fit(X, y)

    tree1 = est1.estimators_[0].tree_
    tree2 = est2.estimators_[0].tree_

    msg = "Tree without `max_samples` restriction should have more nodes"
    assert tree1.node_count > tree2.node_count, msg


def test_balanced_random_forest_pruning(imbalanced_dataset):
    brf = BalancedRandomForestClassifier()
    brf.fit(*imbalanced_dataset)
    n_nodes_no_pruning = brf.estimators_[0].tree_.node_count

    brf_pruned = BalancedRandomForestClassifier(ccp_alpha=0.015)
    brf_pruned.fit(*imbalanced_dataset)
    n_nodes_pruning = brf_pruned.estimators_[0].tree_.node_count

    assert n_nodes_no_pruning > n_nodes_pruning


@pytest.mark.parametrize("ratio", [0.5, 0.1])
@pytest.mark.filterwarnings("ignore:Some inputs do not have OOB scores")
def test_balanced_random_forest_oob_binomial(ratio):
    # Regression test for #655: check that the oob score is closed to 0.5
    # a binomial experiment.
    rng = np.random.RandomState(42)
    n_samples = 1000
    X = np.arange(n_samples).reshape(-1, 1)
    y = rng.binomial(1, ratio, size=n_samples)

    erf = BalancedRandomForestClassifier(oob_score=True, random_state=42)
    erf.fit(X, y)
    assert np.abs(erf.oob_score_ - 0.5) < 0.1
