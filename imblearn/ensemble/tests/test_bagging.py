"""Test the module ensemble classifiers."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
import pytest

from sklearn.datasets import load_iris, make_hastie_10_2
from sklearn.model_selection import (
    GridSearchCV,
    ParameterGrid,
    train_test_split,
)
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_allclose

from imblearn.datasets import make_imbalance
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import RandomUnderSampler

iris = load_iris()


def test_balanced_bagging_classifier():
    # Check classification for various parameter settings.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    grid = ParameterGrid(
        {
            "max_samples": [0.5, 1.0],
            "max_features": [1, 2, 4],
            "bootstrap": [True, False],
            "bootstrap_features": [True, False],
        }
    )

    for base_estimator in [
        None,
        DummyClassifier(strategy="prior"),
        Perceptron(max_iter=1000, tol=1e-3),
        DecisionTreeClassifier(),
        KNeighborsClassifier(),
        SVC(gamma="scale"),
    ]:
        for params in grid:
            BalancedBaggingClassifier(
                base_estimator=base_estimator, random_state=0, **params
            ).fit(X_train, y_train).predict(X_test)


def test_bootstrap_samples():
    # Test that bootstrapping samples generate non-perfect base estimators.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    base_estimator = DecisionTreeClassifier().fit(X_train, y_train)

    # without bootstrap, all trees are perfect on the training set
    # disable the resampling by passing an empty dictionary.
    ensemble = BalancedBaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        max_samples=1.0,
        bootstrap=False,
        n_estimators=10,
        sampling_strategy={},
        random_state=0,
    ).fit(X_train, y_train)

    assert ensemble.score(X_train, y_train) == base_estimator.score(
        X_train, y_train
    )

    # with bootstrap, trees are no longer perfect on the training set
    ensemble = BalancedBaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        max_samples=1.0,
        bootstrap=True,
        random_state=0,
    ).fit(X_train, y_train)

    assert ensemble.score(X_train, y_train) < base_estimator.score(
        X_train, y_train
    )


def test_bootstrap_features():
    # Test that bootstrapping features may generate duplicate features.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ensemble = BalancedBaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        max_features=1.0,
        bootstrap_features=False,
        random_state=0,
    ).fit(X_train, y_train)

    for features in ensemble.estimators_features_:
        assert np.unique(features).shape[0] == X.shape[1]

    ensemble = BalancedBaggingClassifier(
        base_estimator=DecisionTreeClassifier(),
        max_features=1.0,
        bootstrap_features=True,
        random_state=0,
    ).fit(X_train, y_train)

    unique_features = [
        np.unique(features).shape[0]
        for features in ensemble.estimators_features_
    ]
    assert np.median(unique_features) < X.shape[1]


def test_probability():
    # Predict probabilities.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Normal case
        ensemble = BalancedBaggingClassifier(
            base_estimator=DecisionTreeClassifier(), random_state=0
        ).fit(X_train, y_train)

        assert_array_almost_equal(
            np.sum(ensemble.predict_proba(X_test), axis=1),
            np.ones(len(X_test)),
        )

        assert_array_almost_equal(
            ensemble.predict_proba(X_test),
            np.exp(ensemble.predict_log_proba(X_test)),
        )

        # Degenerate case, where some classes are missing
        ensemble = BalancedBaggingClassifier(
            base_estimator=LogisticRegression(
                solver="lbfgs", multi_class="auto"
            ),
            random_state=0,
            max_samples=5,
        )
        ensemble.fit(X_train, y_train)

        assert_array_almost_equal(
            np.sum(ensemble.predict_proba(X_test), axis=1),
            np.ones(len(X_test)),
        )

        assert_array_almost_equal(
            ensemble.predict_proba(X_test),
            np.exp(ensemble.predict_log_proba(X_test)),
        )


def test_oob_score_classification():
    # Check that oob prediction is a good estimation of the generalization
    # error.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    for base_estimator in [DecisionTreeClassifier(), SVC(gamma="scale")]:
        clf = BalancedBaggingClassifier(
            base_estimator=base_estimator,
            n_estimators=100,
            bootstrap=True,
            oob_score=True,
            random_state=0,
        ).fit(X_train, y_train)

        test_score = clf.score(X_test, y_test)

        assert abs(test_score - clf.oob_score_) < 0.1

        # Test with few estimators
        with pytest.warns(UserWarning):
            BalancedBaggingClassifier(
                base_estimator=base_estimator,
                n_estimators=1,
                bootstrap=True,
                oob_score=True,
                random_state=0,
            ).fit(X_train, y_train)


def test_single_estimator():
    # Check singleton ensembles.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf1 = BalancedBaggingClassifier(
        base_estimator=KNeighborsClassifier(),
        n_estimators=1,
        bootstrap=False,
        bootstrap_features=False,
        random_state=0,
    ).fit(X_train, y_train)

    clf2 = make_pipeline(
        RandomUnderSampler(
            random_state=clf1.estimators_[0].steps[0][1].random_state
        ),
        KNeighborsClassifier(),
    ).fit(X_train, y_train)

    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))


def test_error():
    # Test that it gives proper exception on deficient input.
    X, y = make_imbalance(
        iris.data, iris.target, sampling_strategy={0: 20, 1: 25, 2: 50}
    )
    base = DecisionTreeClassifier()

    # Test n_estimators
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, n_estimators=1.5).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, n_estimators=-1).fit(X, y)

    # Test max_samples
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_samples=-1).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_samples=0.0).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_samples=2.0).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_samples=1000).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_samples="foobar").fit(X, y)

    # Test max_features
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_features=-1).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_features=0.0).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_features=2.0).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_features=5).fit(X, y)
    with pytest.raises(ValueError):
        BalancedBaggingClassifier(base, max_features="foobar").fit(X, y)

    # Test support of decision_function
    assert not (
        hasattr(BalancedBaggingClassifier(base).fit(X, y), "decision_function")
    )


def test_gridsearch():
    # Check that bagging ensembles can be grid-searched.
    # Transform iris into a binary classification task
    X, y = iris.data, iris.target.copy()
    y[y == 2] = 1

    # Grid search with scoring based on decision_function
    parameters = {"n_estimators": (1, 2), "base_estimator__C": (1, 2)}

    GridSearchCV(
        BalancedBaggingClassifier(SVC(gamma="scale")),
        parameters,
        cv=3,
        scoring="roc_auc",
    ).fit(X, y)


def test_base_estimator():
    # Check base_estimator and its default values.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ensemble = BalancedBaggingClassifier(None, n_jobs=3, random_state=0).fit(
        X_train, y_train
    )

    assert isinstance(
        ensemble.base_estimator_.steps[-1][1], DecisionTreeClassifier
    )

    ensemble = BalancedBaggingClassifier(
        DecisionTreeClassifier(), n_jobs=3, random_state=0
    ).fit(X_train, y_train)

    assert isinstance(
        ensemble.base_estimator_.steps[-1][1], DecisionTreeClassifier
    )

    ensemble = BalancedBaggingClassifier(
        Perceptron(max_iter=1000, tol=1e-3), n_jobs=3, random_state=0
    ).fit(X_train, y_train)

    assert isinstance(ensemble.base_estimator_.steps[-1][1], Perceptron)


def test_bagging_with_pipeline():
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    estimator = BalancedBaggingClassifier(
        make_pipeline(SelectKBest(k=1), DecisionTreeClassifier()),
        max_features=2,
    )
    estimator.fit(X, y).predict(X)


def test_warm_start(random_state=42):
    # Test if fitting incrementally with warm start gives a forest of the
    # right size and the same results as a normal fit.
    X, y = make_hastie_10_2(n_samples=20, random_state=1)

    clf_ws = None
    for n_estimators in [5, 10]:
        if clf_ws is None:
            clf_ws = BalancedBaggingClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                warm_start=True,
            )
        else:
            clf_ws.set_params(n_estimators=n_estimators)
        clf_ws.fit(X, y)
        assert len(clf_ws) == n_estimators

    clf_no_ws = BalancedBaggingClassifier(
        n_estimators=10, random_state=random_state, warm_start=False
    )
    clf_no_ws.fit(X, y)

    assert {pipe.steps[-1][1].random_state for pipe in clf_ws} == {
        pipe.steps[-1][1].random_state for pipe in clf_no_ws
    }


def test_warm_start_smaller_n_estimators():
    # Test if warm start'ed second fit with smaller n_estimators raises error.
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    clf = BalancedBaggingClassifier(n_estimators=5, warm_start=True)
    clf.fit(X, y)
    clf.set_params(n_estimators=4)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_warm_start_equal_n_estimators():
    # Test that nothing happens when fitting without increasing n_estimators
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf = BalancedBaggingClassifier(
        n_estimators=5, warm_start=True, random_state=83
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # modify X to nonsense values, this should not change anything
    X_train += 1.0

    warn_msg = "Warm-start fitting without increasing n_estimators does not"
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X_train, y_train)
    assert_array_equal(y_pred, clf.predict(X_test))


def test_warm_start_equivalence():
    # warm started classifier with 5+5 estimators should be equivalent to
    # one classifier with 10 estimators
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf_ws = BalancedBaggingClassifier(
        n_estimators=5, warm_start=True, random_state=3141
    )
    clf_ws.fit(X_train, y_train)
    clf_ws.set_params(n_estimators=10)
    clf_ws.fit(X_train, y_train)
    y1 = clf_ws.predict(X_test)

    clf = BalancedBaggingClassifier(
        n_estimators=10, warm_start=False, random_state=3141
    )
    clf.fit(X_train, y_train)
    y2 = clf.predict(X_test)

    assert_array_almost_equal(y1, y2)


def test_warm_start_with_oob_score_fails():
    # Check using oob_score and warm_start simultaneously fails
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    clf = BalancedBaggingClassifier(
        n_estimators=5, warm_start=True, oob_score=True
    )
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_oob_score_removed_on_warm_start():
    X, y = make_hastie_10_2(n_samples=2000, random_state=1)

    clf = BalancedBaggingClassifier(n_estimators=50, oob_score=True)
    clf.fit(X, y)

    clf.set_params(warm_start=True, oob_score=False, n_estimators=100)
    clf.fit(X, y)

    with pytest.raises(AttributeError):
        getattr(clf, "oob_score_")


def test_oob_score_consistency():
    # Make sure OOB scores are identical when random_state, estimator, and
    # training data are fixed and fitting is done twice
    X, y = make_hastie_10_2(n_samples=200, random_state=1)
    bagging = BalancedBaggingClassifier(
        KNeighborsClassifier(),
        max_samples=0.5,
        max_features=0.5,
        oob_score=True,
        random_state=1,
    )
    assert bagging.fit(X, y).oob_score_ == bagging.fit(X, y).oob_score_


def test_estimators_samples():
    # Check that format of estimators_samples_ is correct and that results
    # generated at fit time can be identically reproduced at a later time
    # using data saved in object attributes.
    X, y = make_hastie_10_2(n_samples=200, random_state=1)

    # remap the y outside of the BalancedBaggingclassifier
    # _, y = np.unique(y, return_inverse=True)
    bagging = BalancedBaggingClassifier(
        LogisticRegression(solver="lbfgs", multi_class="auto"),
        max_samples=0.5,
        max_features=0.5,
        random_state=1,
        bootstrap=False,
    )
    bagging.fit(X, y)

    # Get relevant attributes
    estimators_samples = bagging.estimators_samples_
    estimators_features = bagging.estimators_features_
    estimators = bagging.estimators_

    # Test for correct formatting
    assert len(estimators_samples) == len(estimators)
    assert len(estimators_samples[0]) == len(X) // 2
    assert estimators_samples[0].dtype.kind == "i"

    # Re-fit single estimator to test for consistent sampling
    estimator_index = 0
    estimator_samples = estimators_samples[estimator_index]
    estimator_features = estimators_features[estimator_index]
    estimator = estimators[estimator_index]

    X_train = (X[estimator_samples])[:, estimator_features]
    y_train = y[estimator_samples]

    orig_coefs = estimator.steps[-1][1].coef_
    estimator.fit(X_train, y_train)
    new_coefs = estimator.steps[-1][1].coef_

    assert_allclose(orig_coefs, new_coefs)


def test_max_samples_consistency():
    # Make sure validated max_samples and original max_samples are identical
    # when valid integer max_samples supplied by user
    max_samples = 100
    X, y = make_hastie_10_2(n_samples=2 * max_samples, random_state=1)
    bagging = BalancedBaggingClassifier(
        KNeighborsClassifier(),
        max_samples=max_samples,
        max_features=0.5,
        random_state=1,
    )
    bagging.fit(X, y)
    assert bagging._max_samples == max_samples
