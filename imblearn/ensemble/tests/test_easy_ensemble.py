"""Test the module easy ensemble."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
import numpy as np

from sklearn.datasets import load_iris, make_hastie_10_2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal

from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

iris = load_iris()

# Generate a global dataset to use
RND_SEED = 0
X = np.array(
    [
        [0.5220963, 0.11349303],
        [0.59091459, 0.40692742],
        [1.10915364, 0.05718352],
        [0.22039505, 0.26469445],
        [1.35269503, 0.44812421],
        [0.85117925, 1.0185556],
        [-2.10724436, 0.70263997],
        [-0.23627356, 0.30254174],
        [-1.23195149, 0.15427291],
        [-0.58539673, 0.62515052],
    ]
)
Y = np.array([1, 2, 2, 2, 1, 0, 1, 1, 1, 0])


@pytest.mark.parametrize("n_estimators", [10, 20])
@pytest.mark.parametrize(
    "base_estimator",
    [AdaBoostClassifier(n_estimators=5), AdaBoostClassifier(n_estimators=10)],
)
def test_easy_ensemble_classifier(n_estimators, base_estimator):
    # Check classification for various parameter settings.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    eec = EasyEnsembleClassifier(
        n_estimators=n_estimators,
        base_estimator=base_estimator,
        n_jobs=-1,
        random_state=RND_SEED,
    )
    eec.fit(X_train, y_train).score(X_test, y_test)
    assert len(eec.estimators_) == n_estimators
    for est in eec.estimators_:
        assert len(est.named_steps["classifier"]) == base_estimator.n_estimators
    # test the different prediction function
    eec.predict(X_test)
    eec.predict_proba(X_test)
    eec.predict_log_proba(X_test)
    eec.decision_function(X_test)


def test_base_estimator():
    # Check base_estimator and its default values.
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    ensemble = EasyEnsembleClassifier(2, None, n_jobs=-1, random_state=0).fit(
        X_train, y_train
    )

    assert isinstance(ensemble.base_estimator_.steps[-1][1], AdaBoostClassifier)

    ensemble = EasyEnsembleClassifier(
        2, AdaBoostClassifier(), n_jobs=-1, random_state=0
    ).fit(X_train, y_train)

    assert isinstance(ensemble.base_estimator_.steps[-1][1], AdaBoostClassifier)


def test_bagging_with_pipeline():
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    estimator = EasyEnsembleClassifier(
        n_estimators=2,
        base_estimator=make_pipeline(SelectKBest(k=1), AdaBoostClassifier()),
    )
    estimator.fit(X, y).predict(X)


def test_warm_start(random_state=42):
    # Test if fitting incrementally with warm start gives a forest of the
    # right size and the same results as a normal fit.
    X, y = make_hastie_10_2(n_samples=20, random_state=1)

    clf_ws = None
    for n_estimators in [5, 10]:
        if clf_ws is None:
            clf_ws = EasyEnsembleClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                warm_start=True,
            )
        else:
            clf_ws.set_params(n_estimators=n_estimators)
        clf_ws.fit(X, y)
        assert len(clf_ws) == n_estimators

    clf_no_ws = EasyEnsembleClassifier(
        n_estimators=10, random_state=random_state, warm_start=False
    )
    clf_no_ws.fit(X, y)

    assert {pipe.steps[-1][1].random_state for pipe in clf_ws} == {
        pipe.steps[-1][1].random_state for pipe in clf_no_ws
    }


def test_warm_start_smaller_n_estimators():
    # Test if warm start'ed second fit with smaller n_estimators raises error.
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    clf = EasyEnsembleClassifier(n_estimators=5, warm_start=True)
    clf.fit(X, y)
    clf.set_params(n_estimators=4)
    with pytest.raises(ValueError):
        clf.fit(X, y)


def test_warm_start_equal_n_estimators():
    # Test that nothing happens when fitting without increasing n_estimators
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf = EasyEnsembleClassifier(n_estimators=5, warm_start=True, random_state=83)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # modify X to nonsense values, this should not change anything
    X_train += 1.0

    warn_msg = "Warm-start fitting without increasing n_estimators"
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X_train, y_train)
    assert_array_equal(y_pred, clf.predict(X_test))


def test_warm_start_equivalence():
    # warm started classifier with 5+5 estimators should be equivalent to
    # one classifier with 10 estimators
    X, y = make_hastie_10_2(n_samples=20, random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=43)

    clf_ws = EasyEnsembleClassifier(n_estimators=5, warm_start=True, random_state=3141)
    clf_ws.fit(X_train, y_train)
    clf_ws.set_params(n_estimators=10)
    clf_ws.fit(X_train, y_train)
    y1 = clf_ws.predict(X_test)

    clf = EasyEnsembleClassifier(n_estimators=10, warm_start=False, random_state=3141)
    clf.fit(X_train, y_train)
    y2 = clf.predict(X_test)

    assert_allclose(y1, y2)


@pytest.mark.parametrize(
    "n_estimators, msg_error",
    [
        (1.0, "n_estimators must be an integer"),
        (-10, "n_estimators must be greater than zero"),
    ],
)
def test_easy_ensemble_classifier_error(n_estimators, msg_error):
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    with pytest.raises(ValueError, match=msg_error):
        eec = EasyEnsembleClassifier(n_estimators=n_estimators)
        eec.fit(X, y)


def test_easy_ensemble_classifier_single_estimator():
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf1 = EasyEnsembleClassifier(n_estimators=1, random_state=0).fit(X_train, y_train)
    clf2 = make_pipeline(
        RandomUnderSampler(random_state=0), AdaBoostClassifier(random_state=0)
    ).fit(X_train, y_train)

    assert_array_equal(clf1.predict(X_test), clf2.predict(X_test))


def test_easy_ensemble_classifier_grid_search():
    X, y = make_imbalance(
        iris.data,
        iris.target,
        sampling_strategy={0: 20, 1: 25, 2: 50},
        random_state=0,
    )

    parameters = {
        "n_estimators": [1, 2],
        "base_estimator__n_estimators": [3, 4],
    }
    grid_search = GridSearchCV(
        EasyEnsembleClassifier(base_estimator=AdaBoostClassifier()),
        parameters,
        cv=5,
    )
    grid_search.fit(X, y)
