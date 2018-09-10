import pytest

import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.testing import assert_array_equal

from imblearn.ensemble import RUSBoostClassifier


@pytest.fixture
def imbalanced_dataset():
    return make_classification(n_samples=10000, n_features=2, n_informative=2,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94], class_sep=0.8,
                               random_state=0)


@pytest.mark.parametrize('algorithm', ['SAMME', 'SAMME.R'])
def test_rusboost(imbalanced_dataset, algorithm):
    X, y = imbalanced_dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
    classes = np.unique(y)

    n_estimators = 100
    rusboost = RUSBoostClassifier(n_estimators=n_estimators,
                                  algorithm=algorithm,
                                  random_state=0)
    rusboost.fit(X_train, y_train)
    assert_array_equal(classes, rusboost.classes_)

    # check that we have an ensemble of samplers and estimators with a
    # consistent size
    assert len(rusboost.estimators_) > 1
    assert len(rusboost.estimators_) == len(rusboost.samplers_)
    assert len(rusboost.pipelines_) == len(rusboost.samplers_)

    # each sampler in the ensemble should have different random state
    assert (len(set(sampler.random_state for sampler in rusboost.samplers_)) ==
            len(rusboost.samplers_))
    # each estimator in the ensemble should have different random state
    assert (len(set(est.random_state for est in rusboost.estimators_)) ==
            len(rusboost.estimators_))

    # check the consistency of the feature importances
    assert len(rusboost.feature_importances_) == imbalanced_dataset[0].shape[1]

    # check the consistency of the prediction outpus
    y_pred = rusboost.predict_proba(X_test)
    assert y_pred.shape[1] == len(classes)
    assert rusboost.decision_function(X_test).shape[1] == len(classes)

    score = rusboost.score(X_test, y_test)
    assert score > 0.7, "Failed with algorithm {} and score {}".format(
        algorithm, score)

    y_pred = rusboost.predict(X_test)
    assert y_pred.shape == y_test.shape

