import pytest

import numpy as np

from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_equal
from sklearn.model_selection import GridSearchCV

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
    classes = np.unique(y)

    n_estimators = 100
    rusboost = RUSBoostClassifier(n_estimators=n_estimators,
                                  algorithm=algorithm,
                                  random_state=0)
    adaboost = AdaBoostClassifier(n_estimators=n_estimators,
                                  algorithm=algorithm,
                                  random_state=0)
    rusboost.fit(X, y)
    adaboost.fit(X, y)

    assert len(rusboost.feature_importances_) == imbalanced_dataset[0].shape[1]
    assert_array_equal(classes, rusboost.classes_)

    y_pred = rusboost.predict_proba(X)
    assert y_pred.shape[1] == len(classes)
    assert rusboost.decision_function(X).shape[1] == len(classes)

    score = rusboost.score(X, y)
    assert score > 0.8, "Failed with algorithm {} and score {}".format(
        algorithm, score)

    y_pred = rusboost.predict(X)
    assert y_pred.shape == y.shape

    bal_acc_rusboost = balanced_accuracy_score(y, y_pred)
    bal_acc_adaboost = balanced_accuracy_score(y, adaboost.predict(X))
    assert bal_acc_rusboost > bal_acc_adaboost
