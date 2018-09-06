import pytest

import numpy as np

from sklearn.datasets import make_classification
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


def test_rusboost(imbalanced_dataset):
    X, y = imbalanced_dataset

    n_estimators = 10
    rusboost = RUSBoostClassifier(n_estimators=n_estimators, random_state=0)
    rusboost.fit(X, y)

    assert len(rusboost.samplers_) == n_estimators
    assert len(rusboost.estimators_) == n_estimators
    assert len(rusboost.pipelines_) == n_estimators
    assert len(rusboost.feature_importances_) == imbalanced_dataset[0].shape[1]

    rusboost.predict(X)
    rusboost.predict_proba(X)
    rusboost.predict_log_proba(X)
    rusboost.staged_predict(X)
    rusboost.staged_predict_proba(X)
    rusboost.decision_function(X)
