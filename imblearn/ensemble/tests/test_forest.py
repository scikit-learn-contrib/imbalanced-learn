from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_equal

from imblearn.ensemble import BalancedRandomForestClassifier


def test_balanced_random_forest():
    X, y = X, y = make_classification(
        n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
        n_repeated=0, n_classes=3, n_clusters_per_class=1,
        weights=[0.01, 0.05, 0.94], class_sep=0.8, random_state=0)

    n_estimators = 100
    brf = BalancedRandomForestClassifier(n_estimators=n_estimators,
                                         random_state=0)
    brf.fit(X, y)

    assert len(brf.samplers_) == n_estimators
    assert len(brf.estimators_) == n_estimators
    assert len(brf.pipelines_) == n_estimators


def test_balanced_random_forest_attributes():
    X, y = X, y = make_classification(
        n_samples=10000, n_features=2, n_informative=2, n_redundant=0,
        n_repeated=0, n_classes=3, n_clusters_per_class=1,
        weights=[0.01, 0.05, 0.94], class_sep=0.8, random_state=0)

    n_estimators = 100
    brf = BalancedRandomForestClassifier(n_estimators=n_estimators,
                                         random_state=0)
    brf.fit(X, y)

    for idx in range(n_estimators):
        X_res, y_res, _ = brf.samplers_[idx].fit_resample(X, y)
        X_res_2, y_res_2 = brf.pipelines_[idx].named_steps[
            'randomundersampler'].fit_resample(X, y)
        assert_allclose(X_res, X_res_2)
        assert_array_equal(y_res, y_res_2)

        y_pred = brf.estimators_[idx].fit(X_res, y_res).predict(X)
        y_pred_2 = brf.pipelines_[idx].fit(X, y).predict(X)
        assert_array_equal(y_pred, y_pred_2)

        y_pred = brf.estimators_[idx].fit(X_res, y_res).predict_proba(X)
        y_pred_2 = brf.pipelines_[idx].fit(X, y).predict_proba(X)
        assert_array_equal(y_pred, y_pred_2)

