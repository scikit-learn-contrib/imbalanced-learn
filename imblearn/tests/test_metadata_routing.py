import numpy as np
from sklearn import config_context
from sklearn.linear_model import LogisticRegression

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler


class _CostSensitiveSampler(RandomUnderSampler):
    def fit_resample(self, X, y, cost_matrix=None, **params):
        self.cost_matrix_ = cost_matrix
        return super().fit_resample(X, y, **params)


def test_fit_resample_request():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 1, 1])
    cost_matrix = np.arange(16).reshape(4, 2, 2)
    sampler = _CostSensitiveSampler()

    with config_context(enable_metadata_routing=True):
        configured: _CostSensitiveSampler = sampler.set_fit_resample_request(
            cost_matrix=True
        )
        assert configured is sampler
        pipeline = Pipeline([("sampler", configured), ("model", LogisticRegression())])
        pipeline.fit(X, y, cost_matrix=cost_matrix)

    np.testing.assert_array_equal(configured.cost_matrix_, cost_matrix)


def test_sampler_without_metadata_request():
    assert not hasattr(RandomUnderSampler(), "set_fit_resample_request")
