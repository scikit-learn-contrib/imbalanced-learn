import pytest
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets

from imblearn.base import BaseSampler
from imblearn.utils.estimator_checks import check_estimator
from imblearn.utils import check_target_type


class BaseBadSampler(BaseEstimator):
    """Sampler without inputs checking."""

    _sampling_type = "bypass"

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        check_classification_targets(y)
        self.fit(X, y)
        return X, y


class NotFittedSampler(BaseBadSampler):
    """Sampler without target checking."""

    def fit(self, X, y):
        y, _ = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=True)
        return self


class NoAcceptingSparseSampler(BaseBadSampler):
    """Sampler which does not accept sparse matrix."""

    def fit(self, X, y):
        y, _ = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=False)
        self.sampling_strategy_ = "sampling_strategy_"
        return self


class NotPreservingDtypeSampler(BaseSampler):
    _sampling_type = "bypass"

    def _fit_resample(self, X, y):
        return X.astype(np.float64), y.astype(np.int64)


@pytest.mark.filterwarnings("ignore:'y' should be of types")
@pytest.mark.filterwarnings("ignore: Can't check dok sparse matrix for nan")
@pytest.mark.parametrize(
    "Estimator, err_type, err_msg",
    [
        (BaseBadSampler, AssertionError, "ValueError not raised by fit"),
        (NotFittedSampler, AssertionError, "No fitted attribute"),
        (NoAcceptingSparseSampler, TypeError, "A sparse matrix was passed"),
        (
            NotPreservingDtypeSampler,
            AssertionError,
            "X dtype is not preserved",
        ),
    ],
)
def test_check_estimator(Estimator, err_type, err_msg):
    with pytest.raises(err_type, match=err_msg):
        check_estimator(Estimator)
