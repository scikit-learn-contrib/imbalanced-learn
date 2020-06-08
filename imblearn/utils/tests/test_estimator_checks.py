import pytest
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import check_classification_targets

from imblearn.base import BaseSampler

from imblearn.utils.estimator_checks import check_target_type
from imblearn.utils.estimator_checks import check_samplers_one_label
from imblearn.utils.estimator_checks import check_samplers_fit
from imblearn.utils.estimator_checks import check_samplers_sparse
from imblearn.utils.estimator_checks import check_samplers_preserve_dtype


class BaseBadSampler(BaseEstimator):
    """Sampler without inputs checking."""

    _sampling_type = "bypass"

    def fit(self, X, y):
        return self

    def fit_resample(self, X, y):
        check_classification_targets(y)
        self.fit(X, y)
        return X, y


class SamplerSingleClass(BaseSampler):
    """Sampler that would sample even with a single class."""
    _sampling_type = "bypass"

    def fit_resample(self, X, y):
        return self._fit_resample(X, y)

    def _fit_resample(self, X, y):
        return X, y


class NotFittedSampler(BaseBadSampler):
    """Sampler without target checking."""

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        return self


class NoAcceptingSparseSampler(BaseBadSampler):
    """Sampler which does not accept sparse matrix."""

    def fit(self, X, y):
        X, y = self._validate_data(X, y)
        self.sampling_strategy_ = "sampling_strategy_"
        return self


class NotPreservingDtypeSampler(BaseSampler):
    _sampling_type = "bypass"

    def _fit_resample(self, X, y):
        return X.astype(np.float64), y.astype(np.int64)


mapping_estimator_error = {
    "BaseBadSampler": (AssertionError, "ValueError not raised by fit"),
    "SamplerSingleClass": (AssertionError, "Sampler can't balance when only"),
    "NotFittedSampler": (AssertionError, "No fitted attribute"),
    "NoAcceptingSparseSampler": (TypeError, "A sparse matrix was passed"),
    "NotPreservingDtypeSampler": (AssertionError, "X dtype is not preserved"),
}


def _test_single_check(Estimator, check):
    estimator = Estimator()
    name = estimator.__class__.__name__
    err_type, err_msg = mapping_estimator_error[name]
    with pytest.raises(err_type, match=err_msg):
        check(name, estimator)


def test_all_checks():
    _test_single_check(BaseBadSampler, check_target_type)
    _test_single_check(SamplerSingleClass, check_samplers_one_label)
    _test_single_check(NotFittedSampler, check_samplers_fit)
    _test_single_check(NoAcceptingSparseSampler, check_samplers_sparse)
    _test_single_check(
        NotPreservingDtypeSampler, check_samplers_preserve_dtype
    )
