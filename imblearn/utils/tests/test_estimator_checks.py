import numpy as np
import pytest
from sklearn.base import BaseEstimator
from sklearn.utils.multiclass import check_classification_targets
from sklearn_compat.utils.validation import validate_data

from imblearn.base import BaseSampler
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import check_target_type as target_check
from imblearn.utils.estimator_checks import (
    check_sampler_get_feature_names_out,
    check_sampler_get_feature_names_out_pandas,
    check_samplers_2d_target,
    check_samplers_fit,
    check_samplers_fit_resample,
    check_samplers_list,
    check_samplers_multiclass_ova,
    check_samplers_nan,
    check_samplers_one_label,
    check_samplers_pandas,
    check_samplers_pandas_sparse,
    check_samplers_preserve_dtype,
    check_samplers_sample_indices,
    check_samplers_sampling_strategy_fit_resample,
    check_samplers_sparse,
    check_samplers_string,
    check_target_type,
)


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
        X, y = validate_data(self, X=X, y=y)
        return self


class NoAcceptingSparseSampler(BaseBadSampler):
    """Sampler which does not accept sparse matrix."""

    def fit(self, X, y):
        X, y = validate_data(self, X=X, y=y)
        self.sampling_strategy_ = "sampling_strategy_"
        return self


class ArrayOutputSampler(BaseBadSampler):
    """Sampler which does not preserve container types."""

    def fit_resample(self, X, y):
        self.fit(X, y)
        return np.asarray(X), np.asarray(y)


class DuplicatingOverSampler(BaseOverSampler):
    """Over-sampler which modifies classes outside the sampling strategy."""

    _parameter_constraints: dict = {"sampling_strategy": "no_validation"}

    def _fit_resample(self, X, y):
        return np.vstack([X, X]), np.hstack([y, y])


class BadMulticlassOvaSampler(BaseBadSampler):
    """Sampler returning inconsistent results for one-vs-all encoded targets."""

    def fit_resample(self, X, y):
        self.fit(X, y)
        y = np.asarray(y)
        if y.ndim == 2:
            return X, np.zeros_like(y)
        return X, y


class No2DTargetSampler(BaseBadSampler):
    """Sampler which rejects a two-dimensional target."""

    def fit_resample(self, X, y):
        y = np.asarray(y)
        if y.ndim == 2:
            raise ValueError("2D targets are not supported.")
        return super().fit_resample(X, y)


class NotPreservingDtypeSampler(BaseSampler):
    _sampling_type = "bypass"

    _parameter_constraints: dict = {"sampling_strategy": "no_validation"}

    def _fit_resample(self, X, y):
        return X.astype(np.float64), y.astype(np.int64)


class UnexpectedSampleIndicesSampler(BaseSampler):
    _sampling_type = "bypass"

    _parameter_constraints: dict = {"sampling_strategy": "no_validation"}

    def _fit_resample(self, X, y):
        self.sample_indices_ = np.arange(X.shape[0])
        return X, y


class MissingSampleIndicesSampler(BaseSampler):
    _sampling_type = "bypass"

    _parameter_constraints: dict = {"sampling_strategy": "no_validation"}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.sampler_tags.sample_indices = True
        return tags

    def _fit_resample(self, X, y):
        return X, y


class BadFeatureNamesOutSampler(BaseSampler):
    _sampling_type = "bypass"

    _parameter_constraints: dict = {"sampling_strategy": "no_validation"}

    def _fit_resample(self, X, y):
        return X, y

    def get_feature_names_out(self, input_features=None):
        return np.array(["feature0"], dtype=object)


class IndicesSampler(BaseOverSampler):
    def _check_X_y(self, X, y):
        y, binarize_y = target_check(y, indicate_one_vs_all=True)
        X, y = validate_data(
            self,
            X=X,
            y=y,
            reset=True,
            dtype=None,
            ensure_all_finite=False,
        )
        return X, y, binarize_y

    def _fit_resample(self, X, y):
        n_max_count_class = np.bincount(y).max()
        indices = np.random.choice(np.arange(X.shape[0]), size=n_max_count_class * 2)
        return X[indices], y[indices]


def test_check_samplers_string():
    sampler = IndicesSampler()
    check_samplers_string(sampler.__class__.__name__, sampler)


def test_check_samplers_nan():
    sampler = IndicesSampler()
    check_samplers_nan(sampler.__class__.__name__, sampler)


mapping_estimator_error = {
    "BaseBadSampler": (AssertionError, None),
    "SamplerSingleClass": (AssertionError, "Sampler can't balance when only"),
    "NotFittedSampler": (AssertionError, "No fitted attribute"),
    "NoAcceptingSparseSampler": (TypeError, "dense data is required"),
    "DuplicatingOverSampler": (AssertionError, None),
    "BadMulticlassOvaSampler": (AssertionError, None),
    "No2DTargetSampler": (ValueError, "2D targets are not supported"),
    "NotPreservingDtypeSampler": (AssertionError, "X dtype is not preserved"),
    "ArrayOutputSampler": (AssertionError, None),
    "UnexpectedSampleIndicesSampler": (AssertionError, None),
    "MissingSampleIndicesSampler": (AssertionError, None),
    "BadFeatureNamesOutSampler": (AssertionError, None),
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
    _test_single_check(DuplicatingOverSampler, check_samplers_fit_resample)
    _test_single_check(
        DuplicatingOverSampler, check_samplers_sampling_strategy_fit_resample
    )
    _test_single_check(NoAcceptingSparseSampler, check_samplers_sparse)
    pytest.importorskip("pandas")
    _test_single_check(ArrayOutputSampler, check_samplers_pandas)
    _test_single_check(ArrayOutputSampler, check_samplers_pandas_sparse)
    _test_single_check(ArrayOutputSampler, check_samplers_list)
    _test_single_check(BadMulticlassOvaSampler, check_samplers_multiclass_ova)
    _test_single_check(No2DTargetSampler, check_samplers_2d_target)
    _test_single_check(NotPreservingDtypeSampler, check_samplers_preserve_dtype)
    _test_single_check(UnexpectedSampleIndicesSampler, check_samplers_sample_indices)
    _test_single_check(MissingSampleIndicesSampler, check_samplers_sample_indices)
    _test_single_check(BadFeatureNamesOutSampler, check_sampler_get_feature_names_out)
    _test_single_check(
        BadFeatureNamesOutSampler, check_sampler_get_feature_names_out_pandas
    )
