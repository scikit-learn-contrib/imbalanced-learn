"""Utils to check the samplers and compatibility with scikit-learn"""

# Adapated from scikit-learn
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import sys
import traceback

from collections import Counter

import pytest

import numpy as np
from scipy import sparse

from sklearn.datasets import (
    make_classification,
    make_multilabel_classification,
)  # noqa
from sklearn.cluster import KMeans
from sklearn.preprocessing import label_binarize
from sklearn.utils.estimator_checks import (
    check_estimator as sklearn_check_estimator,
    check_parameters_default_constructible,
)
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import set_random_state
from sklearn.utils.multiclass import type_of_target

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling.base import BaseCleaningSampler, BaseUnderSampler
from imblearn.under_sampling import NearMiss, ClusterCentroids


def _yield_sampler_checks(name, Estimator):
    yield check_target_type
    yield check_samplers_one_label
    yield check_samplers_fit
    yield check_samplers_fit_resample
    yield check_samplers_sampling_strategy_fit_resample
    yield check_samplers_sparse
    yield check_samplers_pandas
    yield check_samplers_list
    yield check_samplers_multiclass_ova
    yield check_samplers_preserve_dtype
    yield check_samplers_sample_indices
    yield check_samplers_2d_target


def _yield_classifier_checks(name, Estimator):
    yield check_classifier_on_multilabel_or_multioutput_targets


def _yield_all_checks(name, estimator):
    # trigger our checks if this is a SamplerMixin
    if hasattr(estimator, "fit_resample"):
        for check in _yield_sampler_checks(name, estimator):
            yield check
    if hasattr(estimator, "predict"):
        for check in _yield_classifier_checks(name, estimator):
            yield check


def check_estimator(Estimator, run_sampler_tests=True):
    """Check if estimator adheres to scikit-learn conventions and
    imbalanced-learn

    This estimator will run an extensive test-suite for input validation,
    shapes, etc.
    Additional tests samplers if the Estimator inherits from the corresponding
    mixin from imblearn.base

    Parameters
    ----------
    Estimator : class
        Class to check. Estimator is a class object (not an instance)

    run_sampler_tests=True : bool, default=True
        Will run or not the samplers tests.
    """
    name = Estimator.__name__
    # scikit-learn common tests
    sklearn_check_estimator(Estimator)
    check_parameters_default_constructible(name, Estimator)
    if run_sampler_tests:
        for check in _yield_all_checks(name, Estimator):
            check(name, Estimator)


def check_target_type(name, Estimator):
    # should raise warning if the target is continuous (we cannot raise error)
    X = np.random.random((20, 2))
    y = np.linspace(0, 1, 20)
    estimator = Estimator()
    set_random_state(estimator)
    with pytest.raises(ValueError, match="Unknown label type: 'continuous'"):
        estimator.fit_resample(X, y)
    # if the target is multilabel then we should raise an error
    rng = np.random.RandomState(42)
    y = rng.randint(2, size=(20, 3))
    msg = "Multilabel and multioutput targets are not supported."
    with pytest.raises(ValueError, match=msg):
        estimator.fit_resample(X, y)


def check_samplers_one_label(name, Sampler):
    error_string_fit = "Sampler can't balance when only one class is present."
    sampler = Sampler()
    X = np.random.random((20, 2))
    y = np.zeros(20)
    try:
        sampler.fit_resample(X, y)
    except ValueError as e:
        if "class" not in repr(e):
            print(error_string_fit, Sampler, e)
            traceback.print_exc(file=sys.stdout)
            raise e
        else:
            return
    except Exception as exc:
        print(error_string_fit, traceback, exc)
        traceback.print_exc(file=sys.stdout)
        raise exc


def check_samplers_fit(name, Sampler):
    sampler = Sampler()
    np.random.seed(42)  # Make this test reproducible
    X = np.random.random((30, 2))
    y = np.array([1] * 20 + [0] * 10)
    sampler.fit_resample(X, y)
    assert hasattr(
        sampler, "sampling_strategy_"
    ), "No fitted attribute sampling_strategy_"


def check_samplers_fit_resample(name, Sampler):
    sampler = Sampler()
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    target_stats = Counter(y)
    X_res, y_res = sampler.fit_resample(X, y)
    if isinstance(sampler, BaseOverSampler):
        target_stats_res = Counter(y_res)
        n_samples = max(target_stats.values())
        assert all(value >= n_samples for value in Counter(y_res).values())
    elif isinstance(sampler, BaseUnderSampler):
        n_samples = min(target_stats.values())
        if name == "InstanceHardnessThreshold":
            # IHT does not enforce the number of samples but provide a number
            # of samples the closest to the desired target.
            assert all(
                Counter(y_res)[k] <= target_stats[k]
                for k in target_stats.keys()
            )
        else:
            assert all(value == n_samples for value in Counter(y_res).values())
    elif isinstance(sampler, BaseCleaningSampler):
        target_stats_res = Counter(y_res)
        class_minority = min(target_stats, key=target_stats.get)
        assert all(
            target_stats[class_sample] > target_stats_res[class_sample]
            for class_sample in target_stats.keys()
            if class_sample != class_minority
        )


def check_samplers_sampling_strategy_fit_resample(name, Sampler):
    # in this test we will force all samplers to not change the class 1
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    sampler = Sampler()
    expected_stat = Counter(y)[1]
    if isinstance(sampler, BaseOverSampler):
        sampling_strategy = {2: 498, 0: 498}
        sampler.set_params(sampling_strategy=sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        assert Counter(y_res)[1] == expected_stat
    elif isinstance(sampler, BaseUnderSampler):
        sampling_strategy = {2: 201, 0: 201}
        sampler.set_params(sampling_strategy=sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        assert Counter(y_res)[1] == expected_stat
    elif isinstance(sampler, BaseCleaningSampler):
        sampling_strategy = [2, 0]
        sampler.set_params(sampling_strategy=sampling_strategy)
        X_res, y_res = sampler.fit_resample(X, y)
        assert Counter(y_res)[1] == expected_stat


def check_samplers_sparse(name, Sampler):
    # check that sparse matrices can be passed through the sampler leading to
    # the same results than dense
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    X_sparse = sparse.csr_matrix(X)
    if isinstance(Sampler(), NearMiss):
        samplers = [Sampler(version=version) for version in (1, 2, 3)]
    elif isinstance(Sampler(), ClusterCentroids):
        # set KMeans to full since it support sparse and dense
        samplers = [
            Sampler(
                random_state=0,
                voting="soft",
                estimator=KMeans(random_state=1, algorithm="full"),
            )
        ]
    else:
        samplers = [Sampler()]

    for sampler in samplers:
        set_random_state(sampler)
        X_res_sparse, y_res_sparse = sampler.fit_resample(X_sparse, y)
        X_res, y_res = sampler.fit_resample(X, y)
        assert sparse.issparse(X_res_sparse)
        assert_allclose(X_res_sparse.A, X_res)
        assert_allclose(y_res_sparse, y_res)


def check_samplers_pandas(name, Sampler):
    pd = pytest.importorskip("pandas")
    # Check that the samplers handle pandas dataframe and pandas series
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    X_df = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
    y_df = pd.DataFrame(y)
    y_s = pd.Series(y, name="class")
    sampler = Sampler()
    if isinstance(Sampler(), NearMiss):
        samplers = [Sampler(version=version) for version in (1, 2, 3)]

    else:
        samplers = [Sampler()]

    for sampler in samplers:
        set_random_state(sampler)
        X_res_df, y_res_s = sampler.fit_resample(X_df, y_s)
        X_res_df, y_res_df = sampler.fit_resample(X_df, y_df)
        X_res, y_res = sampler.fit_resample(X, y)

        # check that we return the same type for dataframes or series types
        assert isinstance(X_res_df, pd.DataFrame)
        assert isinstance(y_res_df, pd.DataFrame)
        assert isinstance(y_res_s, pd.Series)

        assert X_df.columns.to_list() == X_res_df.columns.to_list()
        assert y_df.columns.to_list() == y_res_df.columns.to_list()
        assert y_s.name == y_res_s.name

        assert_allclose(X_res_df.to_numpy(), X_res)
        assert_allclose(y_res_df.to_numpy().ravel(), y_res)
        assert_allclose(y_res_s.to_numpy(), y_res)


def check_samplers_list(name, Sampler):
    # Check that the can samplers handle simple lists
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    X_list = X.tolist()
    y_list = y.tolist()
    sampler = Sampler()
    if isinstance(sampler, NearMiss):
        samplers = [Sampler(version=version) for version in (1, 2, 3)]

    else:
        samplers = [sampler]

    for sampler in samplers:
        set_random_state(sampler)
        X_res, y_res = sampler.fit_resample(X, y)
        X_res_list, y_res_list = sampler.fit_resample(X_list, y_list)

        assert isinstance(X_res_list, list)
        assert isinstance(y_res_list, list)

        assert_allclose(X_res, X_res_list)
        assert_allclose(y_res, y_res_list)


def check_samplers_multiclass_ova(name, Sampler):
    # Check that multiclass target lead to the same results than OVA encoding
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    y_ova = label_binarize(y, np.unique(y))
    sampler = Sampler()
    set_random_state(sampler)
    X_res, y_res = sampler.fit_resample(X, y)
    X_res_ova, y_res_ova = sampler.fit_resample(X, y_ova)
    assert_allclose(X_res, X_res_ova)
    assert type_of_target(y_res_ova) == type_of_target(y_ova)
    assert_allclose(y_res, y_res_ova.argmax(axis=1))


def check_samplers_2d_target(name, Sampler):
    X, y = make_classification(
        n_samples=100,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )

    y = y.reshape(-1, 1)  # Make the target 2d
    sampler = Sampler()
    sampler.fit_resample(X, y)


def check_samplers_preserve_dtype(name, Sampler):
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    # Cast X and y to not default dtype
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    sampler = Sampler()
    set_random_state(sampler)
    X_res, y_res = sampler.fit_resample(X, y)
    assert X.dtype == X_res.dtype, "X dtype is not preserved"
    assert y.dtype == y_res.dtype, "y dtype is not preserved"


def check_samplers_sample_indices(name, Sampler):
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    sampler = Sampler()
    sampler.fit_resample(X, y)
    sample_indices = sampler._get_tags().get("sample_indices", None)
    if sample_indices:
        assert hasattr(sampler, "sample_indices_") is sample_indices
    else:
        assert not hasattr(sampler, "sample_indices_")


def check_classifier_on_multilabel_or_multioutput_targets(name, Estimator):
    estimator = Estimator()
    X, y = make_multilabel_classification(n_samples=30)
    msg = "Multilabel and multioutput targets are not supported."
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y)
