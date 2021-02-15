"""Utils to check the samplers and compatibility with scikit-learn"""

# Adapated from scikit-learn
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import sys
import traceback
import warnings

from collections import Counter
from functools import partial

import pytest

import numpy as np
from scipy import sparse

from sklearn.base import clone
from sklearn.datasets import (
    fetch_openml,
    make_classification,
    make_multilabel_classification,
)  # noqa
from sklearn.cluster import KMeans
from sklearn.exceptions import SkipTestWarning
from sklearn.preprocessing import label_binarize
from sklearn.utils.estimator_checks import _maybe_mark_xfail
from sklearn.utils.estimator_checks import _get_check_estimator_ids
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_raises_regex
from sklearn.utils.multiclass import type_of_target

from imblearn.datasets import make_imbalance
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling.base import BaseCleaningSampler, BaseUnderSampler


def _set_checking_parameters(estimator):
    params = estimator.get_params()
    name = estimator.__class__.__name__
    if "n_estimators" in params:
        estimator.set_params(n_estimators=min(5, estimator.n_estimators))
    if name == "ClusterCentroids":
        estimator.set_params(
            voting="soft",
            estimator=KMeans(random_state=0, algorithm="full", n_init=1),
        )
    if name == "KMeansSMOTE":
        estimator.set_params(kmeans_estimator=12)


def _yield_sampler_checks(sampler):
    tags = sampler._get_tags()
    yield check_target_type
    yield check_samplers_one_label
    yield check_samplers_fit
    yield check_samplers_fit_resample
    yield check_samplers_sampling_strategy_fit_resample
    if "sparse" in tags["X_types"]:
        yield check_samplers_sparse
    if "dataframe" in tags["X_types"]:
        yield check_samplers_pandas
    if "string" in tags["X_types"]:
        yield check_samplers_string
    if tags["allow_nan"]:
        yield check_samplers_nan
    yield check_samplers_list
    yield check_samplers_multiclass_ova
    yield check_samplers_preserve_dtype
    # we don't filter samplers based on their tag here because we want to make
    # sure that the fitted attribute does not exist if the tag is not
    # stipulated
    yield check_samplers_sample_indices
    yield check_samplers_2d_target


def _yield_classifier_checks(classifier):
    yield check_classifier_on_multilabel_or_multioutput_targets
    yield check_classifiers_with_encoded_labels


def _yield_all_checks(estimator):
    name = estimator.__class__.__name__
    tags = estimator._get_tags()
    if tags["_skip_test"]:
        warnings.warn(
            f"Explicit SKIP via _skip_test tag for estimator {name}.",
            SkipTestWarning,
        )
        return
    # trigger our checks if this is a SamplerMixin
    if hasattr(estimator, "fit_resample"):
        for check in _yield_sampler_checks(estimator):
            yield check
    if hasattr(estimator, "predict"):
        for check in _yield_classifier_checks(estimator):
            yield check


def parametrize_with_checks(estimators):
    """Pytest specific decorator for parametrizing estimator checks.

    The `id` of each check is set to be a pprint version of the estimator
    and the name of the check with its keyword arguments.
    This allows to use `pytest -k` to specify which tests to run::

        pytest test_check_estimators.py -k check_estimators_fit_returns_self

    Parameters
    ----------
    estimators : list of estimators instances
        Estimators to generated checks for.

    Returns
    -------
    decorator : `pytest.mark.parametrize`

    Examples
    --------
    >>> from sklearn.utils.estimator_checks import parametrize_with_checks
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.tree import DecisionTreeRegressor

    >>> @parametrize_with_checks([LogisticRegression(),
    ...                           DecisionTreeRegressor()])
    ... def test_sklearn_compatible_estimator(estimator, check):
    ...     check(estimator)
    """

    def checks_generator():
        for estimator in estimators:
            name = type(estimator).__name__
            for check in _yield_all_checks(estimator):
                check = partial(check, name)
                yield _maybe_mark_xfail(estimator, check, pytest)

    return pytest.mark.parametrize(
        "estimator, check", checks_generator(), ids=_get_check_estimator_ids
    )


def check_target_type(name, estimator_orig):
    estimator = clone(estimator_orig)
    # should raise warning if the target is continuous (we cannot raise error)
    X = np.random.random((20, 2))
    y = np.linspace(0, 1, 20)
    msg = "Unknown label type: 'continuous'"
    assert_raises_regex(
        ValueError,
        msg,
        estimator.fit_resample,
        X,
        y,
    )
    # if the target is multilabel then we should raise an error
    rng = np.random.RandomState(42)
    y = rng.randint(2, size=(20, 3))
    msg = "Multilabel and multioutput targets are not supported."
    assert_raises_regex(
        ValueError,
        msg,
        estimator.fit_resample,
        X,
        y,
    )


def check_samplers_one_label(name, sampler_orig):
    sampler = clone(sampler_orig)
    error_string_fit = "Sampler can't balance when only one class is present."
    X = np.random.random((20, 2))
    y = np.zeros(20)
    try:
        sampler.fit_resample(X, y)
    except ValueError as e:
        if "class" not in repr(e):
            print(error_string_fit, sampler.__class__.__name__, e)
            traceback.print_exc(file=sys.stdout)
            raise e
        else:
            return
    except Exception as exc:
        print(error_string_fit, traceback, exc)
        traceback.print_exc(file=sys.stdout)
        raise exc
    raise AssertionError(error_string_fit)


def check_samplers_fit(name, sampler_orig):
    sampler = clone(sampler_orig)
    np.random.seed(42)  # Make this test reproducible
    X = np.random.random((30, 2))
    y = np.array([1] * 20 + [0] * 10)
    sampler.fit_resample(X, y)
    assert hasattr(
        sampler, "sampling_strategy_"
    ), "No fitted attribute sampling_strategy_"


def check_samplers_fit_resample(name, sampler_orig):
    sampler = clone(sampler_orig)
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
                Counter(y_res)[k] <= target_stats[k] for k in target_stats.keys()
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


def check_samplers_sampling_strategy_fit_resample(name, sampler_orig):
    sampler = clone(sampler_orig)
    # in this test we will force all samplers to not change the class 1
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
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


def check_samplers_sparse(name, sampler_orig):
    sampler = clone(sampler_orig)
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
    X_res_sparse, y_res_sparse = sampler.fit_resample(X_sparse, y)
    sampler = clone(sampler)
    X_res, y_res = sampler.fit_resample(X, y)
    assert sparse.issparse(X_res_sparse)
    assert_allclose(X_res_sparse.A, X_res, rtol=1e-5)
    assert_allclose(y_res_sparse, y_res)


def check_samplers_pandas(name, sampler_orig):
    pd = pytest.importorskip("pandas")
    sampler = clone(sampler_orig)
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


def check_samplers_list(name, sampler_orig):
    sampler = clone(sampler_orig)
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

    X_res, y_res = sampler.fit_resample(X, y)
    X_res_list, y_res_list = sampler.fit_resample(X_list, y_list)

    assert isinstance(X_res_list, list)
    assert isinstance(y_res_list, list)

    assert_allclose(X_res, X_res_list)
    assert_allclose(y_res, y_res_list)


def check_samplers_multiclass_ova(name, sampler_orig):
    sampler = clone(sampler_orig)
    # Check that multiclass target lead to the same results than OVA encoding
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    y_ova = label_binarize(y, np.unique(y))
    X_res, y_res = sampler.fit_resample(X, y)
    X_res_ova, y_res_ova = sampler.fit_resample(X, y_ova)
    assert_allclose(X_res, X_res_ova)
    assert type_of_target(y_res_ova) == type_of_target(y_ova)
    assert_allclose(y_res, y_res_ova.argmax(axis=1))


def check_samplers_2d_target(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = make_classification(
        n_samples=100,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )

    y = y.reshape(-1, 1)  # Make the target 2d
    sampler.fit_resample(X, y)


def check_samplers_preserve_dtype(name, sampler_orig):
    sampler = clone(sampler_orig)
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
    X_res, y_res = sampler.fit_resample(X, y)
    assert X.dtype == X_res.dtype, "X dtype is not preserved"
    assert y.dtype == y_res.dtype, "y dtype is not preserved"


def check_samplers_sample_indices(name, sampler_orig):
    sampler = clone(sampler_orig)
    X, y = make_classification(
        n_samples=1000,
        n_classes=3,
        n_informative=4,
        weights=[0.2, 0.3, 0.5],
        random_state=0,
    )
    sampler.fit_resample(X, y)
    sample_indices = sampler._get_tags().get("sample_indices", None)
    if sample_indices:
        assert hasattr(sampler, "sample_indices_") is sample_indices
    else:
        assert not hasattr(sampler, "sample_indices_")


def check_samplers_string(name, sampler_orig):
    rng = np.random.RandomState(0)
    sampler = clone(sampler_orig)
    categories = np.array(["A", "B", "C"], dtype=object)
    n_samples = 30
    X = rng.randint(low=0, high=3, size=n_samples).reshape(-1, 1)
    X = categories[X]
    y = rng.permutation([0] * 10 + [1] * 20)

    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.dtype == object
    assert X_res.shape[0] == y_res.shape[0]
    assert_array_equal(np.unique(X_res.ravel()), categories)


def check_samplers_nan(name, sampler_orig):
    rng = np.random.RandomState(0)
    sampler = clone(sampler_orig)
    categories = np.array([0, 1, np.nan], dtype=np.float64)
    n_samples = 100
    X = rng.randint(low=0, high=3, size=n_samples).reshape(-1, 1)
    X = categories[X]
    y = rng.permutation([0] * 40 + [1] * 60)

    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.dtype == np.float64
    assert X_res.shape[0] == y_res.shape[0]
    assert np.any(np.isnan(X_res.ravel()))


def check_classifier_on_multilabel_or_multioutput_targets(name, estimator_orig):
    estimator = clone(estimator_orig)
    X, y = make_multilabel_classification(n_samples=30)
    msg = "Multilabel and multioutput targets are not supported."
    with pytest.raises(ValueError, match=msg):
        estimator.fit(X, y)


def check_classifiers_with_encoded_labels(name, classifier_orig):
    # Non-regression test for #709
    # https://github.com/scikit-learn-contrib/imbalanced-learn/issues/709
    pytest.importorskip("pandas")
    classifier = clone(classifier_orig)
    df, y = fetch_openml("iris", version=1, as_frame=True, return_X_y=True)
    df, y = make_imbalance(
        df,
        y,
        sampling_strategy={
            "Iris-setosa": 30,
            "Iris-versicolor": 20,
            "Iris-virginica": 50,
        },
    )
    classifier.set_params(sampling_strategy={"Iris-setosa": 20, "Iris-virginica": 20})
    classifier.fit(df, y)
    assert set(classifier.classes_) == set(y.cat.categories.tolist())
    y_pred = classifier.predict(df)
    assert set(y_pred) == set(y.cat.categories.tolist())
