﻿"""Class to perform random over-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections.abc import Mapping
from numbers import Real

import numpy as np
from scipy import sparse
from sklearn.utils import _safe_indexing, check_array, check_random_state
from sklearn.utils.sparsefuncs import mean_variance_axis

from ..utils import Substitution, check_target_type
from ..utils._docstring import _random_state_docstring
from ..utils._param_validation import Interval
from ..utils._validation import _check_X
from .base import BaseOverSampler


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RandomOverSampler(BaseOverSampler):
    """Class to perform random over-sampling.

    Object to over-sample the minority class(es) by picking samples at random
    with replacement. The bootstrap can be generated in a smoothed manner.

    Read more in the :ref:`User Guide <random_over_sampler>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    shrinkage : float or dict, default=None
        Parameter controlling the shrinkage applied to the covariance matrix.
        when a smoothed bootstrap is generated. The options are:

        - if `None`, a normal bootstrap will be generated without perturbation.
          It is equivalent to `shrinkage=0` as well;
        - if a `float` is given, the shrinkage factor will be used for all
          classes to generate the smoothed bootstrap;
        - if a `dict` is given, the shrinkage factor will specific for each
          class. The key correspond to the targeted class and the value is
          the shrinkage factor.

        The value needs of the shrinkage parameter needs to be higher or equal
        to 0.

        .. versionadded:: 0.8

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    shrinkage_ : dict or None
        The per-class shrinkage factor used to generate the smoothed bootstrap
        sample. When `shrinkage=None` a normal bootstrap will be generated.

        .. versionadded:: 0.8

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SMOTE : Over-sample using SMOTE.

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    When generating a smoothed bootstrap, this method is also known as Random
    Over-Sampling Examples (ROSE) [1]_.

    .. warning::
       Since smoothed bootstrap are generated by adding a small perturbation
       to the drawn samples, this method is not adequate when working with
       sparse matrices.

    References
    ----------
    .. [1] G Menardi, N. Torelli, "Training and assessing classification
       rules with imbalanced data," Data Mining and Knowledge
       Discovery, 28(1), pp.92-122, 2014.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import RandomOverSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> ros = RandomOverSampler(random_state=42)
    >>> X_res, y_res = ros.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})
    """

    _parameter_constraints: dict = {
        **BaseOverSampler._parameter_constraints,
        "shrinkage": [Interval(Real, 0, None, closed="left"), dict, None],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        shrinkage=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.shrinkage = shrinkage

    def _check_X_y(self, X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X = _check_X(X)
        self._check_n_features(X, reset=True)
        self._check_feature_names(X, reset=True)
        return X, y, binarize_y

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)

        if isinstance(self.shrinkage, Real):
            self.shrinkage_ = {
                klass: self.shrinkage for klass in self.sampling_strategy_
            }
        elif self.shrinkage is None or isinstance(self.shrinkage, Mapping):
            self.shrinkage_ = self.shrinkage

        if self.shrinkage_ is not None:
            missing_shrinkage_keys = (
                self.sampling_strategy_.keys() - self.shrinkage_.keys()
            )
            if missing_shrinkage_keys:
                raise ValueError(
                    "`shrinkage` should contain a shrinkage factor for "
                    "each class that will be resampled. The missing "
                    f"classes are: {repr(missing_shrinkage_keys)}"
                )

            for klass, shrink_factor in self.shrinkage_.items():
                if shrink_factor < 0:
                    raise ValueError(
                        "The shrinkage factor needs to be >= 0. "
                        f"Got {shrink_factor} for class {klass}."
                    )

            # smoothed bootstrap imposes to make numerical operation; we need
            # to be sure to have only numerical data in X
            try:
                X = check_array(X, accept_sparse=["csr", "csc"], dtype="numeric")
            except ValueError as exc:
                raise ValueError(
                    "When shrinkage is not None, X needs to contain only "
                    "numerical data to later generate a smoothed bootstrap "
                    "sample."
                ) from exc

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        sample_indices = range(X.shape[0])
        for class_sample, num_samples in self.sampling_strategy_.items():
            target_class_indices = np.flatnonzero(y == class_sample)
            bootstrap_indices = random_state.choice(
                target_class_indices,
                size=num_samples,
                replace=True,
            )
            sample_indices = np.append(sample_indices, bootstrap_indices)
            if self.shrinkage_ is not None:
                # generate a smoothed bootstrap with a perturbation
                n_samples, n_features = X.shape
                smoothing_constant = (4 / ((n_features + 2) * n_samples)) ** (
                    1 / (n_features + 4)
                )
                if sparse.issparse(X):
                    _, X_class_variance = mean_variance_axis(
                        X[target_class_indices, :],
                        axis=0,
                    )
                    X_class_scale = np.sqrt(X_class_variance, out=X_class_variance)
                else:
                    X_class_scale = np.std(X[target_class_indices, :], axis=0)
                smoothing_matrix = np.diagflat(
                    self.shrinkage_[class_sample] * smoothing_constant * X_class_scale
                )
                X_new = random_state.randn(num_samples, n_features)
                X_new = X_new.dot(smoothing_matrix) + X[bootstrap_indices, :]
                if sparse.issparse(X):
                    X_new = sparse.csr_matrix(X_new, dtype=X.dtype)
                X_resampled.append(X_new)
            else:
                # generate a bootstrap
                X_resampled.append(_safe_indexing(X, bootstrap_indices))

            y_resampled.append(_safe_indexing(y, bootstrap_indices))

        self.sample_indices_ = np.array(sample_indices)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled

    def _more_tags(self):
        return {
            "X_types": ["2darray", "string", "sparse", "dataframe"],
            "sample_indices": True,
            "allow_nan": True,
            "_xfail_checks": {
                "check_complex_data": "Robust to this type of data.",
            },
        }
