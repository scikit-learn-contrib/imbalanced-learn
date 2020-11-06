"""Class to perform random under-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np

from sklearn.utils import check_random_state
from sklearn.utils import _safe_indexing

from ..base import BaseUnderSampler
from ...dask._support import is_dask_container
from ...utils import check_target_type
from ...utils import Substitution
from ...utils._docstring import _random_state_docstring
from ...utils._validation import _deprecate_positional_args


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring,
)
class RandomUnderSampler(BaseUnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    replacement : bool, default=False
        Whether the sample is with or without replacement.

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    See Also
    --------
    NearMiss : Undersample using near-miss samples.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.
    Supports heterogeneous data as object array containing string and numeric
    data.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})
    """

    @_deprecate_positional_args
    def __init__(
        self, *, sampling_strategy="auto", random_state=None, replacement=False
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.replacement = replacement

    def _check_X_y(self, X, y):
        if is_dask_container(y) and hasattr(y, "to_dask_array"):
            y = y.to_dask_array()
            y.compute_chunk_sizes()
        y, binarize_y, self._uniques = check_target_type(
            y,
            indicate_one_vs_all=True,
            return_unique=True,
        )
        if not any([is_dask_container(arr) for arr in (X, y)]):
            X, y = self._validate_data(
                X,
                y,
                reset=True,
                accept_sparse=["csr", "csc"],
                dtype=None,
                force_all_finite=False,
            )
        elif is_dask_container(X) and hasattr(X, "to_dask_array"):
            X = X.to_dask_array()
            X.compute_chunk_sizes()
        return X, y, binarize_y

    @staticmethod
    def _find_target_class_indices(y, target_class):
        target_class_indices = np.flatnonzero(y == target_class)
        if is_dask_container(y):
            return target_class_indices.compute()
        return target_class_indices

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)

        idx_under = []

        for target_class in self._uniques:
            target_class_indices = self._find_target_class_indices(
                y, target_class
            )
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                index_target_class = random_state.choice(
                    target_class_indices.size,
                    size=n_samples,
                    replace=self.replacement,
                )
            else:
                index_target_class = slice(None)

            selected_indices = target_class_indices[index_target_class]
            idx_under.append(selected_indices)

        self.sample_indices_ = np.hstack(idx_under)
        self.sample_indices_.sort()

        return (
            _safe_indexing(X, self.sample_indices_),
            _safe_indexing(y, self.sample_indices_)
        )

    def _more_tags(self):
        return {
            "X_types": [
                "2darray",
                "string",
                "dask-array",
                "dask-dataframe"
            ],
            "sample_indices": True,
            "allow_nan": True,
        }
