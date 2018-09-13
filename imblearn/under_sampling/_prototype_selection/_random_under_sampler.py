"""Class to perform random under-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import numpy as np

from sklearn.utils import check_array
from sklearn.utils import check_consistent_length
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing

from ..base import BaseUnderSampler
from ...utils import check_target_type
from ...utils import Substitution
from ...utils.deprecation import deprecate_parameter
from ...utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class RandomUnderSampler(BaseUnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected.

        .. deprecated:: 0.4
           ``return_indices`` is deprecated. Use the attribute
           ``sample_indices_`` instead.

    {random_state}

    replacement : boolean, optional (default=False)
        Whether the sample is with or without replacement.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    Attributes
    ----------
    sample_indices_ : ndarray, shape (n_new_samples)
        Indices of the samples selected.

        .. versionadded:: 0.4
           ``sample_indices_`` used instead of ``return_indices=True``.

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

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=False,
                 ratio=None):
        super(RandomUnderSampler, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.replacement = replacement

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X = check_array(X, accept_sparse=['csr', 'csc'], dtype=None)
        y = check_array(y, accept_sparse=['csr', 'csc'], dtype=None,
                        ensure_2d=False)
        check_consistent_length(X, y)
        return X, y, binarize_y

    def _fit_resample(self, X, y):
        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')
        random_state = check_random_state(self.random_state)

        idx_under = np.empty((0, ), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(y == target_class)),
                    size=n_samples,
                    replace=self.replacement)
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (idx_under,
                 np.flatnonzero(y == target_class)[index_target_class]),
                axis=0)

        self.sample_indices_ = idx_under

        if self.return_indices:
            return (safe_indexing(X, idx_under), safe_indexing(y, idx_under),
                    idx_under)
        return safe_indexing(X, idx_under), safe_indexing(y, idx_under)
