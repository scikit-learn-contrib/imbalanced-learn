"""Class to perform random over-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT
from __future__ import division

from collections import Counter

import numpy as np
from sklearn.utils import check_X_y, check_random_state, safe_indexing

from .base import BaseOverSampler
from ..utils import check_target_type
from ..utils import Substitution
from ..utils.deprecation import deprecate_parameter
from ..utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class RandomOverSampler(BaseOverSampler):
    """Class to perform random over-sampling.

    Object to over-sample the minority class(es) by picking samples at random
    with replacement.

    Read more in the :ref:`User Guide <random_over_sampler>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        in the corresponding classes.

        .. deprecated:: 0.4
           ``return_indices`` is deprecated. Use the attribute
           ``sample_indices_`` instead.

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
    >>> from imblearn.over_sampling import \
RandomOverSampler # doctest: +NORMALIZE_WHITESPACE
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

    def __init__(self, sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 ratio=None):
        super(RandomOverSampler, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.return_indices = return_indices
        self.random_state = random_state

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], dtype=None)
        return X, y, binarize_y

    def _fit_resample(self, X, y):
        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')

        random_state = check_random_state(self.random_state)
        target_stats = Counter(y)

        sample_indices = range(X.shape[0])

        for class_sample, num_samples in self.sampling_strategy_.items():
            target_class_indices = np.flatnonzero(y == class_sample)
            indices = random_state.randint(
                low=0, high=target_stats[class_sample], size=num_samples)

            sample_indices = np.append(sample_indices,
                                       target_class_indices[indices])
        self.sample_indices_ = np.array(sample_indices)

        if self.return_indices:
            return (safe_indexing(X, sample_indices),
                    safe_indexing(y, sample_indices), sample_indices)
        return (safe_indexing(X, sample_indices),
                safe_indexing(y, sample_indices))
