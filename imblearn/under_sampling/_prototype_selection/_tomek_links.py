"""Class to perform under-sampling by removing Tomek's links."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

from __future__ import division

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import safe_indexing

from ..base import BaseCleaningSampler
from ...utils import Substitution
from ...utils.deprecation import deprecate_parameter
from ...utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class TomekLinks(BaseCleaningSampler):
    """Class to perform under-sampling by removing Tomek's links.

    Read more in the :ref:`User Guide <tomek_links>`.

    Parameters
    ----------
    {sampling_strategy}

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected.

        .. deprecated:: 0.4
           ``return_indices`` is deprecated. Use the attribute
           ``sample_indices_`` instead.


    {random_state}

        .. deprecated:: 0.4
           ``random_state`` is deprecated in 0.4 and will be removed in 0.6.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

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
    This method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] I. Tomek, "Two modifications of CNN," In Systems, Man, and
       Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 2010.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
TomekLinks # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> tl = TomekLinks()
    >>> X_res, y_res = tl.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 897, 0: 100}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 n_jobs=1,
                 ratio=None):
        super(TomekLinks, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.n_jobs = n_jobs

    @staticmethod
    def is_tomek(y, nn_index, class_type):
        """is_tomek uses the target vector and the first neighbour of every
        sample point and looks for Tomek pairs. Returning a boolean vector with
        True for majority Tomek links.

        Parameters
        ----------
        y : ndarray, shape (n_samples, )
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority or not

        nn_index : ndarray, shape (len(y), )
            The index of the closes nearest neighbour to a sample point.

        class_type : int or str
            The label of the minority class.

        Returns
        -------
        is_tomek : ndarray, shape (len(y), )
            Boolean vector on len( # samples ), with True for majority samples
            that are Tomek links.

        """
        links = np.zeros(len(y), dtype=bool)

        # find which class to not consider
        class_excluded = [c for c in np.unique(y) if c not in class_type]

        # there is a Tomek link between two samples if they are both nearest
        # neighbors of each others.
        for index_sample, target_sample in enumerate(y):
            if target_sample in class_excluded:
                continue

            if y[nn_index[index_sample]] != target_sample:
                if nn_index[nn_index[index_sample]] == index_sample:
                    links[index_sample] = True

        return links

    def _fit_resample(self, X, y):
        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')
        # check for deprecated random_state
        if self.random_state is not None:
            deprecate_parameter(self, '0.4', 'random_state')

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X)
        nns = nn.kneighbors(X, return_distance=False)[:, 1]

        links = self.is_tomek(y, nns, self.sampling_strategy_)
        self.sample_indices_ = np.flatnonzero(np.logical_not(links))

        if self.return_indices:
            return (safe_indexing(X, self.sample_indices_),
                    safe_indexing(y, self.sample_indices_),
                    self.sample_indices_)
        return (safe_indexing(X, self.sample_indices_),
                safe_indexing(y, self.sample_indices_))
