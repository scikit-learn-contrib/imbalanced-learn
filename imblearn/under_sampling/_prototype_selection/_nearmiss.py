"""Class to perform under-sampling based on nearmiss methods."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import warnings
from collections import Counter

import numpy as np

from sklearn.utils import safe_indexing

from ..base import BaseUnderSampler
from ...utils import check_neighbors_object
from ...utils import Substitution
from ...utils.deprecation import deprecate_parameter
from ...utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class NearMiss(BaseUnderSampler):
    """Class to perform under-sampling based on NearMiss methods.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

    Parameters
    ----------
    {sampling_strategy}

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

        .. deprecated:: 0.4
           ``return_indices`` is deprecated. Use the attribute
           ``sample_indices_`` instead.

    {random_state}

        .. deprecated:: 0.4
           ``random_state`` is deprecated in 0.4 and will be removed in 0.6.

    version : int, optional (default=1)
        Version of the NearMiss to use. Possible values are 1, 2 or 3.

    n_neighbors : int or object, optional (default=3)
        If ``int``, size of the neighbourhood to consider to compute the
        average distance to the minority point samples.  If object, an
        estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    n_neighbors_ver3 : int or object, optional (default=3)
        If ``int``, NearMiss-3 algorithm start by a phase of re-sampling. This
        parameter correspond to the number of neighbours selected create the
        subset in which the selection will be performed.  If object, an
        estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

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
    The methods are based on [1]_.

    Supports multi-class resampling.

    References
    ----------
    .. [1] I. Mani, I. Zhang. "kNN approach to unbalanced data distributions:
       a case study involving information extraction," In Proceedings of
       workshop on learning from imbalanced datasets, 2003.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
NearMiss # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> nm = NearMiss()
    >>> X_res, y_res = nm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 100, 1: 100}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 version=1,
                 n_neighbors=3,
                 n_neighbors_ver3=3,
                 n_jobs=1,
                 ratio=None):
        super(NearMiss, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.version = version
        self.n_neighbors = n_neighbors
        self.n_neighbors_ver3 = n_neighbors_ver3
        self.n_jobs = n_jobs

    def _selection_dist_based(self,
                              X,
                              y,
                              dist_vec,
                              num_samples,
                              key,
                              sel_strategy='nearest'):
        """Select the appropriate samples depending of the strategy selected.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Original samples.

        y : array-like, shape (n_samples,)
            Associated label to X.

        dist_vec : ndarray, shape (n_samples, )
            The distance matrix to the nearest neigbour.

        num_samples: int
            The desired number of samples to select.

        key : str or int,
            The target class.

        sel_strategy : str, optional (default='nearest')
            Strategy to select the samples. Either 'nearest' or 'farthest'

        Returns
        -------
        idx_sel : ndarray, shape (num_samples,)
            The list of the indices of the selected samples.

        """

        # Compute the distance considering the farthest neighbour
        dist_avg_vec = np.sum(dist_vec[:, -self.nn_.n_neighbors:], axis=1)

        target_class_indices = np.flatnonzero(y == key)
        if (dist_vec.shape[0] != safe_indexing(X,
                                               target_class_indices).shape[0]):
            raise RuntimeError('The samples to be selected do not correspond'
                               ' to the distance matrix given. Ensure that'
                               ' both `X[y == key]` and `dist_vec` are'
                               ' related.')

        # Sort the list of distance and get the index
        if sel_strategy == 'nearest':
            sort_way = False
        elif sel_strategy == 'farthest':
            sort_way = True
        else:
            raise NotImplementedError

        sorted_idx = sorted(
            range(len(dist_avg_vec)),
            key=dist_avg_vec.__getitem__,
            reverse=sort_way)

        # Throw a warning to tell the user that we did not have enough samples
        # to select and that we just select everything
        if len(sorted_idx) < num_samples:
            warnings.warn('The number of the samples to be selected is larger'
                          ' than the number of samples available. The'
                          ' balancing ratio cannot be ensure and all samples'
                          ' will be returned.')

        # Select the desired number of samples
        return sorted_idx[:num_samples]

    def _validate_estimator(self):
        """Private function to create the NN estimator"""

        # check for deprecated random_state
        if self.random_state is not None:
            deprecate_parameter(self, '0.4', 'random_state')

        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

        if self.version == 3:
            self.nn_ver3_ = check_neighbors_object('n_neighbors_ver3',
                                                   self.n_neighbors_ver3)
            self.nn_ver3_.set_params(**{'n_jobs': self.n_jobs})

        if self.version not in (1, 2, 3):
            raise ValueError('Parameter `version` must be 1, 2 or 3, got'
                             ' {}'.format(self.version))

    def _fit_resample(self, X, y):
        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')
        self._validate_estimator()

        idx_under = np.empty((0, ), dtype=int)

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)
        minority_class_indices = np.flatnonzero(y == class_minority)

        self.nn_.fit(safe_indexing(X, minority_class_indices))

        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                target_class_indices = np.flatnonzero(y == target_class)
                X_class = safe_indexing(X, target_class_indices)
                y_class = safe_indexing(y, target_class_indices)

                if self.version == 1:
                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class, n_neighbors=self.nn_.n_neighbors)
                    index_target_class = self._selection_dist_based(
                        X,
                        y,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy='nearest')
                elif self.version == 2:
                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class, n_neighbors=target_stats[class_minority])
                    index_target_class = self._selection_dist_based(
                        X,
                        y,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy='nearest')
                elif self.version == 3:
                    self.nn_ver3_.fit(X_class)
                    dist_vec, idx_vec = self.nn_ver3_.kneighbors(
                        safe_indexing(X, minority_class_indices))
                    idx_vec_farthest = np.unique(idx_vec.reshape(-1))
                    X_class_selected = safe_indexing(X_class, idx_vec_farthest)
                    y_class_selected = safe_indexing(y_class, idx_vec_farthest)

                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class_selected, n_neighbors=self.nn_.n_neighbors)
                    index_target_class = self._selection_dist_based(
                        X_class_selected,
                        y_class_selected,
                        dist_vec,
                        n_samples,
                        target_class,
                        sel_strategy='farthest')
                    # idx_tmp is relative to the feature selected in the
                    # previous step and we need to find the indirection
                    index_target_class = idx_vec_farthest[index_target_class]
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
