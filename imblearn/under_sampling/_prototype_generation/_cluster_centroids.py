"""Class to perform under-sampling by generating centroids based on
clustering."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

from __future__ import division

import numpy as np
from scipy import sparse

from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import safe_indexing

from ..base import BaseUnderSampler
from ...utils import Substitution
from ...utils._docstring import _random_state_docstring

VOTING_KIND = ('auto', 'hard', 'soft')


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class ClusterCentroids(BaseUnderSampler):
    """Perform under-sampling by generating centroids based on
    clustering methods.

    Method that under samples the majority class by replacing a
    cluster of majority samples by the cluster centroid of a KMeans
    algorithm.  This algorithm keeps N majority samples by fitting the
    KMeans algorithm with N cluster to the majority class and using
    the coordinates of the N cluster centroids as the new majority
    samples.

    Read more in the :ref:`User Guide <cluster_centroids>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    estimator : object, optional(default=KMeans())
        Pass a :class:`sklearn.cluster.KMeans` estimator.

    voting : str, optional (default='auto')
        Voting strategy to generate the new samples:

        - If ``'hard'``, the nearest-neighbors of the centroids found using the
          clustering algorithm will be used.
        - If ``'soft'``, the centroids found by the clustering algorithm will
          be used.
        - If ``'auto'``, if the input is sparse, it will default on ``'hard'``
          otherwise, ``'soft'`` will be used.

        .. versionadded:: 0.3.0

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    Notes
    -----
    Supports multi-class resampling by sampling each class independently.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
ClusterCentroids # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> cc = ClusterCentroids(random_state=42)
    >>> X_res, y_res = cc.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    ... # doctest: +ELLIPSIS
    Resampled dataset shape Counter({{...}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 estimator=None,
                 voting='auto',
                 n_jobs=1,
                 ratio=None):
        super(ClusterCentroids, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.estimator = estimator
        self.voting = voting
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the KMeans estimator"""
        if self.estimator is None:
            self.estimator_ = KMeans(
                random_state=self.random_state, n_jobs=self.n_jobs)
        elif isinstance(self.estimator, KMeans):
            self.estimator_ = clone(self.estimator)
        else:
            raise ValueError('`estimator` has to be a KMeans clustering.'
                             ' Got {} instead.'.format(type(self.estimator)))

    def _generate_sample(self, X, y, centroids, target_class):
        if self.voting_ == 'hard':
            nearest_neighbors = NearestNeighbors(n_neighbors=1)
            nearest_neighbors.fit(X, y)
            indices = nearest_neighbors.kneighbors(
                centroids, return_distance=False)
            X_new = safe_indexing(X, np.squeeze(indices))
        else:
            if sparse.issparse(X):
                X_new = sparse.csr_matrix(centroids, dtype=X.dtype)
            else:
                X_new = centroids
        y_new = np.array([target_class] * centroids.shape[0], dtype=y.dtype)

        return X_new, y_new

    def _fit_resample(self, X, y):
        self._validate_estimator()

        if self.voting == 'auto':
            if sparse.issparse(X):
                self.voting_ = 'hard'
            else:
                self.voting_ = 'soft'
        else:
            if self.voting in VOTING_KIND:
                self.voting_ = self.voting
            else:
                raise ValueError("'voting' needs to be one of {}. Got {}"
                                 " instead.".format(VOTING_KIND, self.voting))

        X_resampled, y_resampled = [], []
        for target_class in np.unique(y):
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                self.estimator_.set_params(**{'n_clusters': n_samples})
                self.estimator_.fit(X[y == target_class])
                X_new, y_new = self._generate_sample(
                    X, y, self.estimator_.cluster_centers_, target_class)
                X_resampled.append(X_new)
                y_resampled.append(y_new)
            else:
                target_class_indices = np.flatnonzero(y == target_class)
                X_resampled.append(safe_indexing(X, target_class_indices))
                y_resampled.append(safe_indexing(y, target_class_indices))

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, np.array(y_resampled, dtype=y.dtype)
