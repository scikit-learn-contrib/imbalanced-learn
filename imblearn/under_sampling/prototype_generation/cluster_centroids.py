"""Class to perform under-sampling by generating centroids based on
clustering."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

from __future__ import division, print_function

import numpy as np
from sklearn.cluster import KMeans

from ..base import BaseUnderSampler


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
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    estimator : object, optional(default=KMeans())
        Pass a :class:`sklearn.cluster.KMeans` estimator.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Notes
    -----
    Supports mutli-class resampling by sampling each class independently.

    See :ref:`sphx_glr_auto_examples_under-sampling_plot_cluster_centroids.py`.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
ClusterCentroids # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> cc = ClusterCentroids(random_state=42)
    >>> X_res, y_res = cc.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 100, 1: 100})

    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 estimator=None,
                 n_jobs=1):
        super(ClusterCentroids, self).__init__(
            ratio=ratio, random_state=random_state)
        self.estimator = estimator
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the KMeans estimator"""
        if self.estimator is None:
            self.estimator_ = KMeans(
                random_state=self.random_state, n_jobs=self.n_jobs)
        elif isinstance(self.estimator, KMeans):
            self.estimator_ = self.estimator
        else:
            raise ValueError('`estimator` has to be a KMeans clustering.'
                             ' Got {} instead.'.format(type(self.estimator)))

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """
        self._validate_estimator()

        X_resampled = np.empty((0, X.shape[1]), dtype=X.dtype)
        y_resampled = np.empty((0, ), dtype=y.dtype)

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                n_samples = self.ratio_[target_class]
                self.estimator_.set_params(**{'n_clusters': n_samples})
                self.estimator_.fit(X[y == target_class])
                centroids = self.estimator_.cluster_centers_

                X_resampled = np.concatenate((X_resampled, centroids), axis=0)
                y_resampled = np.concatenate(
                    (y_resampled, np.array([target_class] * n_samples)),
                    axis=0)
            else:

                X_resampled = np.concatenate(
                    (X_resampled, X[y == target_class]), axis=0)
                y_resampled = np.concatenate(
                    (y_resampled, y[y == target_class]), axis=0)

        return X_resampled, y_resampled
