"""Class to perform random over-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division, print_function

from collections import Counter

import numpy as np
from sklearn.utils import check_random_state

from ..base import MultiClassSamplerMixin
from .base import BaseOverSampler
from ..utils import check_neighbors_object


class ADASYN(BaseOverSampler, MultiClassSamplerMixin):
    """Perform over-sampling using ADASYN.

    Perform over-sampling using Adaptive Synthetic Sampling Approach for
    Imbalanced Learning.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    k : int, optional (default=None)
        Number of nearest neighbours to used to construct synthetic samples.

        NOTE: `k` is deprecated from 0.2 and will be replaced in 0.4
        Use ``n_neighbors`` instead.

    n_neighbors : int int or object, optional (default=5)
        If int, number of nearest neighbours to used to construct
        synthetic samples.
        If object, an estimator that inherits from
        `sklearn.neighbors.base.KNeighborsMixin` that will be used to find
        the k_neighbors.

    n_jobs : int, optional (default=1)
        Number of threads to run the algorithm when it is possible.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    Does support multi-class.

    The implementation is based on [1]_.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
    ADASYN # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000,
    ... random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> ada = ADASYN(random_state=42)
    >>> X_res, y_res = ada.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 904, 1: 900})

    References
    ----------
    .. [1] He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. "ADASYN:
       Adaptive synthetic sampling approach for imbalanced learning," In IEEE
       International Joint Conference on Neural Networks (IEEE World Congress
       on Computational Intelligence), pp. 1322-1328, 2008.

    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 k=None,
                 n_neighbors=5,
                 n_jobs=1):
        super(ADASYN, self).__init__(ratio=ratio, random_state=random_state)
        self.k = k
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """

        super(ADASYN, self).fit(X, y)
        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors,
                                          additional_neighbor=1)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

        return self

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
        random_state = check_random_state(self.random_state)

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.ratio_.items():
            if n_samples == 0:
                continue
            X_class = X[y == class_sample]

            self.nn_.fit(X)
            _, nn_index = self.nn_.kneighbors(X_class)
            # The ratio is computed using a one-vs-rest manner. Using majority
            # in multi-class would lead to slightly different results at the
            # cost of introducing a new parameter.
            ratio_nn = (np.sum(y[nn_index[:, 1:]] != class_sample, axis=1) /
                        (self.nn_.n_neighbors - 1))
            if not np.sum(ratio_nn):
                raise RuntimeError('Not any neigbours belong to the majority'
                                   ' class. This case will induce a NaN case'
                                   ' with a division by zero. ADASYN is not'
                                   ' suited for this specific dataset.'
                                   ' Use SMOTE instead.')
            ratio_nn /= np.sum(ratio_nn)
            n_samples_generate = np.round(ratio_nn * n_samples).astype(int)

            for x_i, x_i_nn, num_sample_i in zip(X_class, nn_index,
                                                 n_samples_generate):
                nn_zs = random_state.randint(
                    1, high=self.nn_.n_neighbors, size=num_sample_i)
                for nn_z in nn_zs:
                    step = random_state.uniform()
                    x_gen = x_i + step * (X[x_i_nn[nn_z], :] - x_i)
                    X_resampled = np.vstack((X_resampled, x_gen))
                    y_resampled = np.hstack((y_resampled, class_sample))

        self.logger.info('Over-sampling performed: %s', Counter(y_resampled))

        return X_resampled, y_resampled
