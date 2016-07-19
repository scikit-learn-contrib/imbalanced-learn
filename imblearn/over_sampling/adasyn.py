﻿"""Class to perform random over-sampling."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from ..base import SamplerMixin


class ADASYN(SamplerMixin):

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

    k : int, optional (default=5)
        Number of nearest neighbours to used to construct synthetic samples.

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
    Does not support multi-class.

    The implementation is based on [1]_.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import fetch_mldata
    >>> from imblearn.over_sampling import ADASYN
    >>> pima = fetch_mldata('diabetes_scale')
    >>> X, y = pima['data'], pima['target']
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 500, -1: 268})
    >>> ada = ADASYN(random_state=42)
    >>> X_res, y_res = ada.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 500, -1: 478})

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
                 k=5,
                 n_jobs=1):
        super(ADASYN, self).__init__(ratio=ratio)
        self.random_state = random_state
        self.k = k
        self.n_jobs = n_jobs
        self.nearest_neighbour = NearestNeighbors(n_neighbors=self.k + 1,
                                                  n_jobs=self.n_jobs)

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

        # Keep the samples from the majority class
        X_resampled = X.copy()
        y_resampled = y.copy()

        # Define the number of sample to create
        # We handle only two classes problem for the moment.
        if self.ratio == 'auto':
            num_samples = (self.stats_c_[self.maj_c_] -
                           self.stats_c_[self.min_c_])
        else:
            num_samples = int((self.ratio * self.stats_c_[self.maj_c_]) -
                              self.stats_c_[self.min_c_])

        # Start by separating minority class features and target values.
        X_min = X[y == self.min_c_]

        # Print if verbose is true
        self.logger.debug('Finding the %s nearest neighbours ...', self.k)

        # Look for k-th nearest neighbours, excluding, of course, the
        # point itself.
        self.nearest_neighbour.fit(X)

        # Get the distance to the NN
        _, ind_nn = self.nearest_neighbour.kneighbors(X_min)

        # Compute the ratio of majority samples next to minority samples
        ratio_nn = np.sum(y[ind_nn[:, 1:]] == self.maj_c_, axis=1) / self.k
        # Check that we found at least some neighbours belonging to the
        # majority class
        if not np.sum(ratio_nn):
            raise RuntimeError('Not any neigbours belong to the majority'
                               ' class. This case will induce a NaN case with'
                               ' a division by zero. ADASYN is not suited for'
                               ' this specific dataset. Use SMOTE.')
        # Normalize the ratio
        ratio_nn /= np.sum(ratio_nn)

        # Compute the number of sample to be generated
        num_samples_nn = np.round(ratio_nn * num_samples).astype(int)

        # For each minority samples
        for x_i, x_i_nn, num_sample_i in zip(X_min, ind_nn, num_samples_nn):

            # Pick-up the neighbors wanted
            nn_zs = random_state.randint(1, high=self.k + 1, size=num_sample_i)

            # Create a new sample
            for nn_z in nn_zs:
                step = random_state.uniform()
                x_gen = x_i + step * (x_i - X[x_i_nn[nn_z], :])
                X_resampled = np.vstack((X_resampled, x_gen))
                y_resampled = np.hstack((y_resampled, self.min_c_))

        self.logger.info('Over-sampling performed: %s', Counter(
            y_resampled))

        return X_resampled, y_resampled
