"""Class to perform random over-sampling."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from .over_sampler import OverSampler


class ADASYN(OverSampler):
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

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Whether or not to print information about the processing.

    k : int, optional (default=5)
        Number of nearest neighbours to used to construct synthetic samples.

    n_jobs : int, optional (default=-1)
        Number of threads to run the algorithm when it is possible.

    Attributes
    ----------
    ratio : str or float
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    random_state : int or None
        Seed for random number generation.

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

    References
    ----------
    .. [1] He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. "ADASYN:
       Adaptive synthetic sampling approach for imbalanced learning," In IEEE
       International Joint Conference on Neural Networks (IEEE World Congress
       on Computational Intelligence), pp. 1322-1328, 2008.

    """

    def __init__(self, ratio='auto', random_state=None, verbose=True, k=5,
                 n_jobs=-1):
        """Initialize this object and its instance variables.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balance
            the dataset. Otherwise, the ratio is defined as the number
            of samples in the minority class over the the number of samples
            in the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Whether or not to print information about the processing.

        k : int, optional (default=5)
            Number of nearest neighbours to used to construct synthetic
            samples.

        n_jobs : int, optional (default=-1)
            Number of threads to run the algorithm when it is possible.

        Returns
        -------
        None

        """
        super(ADASYN, self).__init__(ratio=ratio,
                                     random_state=random_state,
                                     verbose=verbose)
        self.k = k
        self.n_jobs = n_jobs
        self.nearest_neighbour = NearestNeighbors(n_neighbors=self.k + 1,
                                                  n_jobs=self.n_jobs)

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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        # Call the parent function
        super(ADASYN, self).fit(X, y)

        return self

    def sample(self, X, y):
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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        # Call the parent function
        super(ADASYN, self).sample(X, y)

        # Keep the samples from the majority class
        X_resampled = X[y == self.maj_c_]
        y_resampled = y[y == self.maj_c_]

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
        if self.verbose:
            print('Finding the {} nearest neighbours...'.format(self.k))

        # Look for k-th nearest neighbours, excluding, of course, the
        # point itself.
        self.nearest_neighbour.fit(X)

        # Get the distance to the NN
        dist_nn, ind_nn = self.nearest_neighbour.kneighbors(X_min)

        # Compute the ratio of majority samples next to minority samples
        ratio_nn = np.sum(y[ind_nn[:, 1:]] == self.maj_c_, axis=1) / self.k
        # Normalize the ratio
        ratio_nn /= np.sum(ratio_nn)

        # Compute the number of sample to be generated
        num_samples_nn = np.round(ratio_nn * num_samples).astype(int)

        # For each minority samples
        for x_i, x_i_nn, num_sample_i in zip(X_min, ind_nn, num_samples_nn):
            # Fix the the seed
            np.random.seed(self.random_state)
            # Pick-up the neighbors wanted
            nn_zs = np.random.randint(1, high=self.k + 1, size=num_sample_i)

            # Create a new sample
            for nn_z in nn_zs:
                step = np.random.uniform()
                x_gen = x_i + step * (x_i - X[x_i_nn[nn_z], :])
                X_resampled = np.vstack((X_resampled, x_gen))
                y_resampled = np.hstack((y_resampled, self.min_c_))

        if self.verbose:
            print("Over-sampling performed: {}".format(Counter(y_resampled)))

        return X_resampled, y_resampled
