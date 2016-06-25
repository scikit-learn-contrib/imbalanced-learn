"""Class to perform under-sampling by generating centroids based on
clustering."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.cluster import KMeans
from sklearn.utils import check_X_y

from .under_sampler import UnderSampler


class ClusterCentroids(UnderSampler):
    """Perform under-sampling by generating centroids based on
    clustering methods.

    Experimental method that under samples the majority class by replacing a
    cluster of majority samples by the cluster centroid of a KMeans algorithm.
    This algorithm keeps N majority samples by fitting the KMeans algorithm
    with N cluster to the majority class and using the coordinates of the N
    cluster centroids as the new majority samples.

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

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible.

    **kwargs : keywords
        Parameter to use for the KMeans object.

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

    Notes
    -----
    This class support multi-class.

    """

    def __init__(self, ratio='auto', random_state=None, verbose=True,
                 n_jobs=-1, **kwargs):
        """Initialisation of clustering centroids object.

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

        n_jobs : int, optional (default=-1)
            The number of threads to open if possible.

        **kwargs : keywords
            Parameter to use for the KMeans object.

        Returns
        -------
        None

        """
        super(ClusterCentroids, self).__init__(ratio=ratio,
                                               return_indices=False,
                                               random_state=random_state,
                                               verbose=verbose)
        self.n_jobs = n_jobs
        self.kwargs = kwargs

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

        super(ClusterCentroids, self).fit(X, y)

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

        super(ClusterCentroids, self).sample(X, y)

        # Compute the number of cluster needed
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio)

        # Create the clustering object
        kmeans = KMeans(n_clusters=num_samples, random_state=self.random_state)
        kmeans.set_params(**self.kwargs)

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it.
            if key == self.min_c_:
                continue

            # Find the centroids via k-means
            kmeans.fit(X[y == key])
            centroids = kmeans.cluster_centers_

            # Concatenate to the minority class
            X_resampled = np.concatenate((X_resampled, centroids), axis=0)
            y_resampled = np.concatenate((y_resampled, np.array([key] *
                                                                num_samples)),
                                         axis=0)

        if self.verbose:
            print("Under-sampling performed: {}".format(Counter(y_resampled)))

        return X_resampled, y_resampled
