"""Class to perform under-sampling based on one-sided selection method."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state

from ..base import SamplerMixin
from .tomek_links import TomekLinks


class OneSidedSelection(SamplerMixin):
    """Class to perform under-sampling based on one-sided selection method.

    Parameters
    ----------
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    size_ngh : int, optional (default=1)
        Size of the neighbourhood to consider to compute the average
        distance to the minority point samples.

    n_seeds_S : int, optional (default=1)
        Number of samples to extract in order to build the set S.

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible.

    **kwargs : keywords
        Parameter to use for the Neareast Neighbours object.

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
    The method is based on [1]_.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import OneSidedSelection
    >>> X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
    ...                            n_informative=3, n_redundant=1, flip_y=0,
    ...                            n_features=20, n_clusters_per_class=1,
    ...                            n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> oss = OneSidedSelection(random_state=42)
    >>> X_res, y_res = oss.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 595, 0: 100})

    References
    ----------
    .. [1] M. Kubat, S. Matwin, "Addressing the curse of imbalanced training
       sets: one-sided selection," In ICML, vol. 97, pp. 179-186, 1997.

    """

    def __init__(self, return_indices=False, random_state=None,
                 size_ngh=1, n_seeds_S=1, n_jobs=-1, **kwargs):
        super(OneSidedSelection, self).__init__()
        self.return_indices = return_indices
        self.random_state = random_state
        self.size_ngh = size_ngh
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs
        self.kwargs = kwargs

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

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """

        random_state = check_random_state(self.random_state)

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(y == self.min_c_)[0]

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Randomly get one sample from the majority class
            # Generate the index to select
            idx_maj_sample = random_state.randint(low=0,
                                                  high=self.stats_c_[key],
                                                  size=self.n_seeds_S)
            maj_sample = X[y == key][idx_maj_sample]

            # Create the set C
            C_x = np.append(X_min,
                            maj_sample,
                            axis=0)
            C_y = np.append(y_min,
                            [key] * self.n_seeds_S)

            # Create the set S
            S_x = X[y == key]
            S_y = y[y == key]

            # Remove the seed from S since that it will be added anyway
            S_x = np.delete(S_x, idx_maj_sample, axis=0)
            S_y = np.delete(S_y, idx_maj_sample, axis=0)

            # Create a k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=self.size_ngh,
                                       n_jobs=self.n_jobs,
                                       **self.kwargs)

            # Fit C into the knn
            knn.fit(C_x, C_y)

            # Classify on S
            pred_S_y = knn.predict(S_x)

            # Find the misclassified S_y
            sel_x = np.squeeze(S_x[np.nonzero(pred_S_y != S_y), :])
            sel_y = S_y[np.nonzero(pred_S_y != S_y)]

            # If we need to offer support for the indices selected
            # We concatenate the misclassified samples with the seed and the
            # minority samples
            if self.return_indices:
                idx_tmp = np.nonzero(y == key)[0][np.nonzero(pred_S_y != S_y)]
                idx_under = np.concatenate((idx_under,
                                            idx_maj_sample,
                                            idx_tmp),
                                           axis=0)

            X_resampled = np.concatenate((X_resampled,
                                          maj_sample,
                                          sel_x),
                                         axis=0)
            y_resampled = np.concatenate((y_resampled,
                                          [key] * self.n_seeds_S,
                                          sel_y),
                                         axis=0)

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X_resampled)
        nns = nn.kneighbors(X_resampled, return_distance=False)[:, 1]

        # Send the information to is_tomek function to get boolean vector back
        self.logger.debug('Looking for majority Tomek links ...')
        links = TomekLinks.is_tomek(y_resampled, nns, self.min_c_)

        self.logger.info('Under-sampling performed: %s', Counter(
            y_resampled[np.logical_not(links)]))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return (X_resampled[np.logical_not(links)],
                    y_resampled[np.logical_not(links)],
                    idx_under[np.logical_not(links)])
        else:
            # Return data set without majority Tomek links.
            return (X_resampled[np.logical_not(links)],
                    y_resampled[np.logical_not(links)])
