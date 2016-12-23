"""Class to perform under-sampling by removing Tomek's links."""
from __future__ import division, print_function

from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors

from ..base import BaseBinarySampler


class TomekLinks(BaseBinarySampler):
    """Class to perform under-sampling by removing Tomek's links.

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

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

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
    This method is based on [1]_.

    It does not support multi-class sampling.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
    TomekLinks # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> tl = TomekLinks(random_state=42)
    >>> X_res, y_res = tl.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 897, 0: 100})

    References
    ----------
    .. [1] I. Tomek, "Two modifications of CNN," In Systems, Man, and
       Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 2010.

    """

    def __init__(self, return_indices=False, random_state=None, n_jobs=1):
        super(TomekLinks, self).__init__(random_state=random_state)
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

        # Initialize the boolean result as false, and also a counter
        links = np.zeros(len(y), dtype=bool)
        count = 0

        # Loop through each sample and looks whether it belongs to the minority
        # class. If it does, we don't consider it since we want to keep all
        # minority samples. If, however, it belongs to the majority sample we
        # look at its first neighbour. If its closest neighbour also has the
        # current sample as its closest neighbour, the two form a Tomek link.
        for ind, ele in enumerate(y):

            if ele == class_type:
                continue

            if y[nn_index[ind]] == class_type:

                # If they form a tomek link, put a True marker on this
                # sample, and increase counter by one.
                if nn_index[nn_index[ind]] == ind:
                    links[ind] = True
                    count += 1

        return links

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

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X)
        nns = nn.kneighbors(X, return_distance=False)[:, 1]

        # Send the information to is_tomek function to get boolean vector back
        self.logger.debug('Looking for majority Tomek links ...')
        links = self.is_tomek(y, nns, self.min_c_)

        self.logger.info('Under-sampling performed: %s',
                         Counter(y[np.logical_not(links)]))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return (X[np.logical_not(links)], y[np.logical_not(links)],
                    np.flatnonzero(np.logical_not(links)))
        else:
            # Return data set without majority Tomek links.
            return X[np.logical_not(links)], y[np.logical_not(links)]
