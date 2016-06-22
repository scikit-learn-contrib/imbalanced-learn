"""Class to perform under-sampling by removing Tomek's links."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from .under_sampler import UnderSampler


class TomekLinks(UnderSampler):
    """Class to perform under-sampling by removing Tomek's links.

    Parameters
    ----------
    return_indices : bool, optional (default=False)
        Either to return or not the indices which will be selected from
        the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Boolean to either or not print information about the processing

    n_jobs : int, optional (default=-1)
        The number of thread to open when it is possible.

    Attributes
    ----------
    ratio_ : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balanced
        the dataset. Otherwise, the ratio will corresponds to the number
        of samples in the minority class over the the number of samples
        in the majority class.

    rs_ : int or None, optional (default=None)
        Seed for random number generation.

    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    """

    def __init__(self, return_indices=False, random_state=None, verbose=True,
                 n_jobs=-1):
        """Initialisation of Tomek's links object.

        Parameters
        ----------
        return_indices : bool, optional (default=False)
            Either to return or not the indices which will be selected from
            the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the processing

        n_jobs : int, optional (default=-1)
            The number of thread to open when it is possible.

        Returns
        -------
        None

        """
        super(TomekLinks, self).__init__(return_indices=return_indices,
                                         random_state=random_state,
                                         verbose=verbose)
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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(TomekLinks, self).fit(X, y)

        return self

    @staticmethod
    def is_tomek(y, nn_index, class_type, verbose=True):
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

        if verbose:
            print("{} Tomek links found.".format(count))

        return links

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

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(TomekLinks, self).sample(X, y)

        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X)
        nns = nn.kneighbors(X, return_distance=False)[:, 1]

        # Send the information to is_tomek function to get boolean vector back
        if self.verbose:
            print("Looking for majority Tomek links...")
        links = self.is_tomek(y, nns, self.min_c_, self.verbose)

        if self.verbose:
            print("Under-sampling performed: {}".format(Counter(
                y[np.logical_not(links)])))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return (X[np.logical_not(links)], y[np.logical_not(links)],
                    np.nonzero(np.logical_not(links))[0])
        else:
            # Return data set without majority Tomek links.
            return X[np.logical_not(links)], y[np.logical_not(links)]
