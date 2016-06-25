"""Class to perform random under-sampling."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.utils import check_X_y

from .under_sampler import UnderSampler


class RandomUnderSampler(UnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        from the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Whether or not to print information about the processing.

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible.

    Attributes
    ----------
    ratio : str or float
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    random state : int or None, optional (default=None)
        Seed for random number generation.

    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary containing the number of occurences of each class.

    Notes
    -----
    This class supports multi-class.

    """

    def __init__(self, ratio='auto', return_indices=False, random_state=None,
                 verbose=True, replacement=True):
        """Initialse the random under-sampler object.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balance
            the dataset. Otherwise, the ratio is defined as the number
            of samples in the minority class over the the number of samples
            in the majority class.

        return_indices : bool, optional (default=False)
            Whether or not to return the indices of the samples randomly
            selected from the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Whether or not to print information about the processing

        n_jobs : int, optional (default=-1)
            The number of threads to open if possible.

        Returns
        -------
        None

        """
        super(RandomUnderSampler, self).__init__(ratio=ratio,
                                                 return_indices=return_indices,
                                                 random_state=random_state,
                                                 verbose=verbose)

        self.replacement = replacement

    def fit(self, X, y):
        """Find the class statistics before performing sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(RandomUnderSampler, self).fit(X, y)

        return self

    def sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, an array will be returned
            containing a boolean for each sample to represent whether
            that sample was selected or not.

        """
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(RandomUnderSampler, self).sample(X, y)

        # Compute the number of clusters needed
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio)

        # All the minority class samples will be preserved
        X_resampled = X[y == self.min_c_]
        y_resampled = y[y == self.min_c_]

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(y == self.min_c_)[0]

        # Loop over the other classes under-picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Pick some elements at random
            np.random.seed(self.random_state)
            indx = range(np.count_nonzero(y == key))
            indx = np.random.choice(indx, size=num_samples,
                                    replace=self.replacement)

            # If we need to offer support for the indices selected
            if self.return_indices:
                idx_tmp = np.nonzero(y == key)[0][indx]
                idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

            # Concatenate to the minority class
            X_resampled = np.concatenate((X_resampled, X[y == key][indx]),
                                         axis=0)
            y_resampled = np.concatenate((y_resampled, y[y == key][indx]),
                                         axis=0)

        if self.verbose:
            print("Under-sampling performed: {}".format(Counter(y_resampled)))

        # Check if the indices of the samples selected should be returned as
        # well
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
