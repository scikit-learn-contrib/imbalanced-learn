"""Class to perform random under-sampling."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.utils import check_X_y

from .under_sampler import UnderSampler


class RandomUnderSampler(UnderSampler):
    """Class to perform random under-sampling.

    Object to under sample the majority class(es) by randomly picking samples
    with or without replacement.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balanced
        the dataset. Otherwise, the ratio will corresponds to the number
        of samples in the minority class over the the number of samples
        in the majority class.

    return_indices : bool, optional (default=True)
        Either to return or not the indices which will be selected from
        the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Boolean to either or not print information about the processing

    n_jobs : int, optional (default=1)
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

    Notes
    -----
    This class supports multi-class.

    """

    def __init__(self, ratio='auto', return_indices=False, random_state=None,
                 verbose=True, replacement=True):
        """Initialse the random under sampler object.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. Otherwise, the ratio will corresponds to the number
            of samples in the minority class over the the number of samples
            in the majority class.

        return_indices : bool, optional (default=True)
            Either to return or not the indices which will be selected from
            the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the processing

        n_jobs : int, optional (default=1)
            The number of thread to open when it is possible.

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

        super(RandomUnderSampler, self).fit(X, y)

        return self

    def transform(self, X, y):
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

        super(RandomUnderSampler, self).transform(X, y)

        # Compute the number of cluster needed
        if self.ratio_ == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio_)

        # All the minority class samples will be preserved
        X_resampled = X[y == self.min_c_]
        y_resampled = y[y == self.min_c_]

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(y == self.min_c_)[0]

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Pick some elements at random
            np.random.seed(self.rs_)
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

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
