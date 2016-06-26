"""Class to perform random over-sampling."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.utils import check_X_y

from .over_sampler import OverSampler


class RandomOverSampler(OverSampler):
    """Class to perform random over-sampling.

    Object to over-sample the minority class(es) by picking samples at random
    with replacement.

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
    Supports multiple classes.
    """

    def __init__(self, ratio='auto', random_state=None, verbose=True):
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

        Returns
        -------
        None

        """
        super(RandomOverSampler, self).__init__(ratio=ratio,
                                                random_state=random_state,
                                                verbose=verbose)

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
        super(RandomOverSampler, self).fit(X, y)

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
        super(RandomOverSampler, self).sample(X, y)

        # Keep the samples from the majority class
        X_resampled = X[y == self.maj_c_]
        y_resampled = y[y == self.maj_c_]

        # Loop over the other classes over picking at random
        for key in self.stats_c_.keys():

            # If this is the majority class, skip it
            if key == self.maj_c_:
                continue

            # Define the number of sample to create
            if self.ratio == 'auto':
                num_samples = int(self.stats_c_[self.maj_c_] -
                                  self.stats_c_[key])
            else:
                num_samples = int((self.ratio * self.stats_c_[self.maj_c_]) -
                                  self.stats_c_[key])

            # Pick some elements at random
            np.random.seed(self.random_state)
            indx = np.random.randint(low=0, high=self.stats_c_[key],
                                     size=num_samples)

            # Concatenate to the majority class
            X_resampled = np.concatenate((X_resampled,
                                          X[y == key],
                                          X[y == key][indx]),
                                         axis=0)

            y_resampled = np.concatenate((y_resampled,
                                          y[y == key],
                                          y[y == key][indx]), axis=0)

        if self.verbose:
            print("Over-sampling performed: {}".format(Counter(y_resampled)))

        return X_resampled, y_resampled
