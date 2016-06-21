"""Base class for sampling"""

from __future__ import division
from __future__ import print_function

import numpy as np

from abc import ABCMeta, abstractmethod

from collections import Counter

from sklearn.utils import check_X_y

from six import string_types


class BaseSampler(object):
    """Basic class with abstact method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, ratio='auto', random_state=None, verbose=True):
        """Initialize this object and its instance variables.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. Otherwise, the ratio will corresponds to the number
            of samples in the minority class over the the number of samples
            in the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the processing

        Returns
        -------
        None

        """
        # The ratio correspond to the number of samples in the minority class
        # over the number of samples in the majority class. Thus, the ratio
        # cannot be greater than 1.0
        if isinstance(ratio, float):
            if ratio > 1:
                raise ValueError('Ration cannot be greater than one.')
            elif ratio <= 0:
                raise ValueError('Ratio cannot be negative.')
            else:
                self.ratio_ = ratio
        elif isinstance(ratio, string_types):
            if ratio == 'auto':
                self.ratio_ = ratio
            else:
                raise ValueError('Unknown string for the parameter ratio.')
        else:
            raise ValueError('Unknown parameter type for ratio.')

        self.rs_ = random_state
        self.verbose = verbose

        # Create the member variables regarding the classes statistics
        self.min_c_ = None
        self.maj_c_ = None
        self.stats_c_ = {}

    @abstractmethod
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

        if self.verbose:
            print("Determining classes statistics... ", end="")

        # Get all the unique elements in the target array
        uniques = np.unique(y)

        # Raise an error if there is only one class
        if uniques.size == 1:
            raise RuntimeError("Only one class detected, aborting...")

        # Create a dictionary containing the class statistics
        self.stats_c_ = Counter(y)

        # Find the minority and majority classes
        self.min_c_ = min(self.stats_c_, key=self.stats_c_.get)
        self.maj_c_ = max(self.stats_c_, key=self.stats_c_.get)

        if self.verbose:
            print('{} classes detected: {}'.format(uniques.size,
                                                   self.stats_c_))

        # Check if the ratio provided at initialisation make sense
        if isinstance(self.ratio_, float):
            if self.ratio_ < (self.stats_c_[self.min_c_] /
                              self.stats_c_[self.maj_c_]):
                raise RuntimeError('The ratio requested at initialisation'
                                   ' should be greater or equal than the'
                                   ' balancing ratio of the current data.')

        return self

    @abstractmethod
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

        """

        # Check that the data have been fitted
        if not self.stats_c_:
            raise RuntimeError('You need to fit the data, first!!!')

        return self

    def fit_transform(self, X, y):
        """Fit the statistics and resample the data directly.

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

        return self.fit(X, y).transform(X, y)
