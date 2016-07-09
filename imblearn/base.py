"""Base class for sampling"""

from __future__ import division
from __future__ import print_function

import warnings
import logging

import numpy as np

from abc import ABCMeta, abstractmethod

from collections import Counter

from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y
from sklearn.externals import six

from six import string_types


class SamplerMixin(six.with_metaclass(ABCMeta, BaseEstimator)):

    """Mixin class for samplers with abstact method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = "sampler"

    def __init__(self, ratio='auto'):
        """Initialize this object and its instance variables.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. Otherwise, the ratio will corresponds to the number
            of samples in the minority class over the the number of samples
            in the majority class.

        Returns
        -------
        None

        """

        self.ratio = ratio
        self.logger = logging.getLogger(__name__)

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

        self.min_c_ = None
        self.maj_c_ = None
        self.stats_c_ = {}
        self.X_shape_ = None

        if hasattr(self, 'ratio'):
            self._validate_ratio()

        self.logger.info('Compute classes statistics ...')

        # Get all the unique elements in the target array
        uniques = np.unique(y)

        # # Raise an error if there is only one class
        # if uniques.size == 1:
        #     raise RuntimeError("Only one class detected, aborting...")
        # Raise a warning for the moment to be compatible with BaseEstimator
        if uniques.size == 1:
            warnings.warn('Only one class detected, something will get wrong',
                          RuntimeWarning)

        # Store the size of X to check at sampling time if we have the
        # same data
        self.X_shape_ = X.shape

        # Create a dictionary containing the class statistics
        self.stats_c_ = Counter(y)

        # Find the minority and majority classes
        self.min_c_ = min(self.stats_c_, key=self.stats_c_.get)
        self.maj_c_ = max(self.stats_c_, key=self.stats_c_.get)

        self.logger.info('%s classes detected: %s', uniques.size,
                         self.stats_c_)

        # Check if the ratio provided at initialisation make sense
        if isinstance(self.ratio, float):
            if self.ratio < (self.stats_c_[self.min_c_] /
                             self.stats_c_[self.maj_c_]):
                raise RuntimeError('The ratio requested at initialisation'
                                   ' should be greater or equal than the'
                                   ' balancing ratio of the current data.')

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

        # Check that the data have been fitted
        if not hasattr(self, 'stats_c_'):
            raise RuntimeError('You need to fit the data, first!!!')

        # Check if the size of the data is identical than at fitting
        if X.shape != self.X_shape_:
            raise RuntimeError('The data that you attempt to resample do not'
                               ' seem to be the one earlier fitted. Use the'
                               ' fitted data.')

        if hasattr(self, 'ratio'):
            self._validate_ratio()

        return self._sample(X, y)

    def fit_sample(self, X, y):
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

        return self.fit(X, y).sample(X, y)

    def _validate_ratio(self):
        # The ratio correspond to the number of samples in the minority class
        # over the number of samples in the majority class. Thus, the ratio
        # cannot be greater than 1.0
        if isinstance(self.ratio, float):
            if self.ratio > 1:
                raise ValueError('Ration cannot be greater than one.')
            elif self.ratio <= 0:
                raise ValueError('Ratio cannot be negative.')

        elif isinstance(self.ratio, string_types):
            if self.ratio != 'auto':
                raise ValueError('Unknown string for the parameter ratio.')
        else:
            raise ValueError('Unknown parameter type for ratio.')

    @abstractmethod
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
        pass

    def __getstate__(self):
        """Prevent logger from being pickled."""
        object_dictionary = self.__dict__.copy()
        del object_dictionary['logger']
        return object_dictionary

    def __setstate__(self, dict):
        """Re-open the logger."""
        logger = logging.getLogger(__name__)
        self.__dict__.update(dict)
        self.logger = logger
