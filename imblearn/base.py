﻿"""Base class for sampling"""

from __future__ import division, print_function

import logging
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter

import numpy as np
from six import string_types
from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target

TARGET_STR = ('minority', 'not minority', 'majority', 'all')


class SamplerMixin(six.with_metaclass(ABCMeta, BaseEstimator)):

    """Mixin class for samplers with abstact method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = 'sampler'

    def __init__(self, ratio='auto', target_classes='not minority'):
        """Initialize this object and its instance variables.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balanced
            the dataset. Otherwise, the ratio will corresponds to the number
            of samples in the minority class over the the number of samples
            in the majority class.

        target_classes : str or tuple target, optional (default='not minority')
            A string or a tuple of target to specify which class to consider
            to apply the balancing.

        Returns
        -------
        None

        """

        self.ratio = ratio
        self.target_classes = target_classes
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

        # # Raise an error if there is only one class
        # if uniques.size == 1:
        #     raise RuntimeError("Only one class detected, aborting...")
        # Raise a warning for the moment to be compatible with BaseEstimator
        self.logger.debug('The number of classes is %s', np.unique(y).size)
        self.logger.debug('Shall we raise a warning: %s',
                          np.unique(y).size == 1)
        if np.unique(y).size == 1:
            warnings.simplefilter('always', UserWarning)
            warnings.warn('Only one class detected, something will get wrong')
            self.logger.debug('The warning should has been raised.')

        # Store the size of X to check at sampling time if we have the
        # same data
        self.X_shape_ = X.shape

        # Create a dictionary containing the class statistics
        self.stats_c_ = Counter(y)

        # Find the minority and majority classes
        self.min_c_ = min(self.stats_c_, key=self.stats_c_.get)
        self.maj_c_ = max(self.stats_c_, key=self.stats_c_.get)

        self.logger.info('%s classes detected: %s', np.unique(y).size,
                         self.stats_c_)

        if hasattr(self, 'target_classes'):
            self.target_classes_ = self._validate_target_classes()
            self.logger.debug('The target classes have been set to %s',
                              self.target_classes_)

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
        """Private function to check if the ratio as a good value"""
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

    def _validate_target_classes(self):
        """Private method to check whether the parameters pass are in
        accordance with the class statistics."""

        if isinstance(self.target_classes, string_types):
            if self.target_classes not in TARGET_STR:
                raise ValueError('The target classes parameters specified is'
                                 ' unknown.')
            else:
                # Convert the string to a tuple with the targeted classes
                if self.target_classes == 'minority':
                    return tuple([self.min_c_])
                elif self.target_classes == 'majority':
                    return tuple([self.maj_c_])
                elif self.target_classes == 'all':
                    return tuple(self.stats_c_.keys())
                elif self.target_classes == 'not minority':
                    return tuple([val for val in
                                                 self.stats_c_.keys()
                                                 if val != self.min_c_])
        elif isinstance(self.target_classes, tuple):
            # Create a tuple with the different class present in the statistics
            classes_tuple = tuple(self.stats_c_.keys())
            for val in self.target_classes:
                if val not in classes_tuple:
                    raise ValueError('The class specified in class target is'
                                     ' unknown and no present in the data'
                                     ' provided')
            return self.target_classes
        else:
            raise ValueError('Unknown type for the target classes parameters.')

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


class BaseBinarySampler(six.with_metaclass(ABCMeta, SamplerMixin)):
    """Base class for all binary class sampler.

    Warning: This class should not be used directly. Use derived classes
    instead.

    """

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

        super(BaseBinarySampler, self).fit(X, y)

        # Check that the target type is binary
        if not type_of_target(y) == 'binary':
            warnings.simplefilter('always', UserWarning)
            warnings.warn('The target type should be binary.')

        return self


class BaseMulticlassSampler(six.with_metaclass(ABCMeta, SamplerMixin)):
    """Base class for all multiclass sampler.

    Warning: This class should not be used directly. Use derived classes
    instead.

    """

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

        super(BaseMulticlassSampler, self).fit(X, y)

        # Check that the target type is either binary or multiclass
        if not (type_of_target(y) == 'binary' or
                type_of_target(y) == 'multiclass'):
            warnings.simplefilter('always', UserWarning)
            warnings.warn('The target type should be binary or multiclass.')

        return self
