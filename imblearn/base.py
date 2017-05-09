"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import logging
import warnings
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.utils import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.validation import check_is_fitted


class SamplerMixin(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Mixin class for samplers with abstract method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = 'sampler'

    def _validate_size_ngh_deprecation(self):
        "Private function to warn about the deprecation about size_ngh."

        # Announce deprecation if necessary
        if self.size_ngh is not None:
            warnings.warn('`size_ngh` will be replaced in version 0.4. Use'
                          ' `n_neighbors` instead.', DeprecationWarning)
            self.n_neighbors = self.size_ngh

    def _validate_k_deprecation(self):
        """Private function to warn about deprecation of k in ADASYN"""
        if self.k is not None:
            warnings.warn('`k` will be replaced in version 0.4. Use'
                          ' `n_neighbors` instead.', DeprecationWarning)
            self.n_neighbors = self.k

    def _validate_k_m_deprecation(self):
        """Private function to warn about deprecation of k in ADASYN"""
        if self.k is not None:
            warnings.warn('`k` will be replaced in version 0.4. Use'
                          ' `k_neighbors` instead.', DeprecationWarning)
            self.k_neighbors = self.k

        if self.m is not None:
            warnings.warn('`m` will be replaced in version 0.4. Use'
                          ' `m_neighbors` instead.', DeprecationWarning)
            self.m_neighbors = self.m

    def _validate_deprecation(self):
        if hasattr(self, 'size_ngh'):
            self._validate_size_ngh_deprecation()
        elif hasattr(self, 'k') and not hasattr(self, 'm'):
            self._validate_k_deprecation()
        elif hasattr(self, 'k') and hasattr(self, 'm'):
            self._validate_k_m_deprecation()

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
        print('call fit')
        X, y = check_X_y(X, y)
        self._validate_deprecation()
        self.X_shape_ = X.shape

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
        check_is_fitted(self, 'ratio_')

        # Check if the size of the data is identical than at fitting
        if X.shape != self.X_shape_:
            raise RuntimeError('The data that you attempt to resample do not'
                               ' seem to be the one earlier fitted. Use the'
                               ' fitted data. Shape of data is {}, got {}'
                               ' instead.'.format(X.shape, self.X_shape_))

        self._validate_deprecation()

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


class BinarySamplerMixin(SamplerMixin):
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
        super(BinarySamplerMixin, self).fit(X, y)
        print('here')
        # Check that the target type is binary
        if not type_of_target(y) == 'binary':
            warnings.simplefilter('always', UserWarning)
            warnings.warn('The target type should be binary.')

        return self


class MultiClassSamplerMixin(SamplerMixin):
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
        super(MultiClassSamplerMixin, self).fit(X, y)
        print('here')
        # Check that the target type is either binary or multiclass
        if not (type_of_target(y) == 'binary' or
                type_of_target(y) == 'multiclass'):
            warnings.simplefilter('always', UserWarning)
            warnings.warn('The target type should be binary or multiclass.')

        return self
