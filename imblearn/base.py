"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import logging
from abc import ABCMeta, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from .utils import check_ratio, check_target_type, hash_X_y


class SamplerMixin(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Mixin class for samplers with abstract method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = 'sampler'

    def _check_X_y(self, X, y):
        """Private function to check that the X and y in fitting are the same
        than in sampling."""
        X_hash, y_hash = hash_X_y(X, y)
        if self.X_hash_ != X_hash or self.y_hash_ != y_hash:
            raise RuntimeError("X and y need to be same array earlier fitted.")

    def sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape \
(n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        """

        # Check the consistency of X and y
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])

        check_is_fitted(self, 'ratio_')
        self._check_X_y(X, y)

        return self._sample(X, y)

    def fit_sample(self, X, y):
        """Fit the statistics and resample the data directly.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {array-like, sparse matrix}, shape \
(n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like, shape (n_samples_new,)
            The corresponding label of `X_resampled`

        """

        return self.fit(X, y).sample(X, y)

    @abstractmethod
    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape \
(n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
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


class BaseSampler(SamplerMixin):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, ratio='auto', random_state=None, sampling_type=None):
        self.ratio = ratio
        self.random_state = random_state
        self.sampling_type = sampling_type
        self.logger = logging.getLogger(__name__)

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        y = check_target_type(y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        # self.sampling_type is already checked in check_ratio
        self.ratio_ = check_ratio(self.ratio, y, self._sampling_type)

        return self
