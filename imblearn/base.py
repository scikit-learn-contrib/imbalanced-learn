﻿"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import logging
from abc import ABCMeta, abstractmethod

import numpy as np

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

        check_is_fitted(self, 'ratio_')
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], multi_output=True)
        if self.target_encoder_ is not None:
            y = self.target_encoder_.inverse_transform(y)
        X_hash, y_hash = hash_X_y(X, y)
        if self.X_hash_ != X_hash or self.y_hash_ != y_hash:
            raise RuntimeError("X and y need to be same array earlier fitted.")

        result = self._sample(X, y)

        if not getattr(self, 'return_indices', False):
            X_res, y_res = result
        else:
            X_res, y_res, indices_res = result

        if self.target_encoder_ is not None:
            # find the case that we have ensemble
            if y_res.ndim == 2:
                y_res = np.hstack([self.target_encoder_.transform(y_res_subset)
                                   for y_res_subset in y_res])
            else:
                y_res = self.target_encoder_.transform(y_res)

        if not getattr(self, 'return_indices', False):
            return X_res, y_res
        else:
            return X_res, y_res, indices_res

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
        logger = logging.getLogger(self.__module__)
        self.__dict__.update(dict)
        self.logger = logger


class BaseSampler(SamplerMixin):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, ratio='auto'):
        self.ratio = ratio
        self.logger = logging.getLogger(self.__module__)

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
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], multi_output=True)
        y = check_target_type(y, self)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        # self.sampling_type is already checked in check_ratio
        self.ratio_ = check_ratio(self.ratio, y, self._sampling_type)

        return self


def _identity(X, y):
    return X, y


class FunctionSampler(SamplerMixin):
    """Construct a sampler from calling an arbitrary callable.

    Read more in the :ref:`User Guide <function_sampler>`.

    Parameters
    ----------
    func : callable or None,
        The callable to use for the transformation. This will be passed the
        same arguments as transform, with args and kwargs forwarded. If func is
        None, then func will be the identity function.

    accept_sparse : bool, optional (default=True)
        Whether sparse input are supported. By default, sparse inputs are
        supported.

    kw_args : dict, optional (default=None)
        The keyword argument expected by ``func``.

    Notes
    -----

    See
    :ref:`sphx_glr_auto_examples_plot_outlier_rejections.py`

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_classification
    >>> from imblearn import FunctionSampler
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

    We can create to select only the first ten samples for instance.

    >>> def func(X, y):
    ...   return X[:10], y[:10]
    >>> sampler = FunctionSampler(func=func)
    >>> X_res, y_res = sampler.fit_sample(X, y)
    >>> np.all(X_res == X[:10])
    True
    >>> np.all(y_res == y[:10])
    True

    We can also create a specific function which take some arguments.

    >>> from collections import Counter
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> def func(X, y, ratio, random_state):
    ...   return RandomUnderSampler(ratio=ratio,
    ...                             random_state=random_state).fit_sample(X, y)
    >>> sampler = FunctionSampler(func=func,
    ...                           kw_args={'ratio': 'auto', 'random_state': 0})
    >>> X_res, y_res = sampler.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(
    ...     sorted(Counter(y_res).items())))
    Resampled dataset shape [(0, 100), (1, 100)]

    """

    def __init__(self, func=None, accept_sparse=True, kw_args=None):
        self.func = func
        self.accept_sparse = accept_sparse
        self.kw_args = kw_args
        self.logger = logging.getLogger(__name__)

    def _check_X_y(self, X, y):
        if self.accept_sparse:
            X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        else:
            X, y = check_X_y(X, y, accept_sparse=False)
        y = check_target_type(y)

        return X, y

    def fit(self, X, y):
        X, y = self._check_X_y(X, y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        # when using a sampler, ratio_ is supposed to exist after fit
        self.ratio_ = 'is_fitted'

        return self

    def _sample(self, X, y, func=None, kw_args=None):
        X, y = self._check_X_y(X, y)
        check_is_fitted(self, 'ratio_')
        X_hash, y_hash = hash_X_y(X, y)
        if self.X_hash_ != X_hash or self.y_hash_ != y_hash:
            raise RuntimeError("X and y need to be same array earlier fitted.")

        if func is None:
            func = _identity

        return func(X, y, **(kw_args if self.kw_args else {}))

    def sample(self, X, y):
        return self._sample(X, y, func=self.func, kw_args=self.kw_args)
