﻿"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import logging
import warnings
from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from .utils import check_sampling_strategy, check_target_type, hash_X_y
from .utils.deprecation import deprecate_parameter


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
        # Check the consistency of X and y
        X, y, binarize_y = self._check_X_y(X, y)

        check_is_fitted(self, 'sampling_strategy_')
        self._check_X_y_hash(X, y)

        output = self._sample(X, y)

        if binarize_y:
            y_sampled = label_binarize(output[1], np.unique(y))
            if len(output) == 2:
                return output[0], y_sampled
            else:
                return output[0], y_sampled, output[2]
        else:
            return output

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
        """Base method defined in each sampler to defined the sampling
        strategy.

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

    def __init__(self, sampling_strategy='auto', ratio=None):
        self.sampling_strategy = sampling_strategy
        # FIXME: remove in 0.6
        self.ratio = ratio
        self.logger = logging.getLogger(self.__module__)

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        return X, y, binarize_y

    def _check_X_y_hash(self, X, y):
        """Private function to check that the X and y in fitting are the same
        than in sampling."""
        X_hash, y_hash = hash_X_y(X, y)
        if self.X_hash_ != X_hash or self.y_hash_ != y_hash:
            raise RuntimeError("X and y need to be same array earlier fitted.")

    @property
    def ratio_(self):
        # FIXME: remove in 0.6
        warnings.warn("'ratio' and 'ratio_' are deprecated. Use "
                      "'sampling_strategy' and 'sampling_strategy_' instead.",
                      DeprecationWarning)
        return self.sampling_strategy_

    def _deprecate_ratio(self):
        # both ratio and sampling_strategy should not be set
        if self.ratio is not None:
            deprecate_parameter(self, '0.4', 'ratio', 'sampling_strategy')
            self.sampling_strategy = self.ratio

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
        self._deprecate_ratio()
        X, y, _ = self._check_X_y(X, y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        # _sampling_type is defined in the children base class
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type)

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
    >>> def func(X, y, sampling_strategy, random_state):
    ...   return RandomUnderSampler(sampling_strategy=sampling_strategy,
    ...                             random_state=random_state).fit_sample(X, y)
    >>> sampler = FunctionSampler(func=func,
    ...                           kw_args={'sampling_strategy': 'auto',
    ...                                    'random_state': 0})
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

    def fit(self, X, y):
        y = check_target_type(y)
        X, y = check_X_y(
            X,
            y,
            accept_sparse=['csr', 'csc'] if self.accept_sparse else False)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        # when using a sampler, ratio_ is supposed to exist after fit
        self.sampling_strategy_ = 'is_fitted'

        return self

    @property
    def ratio_(self):
        # FIXME: remove in 0.6
        warnings.warn("'ratio' and 'ratio_' are deprecated. Use "
                      "'sampling_strategy' and 'sampling_strategy_' instead.",
                      DeprecationWarning)
        return self.sampling_strategy_

    def _sample(self, X, y, func=None, kw_args=None):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(
            X,
            y,
            accept_sparse=['csr', 'csc'] if self.accept_sparse else False)
        check_is_fitted(self, 'sampling_strategy_')
        X_hash, y_hash = hash_X_y(X, y)
        if self.X_hash_ != X_hash or self.y_hash_ != y_hash:
            raise RuntimeError("X and y need to be same array earlier fitted.")

        if func is None:
            func = _identity

        output = func(X, y, **(kw_args if self.kw_args else {}))

        if binarize_y:
            y_sampled = label_binarize(output[1], np.unique(y))
            if len(output) == 2:
                return output[0], y_sampled
            else:
                return output[0], y_sampled, output[2]
        else:
            return output

    def sample(self, X, y):
        return self._sample(X, y, func=self.func, kw_args=self.kw_args)
