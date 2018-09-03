"""Base class for sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np

from sklearn.base import BaseEstimator
from sklearn.externals import six
from sklearn.preprocessing import label_binarize
from sklearn.utils import check_X_y

from .utils import check_sampling_strategy, check_target_type
from .utils.deprecation import deprecate_parameter


class SamplerMixin(six.with_metaclass(ABCMeta, BaseEstimator)):
    """Mixin class for samplers with abstract method.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _estimator_type = 'sampler'

    def fit(self, X, y):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data array.

        y : array-like, shape (n_samples,)
            Target array.

        Returns
        -------
        self : object
            Return the instance itself.

        """
        self._deprecate_ratio()
        X, y, _ = self._check_X_y(X, y)
        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type)
        return self

    def fit_resample(self, X, y):
        """Resample the dataset.

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
            The corresponding label of `X_resampled`.

        """
        self._deprecate_ratio()

        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type)

        output = self._fit_resample(X, y)

        if binarize_y:
            y_sampled = label_binarize(output[1], np.unique(y))
            if len(output) == 2:
                return output[0], y_sampled
            return output[0], y_sampled, output[2]
        return output

    #  define an alias for back-compatibility
    fit_sample = fit_resample

    @abstractmethod
    def _fit_resample(self, X, y):
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


class BaseSampler(SamplerMixin):
    """Base class for sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, sampling_strategy='auto', ratio=None):
        self.sampling_strategy = sampling_strategy
        # FIXME: remove in 0.6
        self.ratio = ratio

    @staticmethod
    def _check_X_y(X, y):
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        return X, y, binarize_y

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


def _identity(X, y):
    return X, y


class FunctionSampler(BaseSampler):
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
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> np.all(X_res == X[:10])
    True
    >>> np.all(y_res == y[:10])
    True

    We can also create a specific function which take some arguments.

    >>> from collections import Counter
    >>> from imblearn.under_sampling import RandomUnderSampler
    >>> def func(X, y, sampling_strategy, random_state):
    ...   return RandomUnderSampler(
    ...       sampling_strategy=sampling_strategy,
    ...       random_state=random_state).fit_resample(X, y)
    >>> sampler = FunctionSampler(func=func,
    ...                           kw_args={'sampling_strategy': 'auto',
    ...                                    'random_state': 0})
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> print('Resampled dataset shape {}'.format(
    ...     sorted(Counter(y_res).items())))
    Resampled dataset shape [(0, 100), (1, 100)]

    """

    _sampling_type = 'bypass'

    def __init__(self, func=None, accept_sparse=True, kw_args=None):
        super(FunctionSampler, self).__init__()
        self.func = func
        self.accept_sparse = accept_sparse
        self.kw_args = kw_args

    def _fit_resample(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc']
                         if self.accept_sparse else False)
        func = _identity if self.func is None else self.func
        output = func(X, y, **(self.kw_args if self.kw_args else {}))
        return output
