"""Miscellaneous samplers objects."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import logging

from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from .base import SamplerMixin
from .utils import check_target_type, hash_X_y


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

    """

    def __init__(self, func=None, accept_sparse=True, kw_args=None,
                 random_state=None):
        self.func = func
        self.accept_sparse = accept_sparse
        self.kw_args = kw_args
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def _check_X_y(self, X, y):
        if self.accept_sparse:
            X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        else:
            X, y = check_X_y(X, y, accept_sparse=False)
        y = check_target_type(y)

        return X, y

    def fit(self, X, y):
        print(self.accept_sparse)
        X, y = self._check_X_y(X, y)
        self.X_hash_, self.y_hash_ = hash_X_y(X, y)
        # when using a sampler, ratio_ is supposed to exist after fit
        self.ratio_ = 'is_fitted'

        return self

    def _sample(self, X, y, func, kw_args):
        X, y = self._check_X_y(X, y)
        check_is_fitted(self, 'ratio_')
        X_hash, y_hash = hash_X_y(X, y)
        if self.X_hash_ != X_hash or self.y_hash_ != y_hash:
            raise RuntimeError("X and y need to be same array earlier fitted.")

        if func is None:
            func = _identity

        return func(X, y, **(kw_args if kw_args else {}))

    def sample(self, X, y):
        return self._sample(X, y, func=self.func, kw_args=self.kw_args)
