"""Compatibility fixes for older version of python, numpy, scipy, and
scikit-learn.

If you add content to this file, please give the version of the package at
which the fix is no longer needed.
"""

import numpy as np
import scipy
import scipy.stats
import sklearn
from sklearn.utils.fixes import parse_version

sp_version = parse_version(scipy.__version__)
sklearn_version = parse_version(sklearn.__version__)


# TODO: Remove when SciPy 1.9 is the minimum supported version
def _mode(a, axis=0):
    if sp_version >= parse_version("1.9.0"):
        return scipy.stats.mode(a, axis=axis, keepdims=True)
    return scipy.stats.mode(a, axis=axis)


# TODO: Remove when scikit-learn 1.1 is the minimum supported version
if sklearn_version >= parse_version("1.1"):
    from sklearn.utils.validation import _is_arraylike_not_scalar
else:
    from sklearn.utils.validation import _is_arraylike

    def _is_arraylike_not_scalar(array):
        """Return True if array is array-like and not a scalar"""
        return _is_arraylike(array) and not np.isscalar(array)
