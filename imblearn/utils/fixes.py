"""Compatibility fixes for older version of python, numpy, scipy, and
scikit-learn.

If you add content to this file, please give the version of the package at
which the fix is no longer needed.
"""

import scipy
import scipy.stats
from sklearn.utils.fixes import parse_version

sp_version = parse_version(scipy.__version__)


# TODO: Remove when SciPy 1.9 is the minimum supported version
def _mode(a, axis=0):
    if sp_version >= parse_version("1.9.0"):
        return scipy.stats.mode(a, axis=axis, keepdims=True)
    return scipy.stats.mode(a, axis=axis)
