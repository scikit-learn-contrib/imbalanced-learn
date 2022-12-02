"""Compatibility fixes for older version of python, numpy and scipy
If you add content to this file, please give the version of the package
at which the fix is no longer needed.

Backdated from scikit-learn.
"""
# Authors: Emmanuelle Gouillart <emmanuelle.gouillart@normalesup.org>
#          Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Fabian Pedregosa <fpedregosa@acm.org>
#          Lars Buitinck
#
# License: BSD 3 clause

from sklearn.externals._packaging.version import parse as parse_version
import scipy
import scipy.stats


sp_version = parse_version(scipy.__version__)


# TODO: Remove when SciPy 1.9 is the minimum supported version
def _mode(a, axis=0):
    if sp_version >= parse_version("1.9.0"):
        return scipy.stats.mode(a, axis=axis, keepdims=True)
    return scipy.stats.mode(a, axis=axis)
