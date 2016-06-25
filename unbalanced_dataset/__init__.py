"""Toolbox for unbalanced dataset in machine learning.

``UnbalancedDataset`` is a set of python methods to deal with unbalanced
datset in machine learning and pattern recognition.

Subpackages
-----------
combine
    Module which provides methods based on over-sampling and under-sampling.
ensemble
    Module which provides methods generating an ensemble of
    under-sampled subsets.
over_sampling
    Module which provides methods to under-sample a dataset.
under-sampling
    Module which provides methods to over-sample a dataset.
utils
    Module which provides helper methods.
"""

from .version import _check_module_dependencies

_check_module_dependencies()

# Boolean controlling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
CHECK_CACHE_VERSION = True

# list all submodules available in nilearn and version
__all__ = ['combine',
           'ensemble',
           'over_sampling',
           'under_sampling',
           'utils']
