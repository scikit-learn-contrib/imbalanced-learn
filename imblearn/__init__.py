"""Toolbox for imbalanced dataset in machine learning.

``imbalanced-learn`` is a set of python methods to deal with imbalanced
datset in machine learning and pattern recognition.

Subpackages
-----------
combine
    Module which provides methods based on over-sampling and under-sampling.
ensemble
    Module which provides methods generating an ensemble of
    under-sampled subsets.
metrics
    Module which provides metrics to quantified the classification performance
    with imbalanced dataset.
over_sampling
    Module which provides methods to under-sample a dataset.
under-sampling
    Module which provides methods to over-sample a dataset.
pipeline
    Module which allowing to create pipeline with scikit-learn estimators.
"""

from .version import _check_module_dependencies, __version__

_check_module_dependencies()

# Boolean controlling whether the joblib caches should be
# flushed if the version of certain modules changes (eg nibabel, as it
# does not respect the backward compatibility in some of its internal
# structures
# This  is used in nilearn._utils.cache_mixin
CHECK_CACHE_VERSION = True

# list all submodules available in imblearn and version
__all__ = [
    'combine', 'ensemble', 'metrics', 'over_sampling', 'under_sampling',
    'pipeline', '__version__'
]
