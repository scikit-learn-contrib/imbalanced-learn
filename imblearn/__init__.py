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
exceptions
    Module including custom warnings and error clases used across
    imbalanced-learn.
metrics
    Module which provides metrics to quantified the classification performance
    with imbalanced dataset.
over_sampling
    Module which provides methods to under-sample a dataset.
under-sampling
    Module which provides methods to over-sample a dataset.
utils
    Module including various utilities.
pipeline
    Module which allowing to create pipeline with scikit-learn estimators.
"""

from ._version import __version__

# list all submodules available in imblearn and version
__all__ = [
    'combine', 'ensemble', 'exceptions', 'metrics', 'over_sampling',
    'under_sampling', 'utils', 'pipeline', '__version__'
]
