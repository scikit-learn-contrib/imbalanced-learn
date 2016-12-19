"""
The :mod:`imblearn.metrics` module includes score functions, performance
metrics and pairwise metrics and distance computations.
"""

from .classification import sensitivity_specificity_support

__all__ = [
    'sensitivity_specificity_support'
]
