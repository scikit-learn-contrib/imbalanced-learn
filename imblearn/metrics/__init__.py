"""
The :mod:`imblearn.metrics` module includes score functions, performance
metrics and pairwise metrics and distance computations.
"""

from .classification import sensitivity_specificity_support
from .classification import sensitivity_score
from .classification import specificity_score

__all__ = [
    'sensitivity_specificity_support',
    'sensitivity_score',
    'specificity_score'
]
