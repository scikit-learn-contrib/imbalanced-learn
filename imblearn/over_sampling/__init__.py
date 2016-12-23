"""
The :mod:`imblearn.over_sampling` provides a set of method to
perform over-sampling.
"""

from .random_over_sampler import RandomOverSampler
from .smote import SMOTE
from .adasyn import ADASYN

__all__ = ['RandomOverSampler', 'SMOTE', 'ADASYN']
