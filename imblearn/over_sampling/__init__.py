"""
The :mod:`imblearn.over_sampling` provides a set of method to
perform over-sampling.
"""

from .adasyn import ADASYN
# from .smote import SMOTE
from .random_over_sampler import RandomOverSampler

__all__ = ['ADASYN', 'RandomOverSampler']#, 'SMOTE']
