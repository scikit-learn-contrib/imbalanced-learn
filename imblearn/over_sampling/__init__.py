"""
The :mod:`imblearn.over_sampling` provides a set of method to
perform over-sampling.
"""

from .over_sampler import OverSampler
from .random_over_sampler import RandomOverSampler
from .smote import SMOTE
from .adasyn import ADASYN

__all__ = ['OverSampler',
           'RandomOverSampler',
           'SMOTE',
           'ADASYN']
