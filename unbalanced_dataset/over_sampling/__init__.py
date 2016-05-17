"""
The :mod:`unbalanced_dataset.over_sampling` provides a set of method to
perform over-sampling.
"""

from .over_sampler import OverSampler
from .smote import SMOTE

__all__ = ['OverSampler',
           'SMOTE']
