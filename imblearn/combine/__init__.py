"""The :mod:`imblearn.combine` provides methods which combine
over-sampling and under-sampling.
"""

from .smote_enn import SMOTEENN
from .smote_tomek import SMOTETomek

__all__ = ['SMOTEENN', 'SMOTETomek']
