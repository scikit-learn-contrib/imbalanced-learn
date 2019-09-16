"""The :mod:`imblearn.combine` provides methods which combine
over-sampling and under-sampling.
"""

from ._smote_enn import SMOTEENN
from ._smote_tomek import SMOTETomek
from ._spider import SPIDER

__all__ = ['SMOTEENN', 'SMOTETomek', 'SPIDER']
