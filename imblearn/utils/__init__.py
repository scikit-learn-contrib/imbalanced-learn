"""
The :mod:`imblearn.utils` module includes various utilities.
"""

from .validation import check_neighbors_object
from .validation import check_target_type
from .validation import hash_X_y
from .validation import check_ratio


__all__ = ['check_neighbors_object',
           'check_target_type',
           'hash_X_y',
           'check_ratio']
