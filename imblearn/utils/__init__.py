"""
The :mod:`imblearn.utils` module includes various utilities.
"""

from ._docstring import Substitution

from .validation import check_neighbors_object
from .validation import check_target_type
from .validation import hash_X_y
from .validation import check_ratio
from .validation import check_sampling_strategy

__all__ = [
    'Substitution', 'check_neighbors_object', 'check_target_type', 'hash_X_y',
    'check_sampling_strategy', 'check_ratio'
]
