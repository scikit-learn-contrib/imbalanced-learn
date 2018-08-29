"""
The :mod:`imblearn.utils` module includes various utilities.
"""

from ._docstring import Substitution

from ._validation import check_neighbors_object
from ._validation import check_target_type
from ._validation import check_ratio
from ._validation import check_sampling_strategy

__all__ = [
    'Substitution', 'check_neighbors_object', 'check_target_type',
    'check_sampling_strategy', 'check_ratio'
]
