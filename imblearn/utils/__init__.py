"""
The :mod:`imblearn.utils` module includes various utilities.
"""

from imblearn.utils._docstring import Substitution
from imblearn.utils._validation import (
    check_neighbors_object,
    check_sampling_strategy,
    check_target_type,
)

__all__ = [
    "check_neighbors_object",
    "check_sampling_strategy",
    "check_target_type",
    "Substitution",
]
