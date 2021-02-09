"""
The :mod:`imblearn.metrics` module includes score functions, performance
metrics and pairwise metrics and distance computations.
"""

from ._classification import sensitivity_specificity_support
from ._classification import sensitivity_score
from ._classification import specificity_score
from ._classification import geometric_mean_score
from ._classification import make_index_balanced_accuracy
from ._classification import classification_report_imbalanced
from ._classification import macro_averaged_mean_absolute_error

__all__ = [
    "sensitivity_specificity_support",
    "sensitivity_score",
    "specificity_score",
    "geometric_mean_score",
    "make_index_balanced_accuracy",
    "classification_report_imbalanced",
    "macro_averaged_mean_absolute_error",
]
