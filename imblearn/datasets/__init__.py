"""
The :mod:`imblearn.datasets` provides methods to generate
imbalanced data.
"""

from .benchmark import fetch_benchmark
from .imbalance import make_imbalance

__all__ = ['fetch_benchmark',
           'make_imbalance']
