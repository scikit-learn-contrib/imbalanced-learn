"""
The :mod:`imblearn.datasets` provides methods to generate
imbalanced data.
"""

from imblearn.datasets._imbalance import make_imbalance
from imblearn.datasets._zenodo import fetch_datasets

__all__ = ["make_imbalance", "fetch_datasets"]
