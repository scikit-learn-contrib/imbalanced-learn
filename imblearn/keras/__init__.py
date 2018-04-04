"""The :mod:`imblearn.keras` provides utilities to deal with imbalanced dataset
in keras."""

from .generator import balanced_batch_generator

__all__ = ['balanced_batch_generator']
