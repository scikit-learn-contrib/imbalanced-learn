"""
The :mod:`unbalanced_dataset.ensemble` module include methods generating
under-sampled subsets combined inside an ensemble.
"""

from .easy_ensemble import EasyEnsemble
from .balance_cascade import BalanceCascade

__all__ = ['EasyEnsemble',
           'BalanceCascade']
