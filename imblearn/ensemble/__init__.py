"""
The :mod:`imblearn.ensemble` module include methods generating
under-sampled subsets combined inside an ensemble.
"""

from .easy_ensemble import EasyEnsemble, BalancedBaggingClassifier
from .balance_cascade import BalanceCascade

__all__ = ['EasyEnsemble', 'BalancedBaggingClassifier', 'BalanceCascade']
