"""
The :mod:`imblearn.ensemble` module include methods generating
under-sampled subsets combined inside an ensemble.
"""

from .easy_ensemble import EasyEnsemble
from .balance_cascade import BalanceCascade

from .classifier import BalancedBaggingClassifier

__all__ = ['EasyEnsemble', 'BalancedBaggingClassifier', 'BalanceCascade']
