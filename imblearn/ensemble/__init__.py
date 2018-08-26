"""
The :mod:`imblearn.ensemble` module include methods generating
under-sampled subsets combined inside an ensemble.
"""

from ._easy_ensemble import EasyEnsemble
from ._easy_ensemble import EasyEnsembleClassifier
from ._balance_cascade import BalanceCascade
from ._bagging import BalancedBaggingClassifier
from ._forest import BalancedRandomForestClassifier

__all__ = ['EasyEnsemble', 'EasyEnsembleClassifier',
           'BalancedBaggingClassifier', 'BalanceCascade',
           'BalancedRandomForestClassifier']
