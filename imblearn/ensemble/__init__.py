"""
The :mod:`imblearn.ensemble` module include methods generating
under-sampled subsets combined inside an ensemble.
"""

from imblearn.ensemble._bagging import BalancedBaggingClassifier
from imblearn.ensemble._easy_ensemble import EasyEnsembleClassifier
from imblearn.ensemble._forest import BalancedRandomForestClassifier
from imblearn.ensemble._weight_boosting import RUSBoostClassifier

__all__ = [
    "BalancedBaggingClassifier",
    "BalancedRandomForestClassifier",
    "EasyEnsembleClassifier",
    "RUSBoostClassifier",
]
