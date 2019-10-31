"""
The :mod:`imblearn.under_sampling` provides methods to under-sample
a dataset.
"""

from ._prototype_generation import ClusterCentroids

from ._prototype_selection import RandomUnderSampler
from ._prototype_selection import TomekLinks
from ._prototype_selection import NearMiss
from ._prototype_selection import CondensedNearestNeighbour
from ._prototype_selection import OneSidedSelection
from ._prototype_selection import NeighbourhoodCleaningRule
from ._prototype_selection import EditedNearestNeighbours
from ._prototype_selection import RepeatedEditedNearestNeighbours
from ._prototype_selection import AllKNN
from ._prototype_selection import InstanceHardnessThreshold

__all__ = [
    "ClusterCentroids",
    "RandomUnderSampler",
    "InstanceHardnessThreshold",
    "NearMiss",
    "TomekLinks",
    "EditedNearestNeighbours",
    "RepeatedEditedNearestNeighbours",
    "AllKNN",
    "OneSidedSelection",
    "CondensedNearestNeighbour",
    "NeighbourhoodCleaningRule",
]
