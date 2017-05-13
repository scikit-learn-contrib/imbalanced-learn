"""
The :mod:`imblearn.under_sampling` provides methods to under-sample
a dataset.
"""

from .prototype_generation import ClusterCentroids

from .prototype_selection import RandomUnderSampler
from .prototype_selection import TomekLinks
from .prototype_selection import NearMiss
from .prototype_selection import CondensedNearestNeighbour
from .prototype_selection import OneSidedSelection
from .prototype_selection import NeighbourhoodCleaningRule
from .prototype_selection import EditedNearestNeighbours
from .prototype_selection import RepeatedEditedNearestNeighbours
from .prototype_selection import AllKNN
from .prototype_selection import InstanceHardnessThreshold

__all__ = ['ClusterCentroids',
           'RandomUnderSampler',
           'InstanceHardnessThreshold',
           'NearMiss',
           'TomekLinks',
           'EditedNearestNeighbours',
           'RepeatedEditedNearestNeighbours',
           'AllKNN',
           'OneSidedSelection',
           'CondensedNearestNeighbour',
           'NeighbourhoodCleaningRule']
