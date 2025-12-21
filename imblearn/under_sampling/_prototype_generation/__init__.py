"""
The :mod:`imblearn.under_sampling.prototype_generation` submodule contains
methods that generate new samples in order to balance the dataset.
"""

from imblearn.under_sampling._prototype_generation._cluster_centroids import (
    ClusterCentroids,
)

__all__ = ["ClusterCentroids"]
