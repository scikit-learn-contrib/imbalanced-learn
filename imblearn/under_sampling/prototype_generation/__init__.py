"""
The :mod:`imblearn.under_sampling.prototype_generation` submodule
contains the method in which a new samples are generated such that the
dataset become more balanced.
"""

from .cluster_centroids import ClusterCentroids

__all__ = [
    'ClusterCentroids'
]
