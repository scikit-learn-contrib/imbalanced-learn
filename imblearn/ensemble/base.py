"""
Base class for the ensemble method.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from ..base import BaseSampler


class BaseEnsembleSampler(BaseSampler):
    """Base class for ensemble algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = 'ensemble'
