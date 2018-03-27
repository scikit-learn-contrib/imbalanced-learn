"""
Base class for the over-sampling method.
"""
# Authors: Bernhard Schlegel <bernhard.schlegel@mytum.de>
# License: MIT


from ..base import BaseSampler


class BaseScaler(BaseSampler):
    """Base class for over-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = 'scaling'
