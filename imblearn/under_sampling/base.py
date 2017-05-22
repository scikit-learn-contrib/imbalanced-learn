"""
Base class for the under-sampling method.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from ..base import BaseSampler


class BaseUnderSampler(BaseSampler):
    """Base class for under-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, ratio='auto', random_state=None):
        super(BaseUnderSampler, self).__init__(ratio=ratio,
                                               random_state=random_state,
                                               sampling_type='under-sampling')


class BaseCleaningSampler(BaseSampler):
    """Base class for under-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, ratio='auto', random_state=None):
        super(BaseCleaningSampler, self).__init__(
            ratio=ratio,
            random_state=random_state,
            sampling_type='clean-sampling')
