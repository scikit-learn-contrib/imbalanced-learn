"""
Base class for the over-sampling method.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from ..base import BaseSampler


class BaseOverSampler(BaseSampler):
    """Base class for over-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, ratio='auto', random_state=None):
        super(BaseOverSampler, self).__init__(ratio=ratio,
                                              random_state=random_state,
                                              sampling_type='over-sampling')
