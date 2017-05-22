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

    def __init__(self, ratio='auto', random_state=None):
        super(BaseEnsembleSampler, self).__init__(ratio=ratio,
                                                  random_state=random_state,
                                                  sampling_type='ensemble')
