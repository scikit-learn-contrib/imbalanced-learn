"""Base class for the preprocess-sampling method."""

# Author: Matthew Eding
# License: MIT

from ...base import BaseSampler


class BasePreprocessSampler(BaseSampler):
    """Base class for preprocess-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """
    _sampling_type = 'preprocess-sampling'

    _sampling_strategy_docstring = \
        """sampling_strategy : str, list or callable
        Sampling information to sample the data set.

        - When ``str``, specify the class targeted by the resampling. Note the
          the number of samples will not be equal in each. Possible choices
          are:

            ``'minority'``: resample only the minority class;

            ``'not minority'``: resample all classes but the minority class;

            ``'not majority'``: resample all classes but the majority class;

            ``'all'``: resample all classes;

            ``'auto'``: equivalent to ``'not majority'``.

        - When ``list``, the list contains the classes targeted by the
          resampling.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.
        """.rstrip()
